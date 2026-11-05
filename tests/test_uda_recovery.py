import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

from src import uda_scheduler
from src.run_paths import config_output_dir, mark_training_complete


def _write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")


def _run_args(start, niters, restart=False, base_file="loop_niter_0000_v2.json"):
    return SimpleNamespace(
        base_file=base_file,
        start_iteration=start,
        niters=niters,
        restart_iteration=restart,
    )


def test_scheduler_fresh_run_and_round_boundary_recovery(tmp_path, monkeypatch):
    root = tmp_path / "project"
    caller = tmp_path / "caller"
    base_dir = root / "configs"
    caller.mkdir()
    root.mkdir()
    config = {
        "root_dir": "data",
        "path_results": "results",
        "isloop": True,
        "split_submission": "lightbox",
        "rows": 512,
        "cols": 512,
        "image_augmentation": {},
        "total_epochs": 2,
    }
    template = base_dir / "loop_niter_0000_v2.json"
    _write_config(template, config)
    seed = config_output_dir(config, "configs/loop_niter_0000_v2.json", root) / "ckpt/init.pth"
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"seed-state-dict")
    monkeypatch.chdir(caller)

    def fake_format(previous_arg, project_root=None):
        assert Path(config["root_dir"]).is_absolute() is False
        assert Path.cwd() == caller
        return None

    monkeypatch.setitem(sys.modules, "src.pose_utils", types.SimpleNamespace(format_and_copy_json=fake_format))

    def fake_subprocess_run(command, cwd, check):
        assert Path(cwd) == root
        assert check is True
        script_name = Path(command[1]).name
        if script_name == "train.py":
            config_arg = command[-1]
            iteration = int(Path(config_arg).stem.split("niter_")[-1][:4])
            current_config = json.loads((root / config_arg).read_text(encoding="utf-8"))
            output = config_output_dir(current_config, config_arg, root)
            current_checkpoint = output / "ckpt/init.pth"
            current_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            current_checkpoint.write_bytes(f"round-{iteration}-state-dict".encode())
            parent_arg = f"configs/loop_niter_{iteration - 1:04d}_v2.json"
            parent_config = json.loads((root / parent_arg).read_text(encoding="utf-8"))
            parent_checkpoint = config_output_dir(parent_config, parent_arg, root) / "ckpt/init.pth"
            mark_training_complete(
                output,
                current_config,
                config_arg,
                parent_checkpoint=parent_checkpoint,
                project_root=root,
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(uda_scheduler.subprocess, "run", fake_subprocess_run)
    uda_scheduler.run_uda_iterations(_run_args(1, 3), base_dir, root, root)
    uda_scheduler.run_uda_iterations(_run_args(3, 4), base_dir, root, root)

    for iteration in range(1, 4):
        config_arg = f"configs/loop_niter_{iteration:04d}_v2.json"
        round_output = config_output_dir(config, config_arg, root)
        assert (round_output / "training_complete.json").is_file()
        assert (round_output / "ckpt/init.pth").is_file()


def test_retraining_completed_ancestor_refuses_existing_descendants_before_writes(tmp_path, monkeypatch):
    root = tmp_path / "project"
    base_dir = root / "configs"
    base_dir.mkdir(parents=True)
    config = {
        "root_dir": "data",
        "path_results": "results",
        "isloop": True,
        "split_submission": "lightbox",
        "rows": 512,
        "cols": 512,
        "image_augmentation": {},
        "total_epochs": 2,
    }
    template = base_dir / "loop_niter_0000.json"
    _write_config(template, config)
    _write_config(base_dir / "loop_niter_0001.json", {**config, "sentinel": "preserve-me"})
    seed = config_output_dir(config, "configs/loop_niter_0000.json", root) / "ckpt/init.pth"
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"seed-state-dict")
    current_output = config_output_dir(config, "configs/loop_niter_0001.json", root)
    (current_output / "ckpt").mkdir(parents=True)
    (current_output / "ckpt/init.pth").write_bytes(b"old-round-one")
    (current_output / "training_complete.json").write_text("{}", encoding="utf-8")
    descendant_output = config_output_dir(config, "configs/loop_niter_0002.json", root)
    descendant_output.mkdir(parents=True)
    (descendant_output / "result.txt").write_text("preserve", encoding="utf-8")
    before_config = (base_dir / "loop_niter_0001.json").read_bytes()
    before_marker = (current_output / "training_complete.json").read_bytes()
    ran = []
    monkeypatch.setattr(uda_scheduler.subprocess, "run", lambda *a, **kw: ran.append(a))

    with pytest.raises(FileExistsError, match="downstream results already exist"):
        uda_scheduler.run_uda_iterations(_run_args(1, 2, restart=True, base_file=template.name), base_dir, root, root)

    assert ran == []
    assert (base_dir / "loop_niter_0001.json").read_bytes() == before_config
    assert (current_output / "training_complete.json").read_bytes() == before_marker
    assert (descendant_output / "result.txt").read_text(encoding="utf-8") == "preserve"


def test_lock_resolves_relative_root_against_project_not_caller_cwd(tmp_path, monkeypatch):
    from src.pseudo_cache_lock import target_pseudo_cache_lock

    root = tmp_path / "project"
    caller = tmp_path / "caller"
    root.mkdir()
    caller.mkdir()
    config_path = root / "config.json"
    _write_config(config_path, {"root_dir": "data", "split_submission": "lightbox"})
    (root / "data").mkdir()
    monkeypatch.chdir(caller)

    with target_pseudo_cache_lock(config_path, project_root=root) as lock_path:
        assert lock_path == root / "data/.uda_cache_lightbox.lock"
        assert lock_path.is_file()
    assert not lock_path.exists()
