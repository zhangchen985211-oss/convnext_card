import json
from pathlib import Path

import pytest

from src.run_paths import (
    config_output_dir,
    mark_training_complete,
    normalize_config_paths,
    parse_config_iteration,
    replace_config_iteration,
    validate_training_complete,
)


def test_normalize_paths_is_copy_and_uses_project_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    caller = tmp_path / "caller"
    project.mkdir()
    caller.mkdir()
    monkeypatch.chdir(caller)
    config = {
        "root_dir": "data",
        "path_results": "results",
        "cad_graph": "assets/graph.json",
        "path_pretrain": "~/weights/source.pth",
        "project_root": ".",
        "empty": "",
    }

    normalized = normalize_config_paths(config, project)

    assert normalized is not config
    assert config["root_dir"] == "data"
    assert normalized["root_dir"] == str(project / "data") + "/"
    assert normalized["path_results"] == str(project / "results")
    assert normalized["cad_graph"] == str(project / "assets/graph.json")
    assert normalized["path_pretrain"] == str(Path.home() / "weights/source.pth")
    assert normalized["project_root"] == str(project)
    assert normalized["empty"] == ""
    assert config_output_dir(config, "cfg/loop_niter_0000.json", project) == project / "results/cfg/loop_niter_0000.json"


@pytest.mark.parametrize("value", ["<DATA_ROOT>/", "<OUTPUT_ROOT>/results", "<PROJECT_ROOT>/assets/graph.json"])
def test_normalize_rejects_unresolved_path_placeholders(tmp_path, value):
    with pytest.raises(ValueError, match="unresolved path placeholder"):
        normalize_config_paths({"path_results": value}, tmp_path)


def test_iteration_helpers_preserve_suffix_and_reject_ambiguity():
    assert parse_config_iteration("cfg/loop_niter_0001_v2.json") == 1
    assert replace_config_iteration("cfg/loop_niter_0001_v2.json", 23) == "cfg/loop_niter_0023_v2.json"
    assert parse_config_iteration("loop_niter_0000.json") == 0
    with pytest.raises(ValueError, match="unambiguous"):
        parse_config_iteration("loop_niter_0000_copy_niter_0000.json")
    with pytest.raises(ValueError, match="unambiguous"):
        parse_config_iteration("loop_niter_00000.json")


def test_completion_marker_verifies_checkpoint_lineage(tmp_path):
    root = tmp_path / "project"
    config_dir = root / "cfg"
    config_dir.mkdir(parents=True)
    config = {
        "root_dir": "data",
        "path_results": "results",
        "isloop": True,
        "split_submission": "lightbox",
        "total_epochs": 2,
    }
    previous_arg = "cfg/loop_niter_0000.json"
    current_arg = "cfg/loop_niter_0001_v2.json"
    (config_dir / "loop_niter_0000.json").write_text(json.dumps(config), encoding="utf-8")
    (config_dir / "loop_niter_0000_v2.json").write_text(json.dumps(config), encoding="utf-8")
    (config_dir / "loop_niter_0001_v2.json").write_text(json.dumps(config), encoding="utf-8")
    previous_arg = "cfg/loop_niter_0000_v2.json"
    previous_output = config_output_dir(config, previous_arg, root)
    current_output = config_output_dir(config, current_arg, root)
    (previous_output / "ckpt").mkdir(parents=True)
    (current_output / "ckpt").mkdir(parents=True)
    (previous_output / "ckpt/init.pth").write_bytes(b"seed-state-dict")
    (current_output / "ckpt/init.pth").write_bytes(b"round-one-state-dict")

    mark_training_complete(current_output, config, current_arg, project_root=root)
    payload = json.loads((current_output / "training_complete.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert "parent_checkpoint_sha256" in payload
    validate_training_complete(current_output, config, current_arg, root)

    (previous_output / "ckpt/init.pth").write_bytes(b"replaced-seed-state-dict")
    with pytest.raises(RuntimeError, match="Parent checkpoint changed"):
        validate_training_complete(current_output, config, current_arg, root)


def test_legacy_completion_marker_is_not_accepted_as_verified_parent(tmp_path):
    root = tmp_path / "project"
    config_dir = root / "cfg"
    config_dir.mkdir(parents=True)
    config = {"root_dir": "data", "path_results": "results", "isloop": True, "total_epochs": 2}
    config_arg = "cfg/loop_niter_0001.json"
    (config_dir / "loop_niter_0001.json").write_text(json.dumps(config), encoding="utf-8")
    output = config_output_dir(config, config_arg, root)
    (output / "training_complete.json").parent.mkdir(parents=True)
    (output / "training_complete.json").write_text(json.dumps({"config": config_arg}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no verified UDA lineage"):
        validate_training_complete(output, config, config_arg, root)
