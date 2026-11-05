"""Fail-fast UDA scheduling and explicit round-boundary recovery."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.run_paths import (
    assert_no_downstream_results,
    config_output_dir,
    normalize_config_paths,
    parse_config_iteration,
    project_config_arg,
    replace_config_iteration,
    validate_training_complete,
)


def run_uda_iterations(args, base_dir, project_root, route_dir):
    root = Path(project_root).expanduser().resolve()
    base_dir, route_dir = Path(base_dir).expanduser(), Path(route_dir).expanduser()
    base_dir = (base_dir if base_dir.is_absolute() else root / base_dir).resolve()
    route_dir = (route_dir if route_dir.is_absolute() else root / route_dir).resolve()
    start = int(getattr(args, "start_iteration", 1))
    restart = bool(getattr(args, "restart_iteration", False))
    if Path(args.base_file).name != args.base_file or parse_config_iteration(args.base_file) != 0:
        raise ValueError("--base-file must be a filename with one niter_0000 round token")
    if not 1 <= start < args.niters <= 10000:
        raise ValueError("require 1 <= --start-iteration < --niters; niters includes round 0000")
    template_arg = project_config_arg(base_dir / args.base_file, root)
    template = root / template_arg
    config = normalize_config_paths(json.loads(template.read_text(encoding="utf-8")), root)
    if not config.get("isloop") or config.get("split_submission") not in ("lightbox", "sunlamp"):
        raise ValueError("UDA requires isloop=true and a lightbox/sunlamp split")
    for iteration in range(start, args.niters):
        current = Path(replace_config_iteration(template, iteration))
        previous = Path(replace_config_iteration(template, iteration - 1))
        previous_arg = project_config_arg(previous, root)
        previous_config = normalize_config_paths(json.loads(previous.read_text(encoding="utf-8")), root)
        for key in ("root_dir", "path_results", "rows", "cols", "split_submission", "image_augmentation"):
            if previous_config.get(key) != config.get(key):
                raise ValueError(f"Previous round and template disagree on {key}; use a separate experiment")
        previous_output = config_output_dir(previous_config, previous_arg, root)
        if not (previous_output / "ckpt" / "init.pth").is_file():
            raise FileNotFoundError(f"Missing previous checkpoint: {previous_output / 'ckpt' / 'init.pth'}")
        if iteration > 1:
            validate_training_complete(previous_output, previous_config, previous_arg, root)
        # Canonicalize the not-yet-created path without using an absolute output component.
        current_arg = current.resolve().relative_to(root.resolve()).as_posix()
        destination = config_output_dir(config, current_arg, root)
        # Check before copying configs, clearing markers, or mutating shared labels/maps.
        assert_no_downstream_results(current, config, root)
        has_results = destination.exists() and any(destination.iterdir())
        if has_results and not (restart and iteration == start):
            raise FileExistsError(f"{destination} already contains results; use --start-iteration for the next round, "
                                  "or --restart-iteration to retrain this partial round from its predecessor")
        if current.exists() and current.read_bytes() != template.read_bytes() and not (restart and iteration == start):
            raise FileExistsError(f"{current} differs from the template; preserve it in a separate experiment")
        shutil.copyfile(template, current)
        marker = destination / "training_complete.json"
        if marker.exists():
            marker.unlink()
        subprocess.run([sys.executable, str(route_dir / "generate_pseudo_labels.py"), "--cfg", previous_arg], cwd=root, check=True)
        from src.pose_utils import format_and_copy_json
        format_and_copy_json(previous_arg, project_root=root)
        subprocess.run([sys.executable, str(root / "create_maps.py"), "--cfg", current_arg], cwd=root, check=True)
        subprocess.run([sys.executable, str(route_dir / "train.py"), "--cfg", current_arg], cwd=root, check=True)
        if not (destination / "training_complete.json").is_file():
            raise RuntimeError(f"Trainer exited without a completion marker: {destination}")
        validate_training_complete(destination, config, current_arg, root)
