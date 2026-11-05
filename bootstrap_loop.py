"""Seed iteration 0000 with a selected synthetic-trained ConvNeXt checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_paths import (  # noqa: E402
    assert_no_downstream_results,
    config_output_dir,
    normalize_config_paths,
    parse_config_iteration,
    project_config_arg,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="iteration-0000 route config")
    parser.add_argument("--checkpoint", required=True, help="selected synthetic checkpoint")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.cfg = project_config_arg(args.cfg, PROJECT_ROOT)
    config_path = PROJECT_ROOT / args.cfg
    config = normalize_config_paths(json.loads(config_path.read_text(encoding="utf-8")), PROJECT_ROOT)
    if not config.get("isloop") or parse_config_iteration(config_path) != 0:
        raise ValueError("bootstrap requires an isloop=true niter_0000 config")
    destination = config_output_dir(config, args.cfg, PROJECT_ROOT) / "ckpt" / "init.pth"
    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} exists; pass --force to replace it")
    assert_no_downstream_results(config_path, config, PROJECT_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Seed checkpoint: {destination}")


if __name__ == "__main__":
    main()
