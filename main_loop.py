"""Route B UDA scheduler for ConvNeXt-CARD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Route B ConvNeXt UDA loop")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-file", required=True)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument("--start-iteration", type=int, default=1, help="first round to train (round-boundary recovery)")
    parser.add_argument("--restart-iteration", action="store_true", help="explicitly retrain an existing first selected round")
    args = parser.parse_args()
    base_dir = Path(args.base_config).expanduser()
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    base_dir = base_dir.resolve()
    from src.pseudo_cache_lock import target_pseudo_cache_lock
    from src.run_paths import parse_config_iteration

    if Path(args.base_file).name != args.base_file or parse_config_iteration(args.base_file) != 0:
        parser.error("--base-file must be a filename with one niter_0000 round token")

    with target_pseudo_cache_lock(base_dir / args.base_file, PROJECT_ROOT):
        _run_iterations(args, base_dir)


def _run_iterations(args, base_dir):
    from src.uda_scheduler import run_uda_iterations
    run_uda_iterations(args, base_dir, PROJECT_ROOT, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
