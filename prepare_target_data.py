"""Prepare a separate unlabeled SPEED+ target training view."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.target_data import prepare_target_view  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("lightbox", "sunlamp"), required=True)
    parser.add_argument("--copy-images", action="store_true", help="copy instead of symlink")
    args = parser.parse_args()
    target = prepare_target_view(args.data_root, args.split, args.copy_images)
    print(f"Prepared {target}; ground-truth target poses remain outside the training view")


if __name__ == "__main__":
    main()
