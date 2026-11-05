"""Prepare an unlabeled target view without importing training dependencies."""

import json
import shutil
from pathlib import Path


def prepare_target_view(data_root, split, copy_images=False):
    if split not in ("lightbox", "sunlamp"):
        raise ValueError("Target split must be lightbox or sunlamp")
    root = Path(data_root).expanduser().resolve()
    source, target = root / split, root / f"{split}_train"
    source_images, target_images = source / "images", target / "images"
    if not source_images.is_dir() or not (source / "test.json").is_file():
        raise FileNotFoundError(f"Expected {source_images} and {source / 'test.json'}")
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "image_source.json"
    if target_images.is_symlink():
        if target_images.resolve() != source_images.resolve():
            raise FileExistsError(f"{target_images} points to a different image directory")
    elif target_images.exists():
        if not marker.is_file():
            raise FileExistsError(f"{target_images} exists without an image_source.json manifest; inspect it manually")
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest != {"mode": "copy", "source": str(source_images)}:
            raise FileExistsError(f"{marker} describes a different source or preparation mode")
        expected = {p.relative_to(source_images): p.stat().st_size for p in source_images.rglob("*") if p.is_file()}
        actual = {p.relative_to(target_images): p.stat().st_size for p in target_images.rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError("Copied target images are incomplete or changed; inspect the directories before retrying")
    elif copy_images:
        shutil.copytree(source_images, target_images)
        marker.write_text(json.dumps({"mode": "copy", "source": str(source_images)}, indent=2), encoding="utf-8")
    else:
        target_images.symlink_to(source_images, target_is_directory=True)
    (target / "kptsmap").mkdir(exist_ok=True)
    # Public target poses stay in the original test.json for frozen evaluation.
    # Pseudo-label generation alone creates target_train/train.json.
    return target
