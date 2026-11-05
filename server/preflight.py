#!/usr/bin/env python3
"""Check this uploaded route's imports, CUDA device and SPEED+ manifests."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cad_graph import load_kpts_mat


def require_versions() -> None:
    """Fail before a long job when the uploaded route's API contract is absent."""
    from packaging.version import Version

    constraints = {
        "torch": ("2.1", None),
        "torchvision": ("0.16", None),
        "kornia": ("0.6.12", "0.7.0"),
        "timm": ("0.9", "1.0"),
    }
    errors = []
    for package, (minimum, maximum_exclusive) in constraints.items():
        installed = importlib.metadata.version(package)
        version = Version(installed.split("+")[0])
        if version < Version(minimum) or (
            maximum_exclusive is not None and version >= Version(maximum_exclusive)
        ):
            bound = f">={minimum}"
            if maximum_exclusive is not None:
                bound += f",<{maximum_exclusive}"
            errors.append(f"{package} {installed} (requires {bound})")
    if errors:
        raise RuntimeError("Incompatible training environment: " + "; ".join(errors))


def check_split(root: Path, split: str, json_names: tuple[str, ...], scan_images: bool):
    image_dir = root / split / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    for json_name in json_names:
        path = root / split / json_name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path} must be a non-empty JSON array")
        if scan_images:
            missing = [item.get("filename", "<missing filename>") for item in entries
                       if not item.get("filename") or not (image_dir / item["filename"]).is_file()]
            if missing:
                raise FileNotFoundError(f"{path}: {len(missing)} missing images; examples={missing[:5]}")
        print(f"{split}/{json_name}: {len(entries)} records; images={'checked' if scan_images else 'not scanned'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-only", action="store_true", help="skip Lightbox/Sunlamp checks")
    parser.add_argument("--skip-image-scan", action="store_true", help="only check manifests and directories")
    parser.add_argument("--allow-cpu", action="store_true", help="do not fail when CUDA is unavailable")
    args = parser.parse_args()
    data = Path(args.data_root).expanduser().resolve()
    for package in ("torch", "torchvision", "kornia", "cv2", "scipy", "tqdm", "timm"):
        importlib.import_module(package)
    require_versions()
    importlib.import_module("src.pose_utils")
    importlib.import_module("train")
    from models.poseconvnext import get_convnext_card
    import torch
    print(f"Python: {sys.version.split()[0]}; PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            print(f"CUDA {index}: {torch.cuda.get_device_name(index)}; free={free / 2**30:.1f} GiB / total={total / 2**30:.1f} GiB")
    elif not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; use the configured CUDA-enabled environment or pass --allow-cpu for a code-only check")
    if not data.is_dir():
        raise FileNotFoundError(data)
    for name in ("camera.json", "kpts.mat"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    json.loads((data / "camera.json").read_text(encoding="utf-8"))
    points = load_kpts_mat(data / "kpts.mat")
    # Importing the model alone does not catch an old timm that lacks the
    # configured ConvNeXt feature API. Build a CPU probe with random weights;
    # this validates the model constructor without allocating a training batch.
    adjacency = torch.eye(points.shape[0])
    distances = torch.zeros_like(adjacency)
    probe = get_convnext_card(
        {
            "backbone": "convnext_tiny",
            "pretrained": False,
            "num_keypoints": 11,
            "decoder_channels": 128,
            "relation_heads": 4,
        },
        adjacency=adjacency,
        distances=distances,
    )
    del probe
    check_split(data, "synthetic", ("train.json", "validation.json"), not args.skip_image_scan)
    if not args.source_only:
        check_split(data, "lightbox", ("test.json",), not args.skip_image_scan)
        check_split(data, "sunlamp", ("test.json",), not args.skip_image_scan)
    print("Preflight passed.")


if __name__ == "__main__":
    main()
