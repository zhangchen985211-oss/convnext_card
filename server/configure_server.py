#!/usr/bin/env python3
"""Create server-local configs from templates; does not modify templates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Configure this uploaded route for the server")
    parser.add_argument("--data-root", required=True, help="SPEED+ root containing camera.json and kpts.mat")
    parser.add_argument("--output-root", required=True, help="Writable directory for checkpoints/logs/maps")
    parser.add_argument("--force", action="store_true", help="replace generated server configs")
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="disable timm ImageNet weights (offline mode)")
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"SPEED+ data root is not a directory: {data_root}")
    missing = [str(data_root / name) for name in ("camera.json", "kpts.mat", "synthetic/train.json", "synthetic/validation.json", "synthetic/images") if not (data_root / name).exists()]
    if missing:
        parser.error("missing required source data:\n  " + "\n  ".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)
    specs = [('configs/experiment_convnext_card.json', 'server_configs/source.json', 'route_b_convnext_card'), ('configs_loop_lightbox/loop_convnext_card_niter_0000.json', 'server_configs/lightbox/loop_convnext_card_niter_0000.json', 'route_b_convnext_lightbox'), ('configs_loop_sunlamp/loop_convnext_card_sunlamp_niter_0000.json', 'server_configs/sunlamp/loop_convnext_card_sunlamp_niter_0000.json', 'route_b_convnext_sunlamp')]
    for source_rel, output_rel, result_leaf in specs:
        source = ROOT / source_rel
        destination = ROOT / output_rel
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() and not args.force:
            raise FileExistsError(f"{destination} exists; pass --force to replace generated configs")
        config = json.loads(source.read_text(encoding="utf-8"))
        config["root_dir"] = str(data_root) + os.sep
        config["path_results"] = str(output_root / result_leaf)
        if "project_root" in config:
            config["project_root"] = str(ROOT)
        config["cad_graph"] = str(ROOT / "assets" / "speed_keypoint_graph.json")
        if args.no_pretrained_backbone:
            config["pretrained"] = False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
