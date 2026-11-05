"""Generate Route B pseudo labels using the shared staged quality protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loaders import speedplus_segmentation_precomputed as speedplus  # noqa: E402
from src import pose_utils  # noqa: E402
from src.run_paths import config_output_dir, normalize_config_paths, project_config_arg  # noqa: E402
from src.checkpoints import load_model_checkpoint  # noqa: E402

from models.poseconvnext import get_convnext_card  # noqa: E402
from src.cad_graph import (  # noqa: E402
    build_speedplus_graph,
    load_graph,
    load_kpts_mat,
    save_graph,
)
from src.input_transforms import build_input_transforms  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Route B ConvNeXt pseudo labels")
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    args.cfg = project_config_arg(args.cfg, PROJECT_ROOT)
    config = normalize_config_paths(pose_utils.load_config(args.cfg), PROJECT_ROOT)
    device = config["device"]
    points = load_kpts_mat(Path(config["root_dir"]) / "kpts.mat")
    graph_path = config.get("cad_graph") or config.get("graph_path")
    if graph_path and Path(graph_path).exists():
        _, adjacency, distances = load_graph(graph_path, expected_points=points)
    else:
        adjacency, distances = build_speedplus_graph(points)
        if graph_path:
            save_graph(graph_path, points, adjacency, distances)
    # The checkpoint supplies all encoder weights during inference.
    config["pretrained"] = False
    model = get_convnext_card(
        config,
        adjacency=torch.from_numpy(adjacency),
        distances=torch.from_numpy(distances),
    ).to(device)
    output_dir = config_output_dir(config, args.cfg, PROJECT_ROOT)
    checkpoint = output_dir / "ckpt" / "init.pth"
    load_model_checkpoint(model, checkpoint, device)
    model.legacy_output = True
    model.eval()
    transform_input, aug = build_input_transforms(config, training=False)
    dataset = speedplus.PyTorchSatellitePoseEstimationDataset(
        split=config["split_submission"],
        speed_root=config["root_dir"],
        transform_input=transform_input,
        config=config,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get("batch_size_test", 1),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        drop_last=False,
        pin_memory=bool(config.get("pin_memory", True)),
    )
    k_mat = pose_utils.get_kmat_scaled(config, device)
    world = pose_utils.get_world_kpts(config, device)
    dist = pose_utils.get_coefs(config, device)
    path_json = str(output_dir / "train.json")
    with torch.no_grad():
        pose_utils.generate_json_loop(loader, aug, model, world, k_mat, dist, device, config, path_json)


if __name__ == "__main__":
    main()
