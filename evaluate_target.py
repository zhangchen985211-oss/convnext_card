"""Evaluate a ConvNeXt-CARD checkpoint on public Lightbox/Sunlamp labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run ConvNeXt-CARD inference and score predictions against labeled SPEED+ poses."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--gt-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Evaluate only the first N labeled images; partial runs are not official scores.")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--reprojection-error", type=float, default=None)
    parser.add_argument("--fallback-topks", default=None)
    parser.add_argument("--fallback-reprojection-errors", default=None)
    parser.add_argument("--min-inliers", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        parser.error("batch size must be positive; worker and sample limits must be non-negative")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from src import pose_utils
    from src.checkpoints import load_model_checkpoint
    from src.input_transforms import build_input_transforms
    from src.pose_estimation import estimate_pose
    from src.pose_runtime import import_runtime_dependencies
    from src.run_paths import normalize_config_paths, project_config_arg
    from src.target_evaluation import JsonPoseDataset, load_scaled_camera
    from src.official_scoring import score_prediction_json
    from src.cad_graph import build_speedplus_graph, load_graph, load_kpts_mat
    from models.poseconvnext import get_convnext_card

    os.chdir(PROJECT_ROOT)
    args.cfg = project_config_arg(args.cfg, PROJECT_ROOT)
    fallback_topks = None if args.fallback_topks is None else [
        int(value) for value in args.fallback_topks.split(",") if value.strip()
    ]
    fallback_reprojection_errors = None if args.fallback_reprojection_errors is None else [
        float(value)
        for value in args.fallback_reprojection_errors.split(",")
        if value.strip()
    ]
    config = normalize_config_paths(pose_utils.load_config(args.cfg), PROJECT_ROOT)
    runtime = import_runtime_dependencies()
    if args.device:
        config["device"] = args.device
    device = torch.device(config.get("device", "cuda"))
    dataset_dir = Path(args.dataset_dir)
    dataset_root = Path(args.dataset_root) if args.dataset_root else dataset_dir.parent
    gt_json = Path(args.gt_json) if args.gt_json else dataset_dir / "test.json"
    output_json = Path(args.output_json) if args.output_json else dataset_dir / "convnext_card_predictions.json"
    metrics_json = Path(args.metrics_json) if args.metrics_json else dataset_dir / "convnext_card_metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)

    graph_path = config.get("cad_graph") or config.get("graph_path")
    world = load_kpts_mat(dataset_root / "kpts.mat")
    if graph_path and Path(graph_path).exists():
        _, adjacency, distances = load_graph(graph_path, expected_points=world)
    else:
        adjacency, distances = build_speedplus_graph(world)
    config["pretrained"] = False
    model = get_convnext_card(
        config,
        adjacency=torch.from_numpy(adjacency),
        distances=torch.from_numpy(distances),
    ).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.legacy_output = True
    model.eval()

    transform_input, normalize = build_input_transforms(config, training=False)
    dataset = JsonPoseDataset(dataset_dir, gt_json, transform_input)
    evaluation_dataset = (
        Subset(dataset, range(min(args.max_samples, len(dataset))))
        if args.max_samples
        else dataset
    )
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=bool(config.get("pin_memory", True)),
    )
    k_mat_np, dist_np = load_scaled_camera(dataset_root, config["rows"], config["cols"])
    k_mat = torch.from_numpy(k_mat_np).unsqueeze(0).repeat(args.batch_size, 1, 1).to(device).float()
    dist = torch.from_numpy(dist_np).unsqueeze(0).repeat(args.batch_size, 1).to(device).float()
    predictions = []
    failures = []
    solver_diagnostics = []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=80):
            images = pose_utils.preprocess_for_model(batch["image"].to(device), normalize, k_mat, dist, config)
            outputs = model(images)
            for index in range(images.shape[0]):
                selected, failure_reason = estimate_pose(
                    [{"hm_c": item["hm_c"][index : index + 1]} for item in outputs],
                    images[index : index + 1],
                    world,
                    k_mat_np,
                    config,
                    topk=args.topk,
                    reprojection_error=args.reprojection_error,
                    fallback_topks=fallback_topks,
                    fallback_reprojection_errors=fallback_reprojection_errors,
                    min_inliers=args.min_inliers,
                    runtime=runtime,
                )
                filename = batch["filename"][index]
                if selected is None:
                    failures.append({"filename": filename, "reason": failure_reason})
                    continue
                q = pose_utils.pose_result_to_quaternion(selected, device)
                solver_diagnostics.append({
                    "filename": filename,
                    "strategy": selected["strategy"],
                    "inlier_count": selected["inlier_count"],
                    "support_count": selected["support_count"],
                    "ransac_verified": selected["ransac_verified"],
                    "quality_passed": selected["quality_passed"],
                    "full_mean_reprojection_error": selected["full_mean_reprojection_error"],
                    "refinement": selected.get("refinement"),
                })
                predictions.append(
                    {
                        "filename": filename,
                        "q_vbs2tango_true": q.tolist(),
                        "r_Vo2To_vbs_true": np.squeeze(selected["tvecs"]).tolist(),
                    }
                )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics_path = output_json.with_name(output_json.stem + "_solver_diagnostics.jsonl")
    diagnostics_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in solver_diagnostics), encoding="utf-8")
    metrics = score_prediction_json(predictions, gt_json)
    metrics.update({
        "failed_samples": failures,
        "checkpoint": str(args.checkpoint),
        "evaluated_sample_count": len(evaluation_dataset),
        "evaluation_truncated": len(evaluation_dataset) < len(dataset),
    })
    metrics["solver_diagnostics"] = str(diagnostics_path)
    metrics["strategy_counts"] = {
        strategy: sum(item["strategy"] == strategy for item in solver_diagnostics)
        for strategy in sorted({item["strategy"] for item in solver_diagnostics})
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        key: metrics[key]
        for key in (
            "checkpoint", "gt_count", "pred_count", "matched_count", "official_valid",
            "official_avg_total_score", "evaluated_sample_count", "evaluation_truncated",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
