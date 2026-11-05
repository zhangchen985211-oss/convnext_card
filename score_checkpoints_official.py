"""Batch score ConvNeXt-CARD checkpoints on labeled SPEED+ target splits."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.official_scoring import (  # noqa: E402
    OFFICIAL_ROTATION_THRESHOLD_DEG,
    OFFICIAL_ROTATION_THRESHOLD_RAD,
    OFFICIAL_TRANSLATION_THRESHOLD,
    run_self_test,
)
from src.run_paths import config_output_dir, normalize_config_paths, project_config_arg  # noqa: E402


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _model_id(config_path: Path, config_dir: Path, checkpoint_path: Path) -> str:
    try:
        config_name = config_path.relative_to(config_dir).with_suffix("").as_posix()
    except ValueError:
        config_name = config_path.stem
    checkpoint_name = checkpoint_path.stem
    return f"{config_name.replace('/', '__')}__{checkpoint_name}"


def discover_checkpoints(
    results_dir: Path,
    config_dir: Path,
    split: str,
    checkpoint_pattern: str = "init.pth",
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Match each target-split config to weights in its configured output tree."""
    results_dir = results_dir.resolve()
    config_dir = config_dir.resolve()
    project_root = project_root.resolve()
    checkpoints = []
    config_paths = sorted(config_dir.glob("**/*.json"))
    for config_path in config_paths:
        try:
            config_arg = project_config_arg(config_path, project_root)
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            if raw_config.get("split_submission") != split:
                continue
            config = normalize_config_paths(raw_config, project_root)
            run_dir = config_output_dir(config, config_arg, project_root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Skipping config {config_path}: {error}", file=sys.stderr)
            continue
        try:
            run_dir.relative_to(results_dir)
        except ValueError:
            continue
        checkpoint_dir = run_dir / "ckpt"
        for checkpoint_path in sorted(checkpoint_dir.glob(checkpoint_pattern)):
            if not checkpoint_path.is_file():
                continue
            iteration_match = re.search(r"niter_(\d{4})(?!\d)", config_path.name)
            checkpoints.append({
                "niter": int(iteration_match.group(1)) if iteration_match else None,
                "model_id": _model_id(config_path, config_dir, checkpoint_path),
                "checkpoint_path": checkpoint_path.resolve(),
                "config_path": config_path.resolve(),
                "config_arg": config_arg,
            })
    checkpoints.sort(key=lambda item: (
        item["niter"] if item["niter"] is not None else math.inf,
        item["model_id"],
    ))
    return checkpoints


def result_from_metrics(
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    split: str,
    dataset_dir: Path,
    gt_json_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_count = int(metrics.get("gt_count", 0))
    solved_count = int(metrics.get("matched_count", 0))
    evaluated_count = int(metrics.get("evaluated_sample_count", gt_count))
    failed_count = max(0, gt_count - solved_count)
    official_values = [
        metrics.get(f"official_avg_{name}_score")
        for name in ("translation", "rotation", "total")
    ]
    official_valid = (
        bool(metrics.get("official_valid"))
        and not bool(metrics.get("evaluation_truncated"))
        and gt_count > 0
        and solved_count == gt_count
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in official_values
        )
    )
    result = {
        "niter": checkpoint["niter"],
        "model_id": checkpoint["model_id"],
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "config_path": str(checkpoint["config_path"]),
        "status": "ok" if official_valid else (
            "partial" if metrics.get("evaluation_truncated") else "pose_failures"
        ),
        "error": None,
        "official_valid": official_valid,
        "translation_score": metrics.get("official_avg_translation_score") if official_valid else None,
        "rotation_score": metrics.get("official_avg_rotation_score") if official_valid else None,
        "total_score": metrics.get("official_avg_total_score") if official_valid else None,
        "solved_translation_score": metrics.get("solved_only_official_avg_translation_score"),
        "solved_rotation_score": metrics.get("solved_only_official_avg_rotation_score"),
        "solved_total_score": metrics.get("solved_only_official_avg_total_score"),
        "sample_count": gt_count,
        "evaluated_count": evaluated_count,
        "solved_count": solved_count,
        "failed_count": failed_count,
        "success_rate": float(solved_count / gt_count) if gt_count else 0.0,
        "split": split,
        "dataset_dir": str(dataset_dir),
        "gt_json_path": str(gt_json_path or dataset_dir / "test.json"),
        "strategy_counts": metrics.get("strategy_counts", {}),
        "solver_diagnostics": metrics.get("solver_diagnostics"),
    }
    if not official_valid and result["status"] == "pose_failures":
        result["error"] = f"{failed_count} of {gt_count} labeled samples have no valid prediction"
    failures = [
        {
            "niter": checkpoint["niter"],
            "model_id": checkpoint["model_id"],
            "checkpoint_path": str(checkpoint["checkpoint_path"]),
            **failure,
        }
        for failure in metrics.get("failed_samples", [])
    ]
    return result, failures


def sort_results_for_ranking(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, float, int, float, float, str]:
        valid = bool(item.get("official_valid"))
        total = item.get("total_score")
        solved_total = item.get("solved_total_score")
        niter = item.get("niter")
        return (
            0 if valid else 1,
            float(total) if total is not None else math.inf,
            int(item["failed_count"]) if item.get("failed_count") is not None else 10**9,
            float(solved_total) if solved_total is not None else math.inf,
            float(niter) if niter is not None else math.inf,
            str(item.get("model_id", "")),
        )

    return sorted(results, key=key)


def write_curve_json(path: Path, results: list[dict[str, Any]]) -> None:
    payload = {
        "x": [item["model_id"] for item in results],
        "translation_score": [item.get("translation_score") for item in results],
        "rotation_score": [item.get("rotation_score") for item in results],
        "total_score": [item.get("total_score") for item in results],
        "solved_translation_score": [item.get("solved_translation_score") for item in results],
        "solved_rotation_score": [item.get("solved_rotation_score") for item in results],
        "official_valid": [item.get("official_valid") for item in results],
        "solved_total_score": [item.get("solved_total_score") for item in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_scores(path: Path, results: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-score-convnext")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["model_id"] + ("" if item.get("official_valid") else "*") for item in results]
    x_values = np.arange(len(labels))
    series = (
        ("translation_score", "solved_translation_score", "translation_score", "s"),
        ("rotation_score", "solved_rotation_score", "rotation_score", "o"),
        ("total_score", "solved_total_score", "total_score", "^"),
    )
    figure, axis = plt.subplots(figsize=(max(10.0, len(labels) * 0.58), 6.0))
    for official_key, solved_key, label, marker in series:
        values = [
            item.get(official_key) if item.get("official_valid") else np.nan
            for item in results
        ]
        line = axis.plot(x_values, values, marker=marker, label=label)[0]
        invalid_x = []
        invalid_y = []
        for index, item in enumerate(results):
            value = item.get(solved_key)
            if not item.get("official_valid") and value is not None:
                invalid_x.append(index)
                invalid_y.append(float(value))
        if invalid_x:
            axis.scatter(
                invalid_x, invalid_y, facecolors="none", edgecolors=line.get_color(),
                linewidths=1.7, marker=marker,
            )
    axis.set_xticks(x_values, labels, rotation=45, ha="right")
    axis.set_xlabel("checkpoint")
    axis.set_ylabel("score")
    axis.set_title("Official SPEED+ score by ConvNeXt-CARD checkpoint (* = partial diagnostic)")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=False, default="")
    parser.add_argument("--config-dir", required=False, default="")
    parser.add_argument("--split", choices=("lightbox", "sunlamp"), default="sunlamp")
    parser.add_argument("--dataset-dir", required=False, default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--gt-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--checkpoint-pattern", default="init.pth",
                        help="Filename glob under each matching config's ckpt directory.")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--reprojection-error", type=float, default=None)
    parser.add_argument("--fallback-topks", default=None)
    parser.add_argument("--fallback-reprojection-errors", default=None)
    parser.add_argument("--min-inliers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit inference to the first N images; truncated runs are not official scores.")
    parser.add_argument("--limit-checkpoints", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if not args.results_dir or not args.config_dir or not args.dataset_dir:
        raise ValueError("--results-dir, --config-dir, and --dataset-dir are required")
    if args.max_samples < 0 or args.limit_checkpoints < 0 or args.num_workers < 0 or args.batch_size < 1:
        raise ValueError("batch size must be positive; worker, sample, and checkpoint limits must be non-negative")
    if not args.checkpoint_pattern or Path(args.checkpoint_pattern).name != args.checkpoint_pattern:
        raise ValueError("--checkpoint-pattern must be a filename glob, without directories")

    results_dir = _resolve_project_path(args.results_dir, PROJECT_ROOT)
    config_dir = _resolve_project_path(args.config_dir, PROJECT_ROOT)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else dataset_dir.parent
    gt_json = Path(args.gt_json).expanduser().resolve() if args.gt_json else dataset_dir / "test.json"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_dir / f"official_score_{args.split}"
    )
    checkpoints = discover_checkpoints(
        results_dir, config_dir, args.split, args.checkpoint_pattern, PROJECT_ROOT
    )
    total_checkpoints_found = len(checkpoints)
    if args.limit_checkpoints:
        checkpoints = checkpoints[:args.limit_checkpoints]
    if not checkpoints:
        raise RuntimeError(f"No {args.split} checkpoints matching {args.checkpoint_pattern!r} under {results_dir}")

    all_results = []
    all_failures = []
    evaluator = PROJECT_ROOT / "evaluate_target.py"
    forwarded = [
        ("--device", args.device),
        ("--topk", args.topk),
        ("--reprojection-error", args.reprojection_error),
        ("--fallback-topks", args.fallback_topks),
        ("--fallback-reprojection-errors", args.fallback_reprojection_errors),
        ("--min-inliers", args.min_inliers),
    ]
    for index, checkpoint in enumerate(checkpoints, start=1):
        model_output_dir = output_dir / checkpoint["model_id"]
        prediction_json = model_output_dir / "predictions.json"
        metrics_json = model_output_dir / "metrics.json"
        command = [
            sys.executable, str(evaluator),
            "--cfg", checkpoint["config_arg"],
            "--checkpoint", str(checkpoint["checkpoint_path"]),
            "--dataset-dir", str(dataset_dir),
            "--dataset-root", str(dataset_root),
            "--gt-json", str(gt_json),
            "--output-json", str(prediction_json),
            "--metrics-json", str(metrics_json),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--max-samples", str(args.max_samples),
        ]
        for option, value in forwarded:
            if value is not None and value != "":
                command.extend((option, str(value)))
        print(f"\n[{index}/{len(checkpoints)}] evaluating {checkpoint['model_id']}", flush=True)
        try:
            model_output_dir.mkdir(parents=True, exist_ok=True)
            inference_log = model_output_dir / "inference.log"
            with inference_log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command, cwd=PROJECT_ROOT, check=False,
                    stdout=stream, stderr=subprocess.STDOUT,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"evaluate_target.py exited with status {completed.returncode}; see {inference_log}"
                )
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
            result, failures = result_from_metrics(checkpoint, metrics, args.split, dataset_dir, gt_json)
            result["predictions_json"] = str(prediction_json)
            result["metrics_json"] = str(metrics_json)
            result["inference_log"] = str(inference_log)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            result = {
                "niter": checkpoint["niter"],
                "model_id": checkpoint["model_id"],
                "checkpoint_path": str(checkpoint["checkpoint_path"]),
                "config_path": str(checkpoint["config_path"]),
                "status": "error",
                "error": str(error),
                "official_valid": False,
                "total_score": None,
                "failed_count": None,
            }
            failures = [{
                "niter": checkpoint["niter"],
                "model_id": checkpoint["model_id"],
                "checkpoint_path": str(checkpoint["checkpoint_path"]),
                "stage": "evaluation",
                "reason": str(error),
            }]
        all_results.append(result)
        all_failures.extend(failures)
        print(
            f"{checkpoint['model_id']}: valid={result.get('official_valid')} "
            f"total={result.get('total_score')} failed={result.get('failed_count')}",
            flush=True,
        )

    generated_at = datetime.now().isoformat(timespec="seconds")
    sorted_results = sort_results_for_ranking(all_results)
    results_by_iteration = sorted(
        all_results,
        key=lambda item: (
            item.get("niter") if item.get("niter") is not None else math.inf,
            item.get("model_id", ""),
        ),
    )
    summary = {
        "generated_at": generated_at,
        "results_dir": str(results_dir),
        "config_dir": str(config_dir),
        "split": args.split,
        "output_dir": str(output_dir),
        "total_checkpoints_found": total_checkpoints_found,
        "total_checkpoints_scored": len(all_results),
        "max_samples": args.max_samples,
        "formula": {
            "reference_url": "https://kelvins.esa.int/pose-estimation-2021/scoring/",
            "translation_score": "||t_pred - t_gt|| / ||t_gt||",
            "rotation_score": "2 * arccos(abs(dot(normalize(q_pred), normalize(q_gt))))",
            "rotation_score_unit": "radians",
            "translation_threshold": OFFICIAL_TRANSLATION_THRESHOLD,
            "rotation_threshold_degrees": OFFICIAL_ROTATION_THRESHOLD_DEG,
            "rotation_threshold_radians": OFFICIAL_ROTATION_THRESHOLD_RAD,
            "thresholding": "component-wise",
        },
        "results": sorted_results,
    }
    failures_payload = {
        "generated_at": generated_at,
        "split": args.split,
        "failure_count": len(all_failures),
        "failures": all_failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_scores_sorted.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "failures.json").write_text(
        json.dumps(failures_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_curve_json(output_dir / "official_scores_curve.json", results_by_iteration)
    plot_scores(output_dir / "official_scores_curve.png", results_by_iteration)
    print(f"\nScores:   {output_dir / 'official_scores_sorted.json'}")
    print(f"Failures: {output_dir / 'failures.json'}")
    print(f"Plot:     {output_dir / 'official_scores_curve.png'}")
    return 1 if any(item["status"] == "error" for item in all_results) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
