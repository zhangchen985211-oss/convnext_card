from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import score_checkpoints_official as batch
from src.official_scoring import (
    OFFICIAL_ROTATION_THRESHOLD_DEG,
    OFFICIAL_ROTATION_THRESHOLD_RAD,
    OFFICIAL_TRANSLATION_THRESHOLD,
    official_pose_score,
    run_self_test,
    score_prediction_json,
)


def _write_config(path: Path, results_root: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "path_results": str(results_root),
        "split_submission": split,
    }), encoding="utf-8")


def test_checkpoint_discovery_matches_config_output_and_split(tmp_path):
    project = tmp_path / "project"
    results = tmp_path / "results" / "route_b_convnext_lightbox"
    configs = project / "server_configs" / "lightbox"
    lightbox_config = configs / "loop_niter_0001.json"
    sunlamp_config = project / "server_configs" / "sunlamp" / "loop_niter_0001.json"
    _write_config(lightbox_config, results, "lightbox")
    _write_config(sunlamp_config, results, "sunlamp")
    expected = results / "server_configs/lightbox/loop_niter_0001.json/ckpt/init.pth"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"checkpoint")
    other = results / "server_configs/sunlamp/loop_niter_0001.json/ckpt/init.pth"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"checkpoint")

    discovered = batch.discover_checkpoints(
        results, configs, "lightbox", project_root=project,
    )

    assert len(discovered) == 1
    assert discovered[0]["checkpoint_path"] == expected
    assert discovered[0]["config_path"] == lightbox_config
    assert discovered[0]["niter"] == 1
    assert discovered[0]["config_arg"] == "server_configs/lightbox/loop_niter_0001.json"


def test_partial_or_failed_metrics_never_rank_as_official():
    checkpoint = {
        "niter": 4,
        "model_id": "loop__init",
        "checkpoint_path": Path("weights/init.pth"),
        "config_path": Path("configs/loop.json"),
    }
    partial, _ = batch.result_from_metrics(checkpoint, {
        "gt_count": 10,
        "matched_count": 3,
        "evaluated_sample_count": 3,
        "evaluation_truncated": True,
        "official_valid": False,
        "official_avg_total_score": None,
        "solved_only_official_avg_total_score": 0.04,
    }, "lightbox", Path("dataset/lightbox"))
    complete, _ = batch.result_from_metrics(checkpoint, {
        "gt_count": 10,
        "matched_count": 10,
        "evaluated_sample_count": 10,
        "evaluation_truncated": False,
        "official_valid": True,
        "official_avg_translation_score": 0.01,
        "official_avg_rotation_score": 0.02,
        "official_avg_total_score": 0.03,
        "solved_only_official_avg_total_score": 0.03,
    }, "lightbox", Path("dataset/lightbox"))

    assert partial["status"] == "partial"
    assert partial["official_valid"] is False
    assert partial["total_score"] is None
    assert partial["failed_count"] == 7
    assert complete["official_valid"] is True
    assert complete["total_score"] == 0.03
    ranked = batch.sort_results_for_ranking([partial, complete])
    assert [item["official_valid"] for item in ranked] == [True, False]


def test_official_scoring_formula_and_threshold_metadata():
    run_self_test()
    gt_t = np.array([1.0, 0.0, 0.0])
    gt_q = [1.0, 0.0, 0.0, 0.0]
    t_pred = [1.0 + 2.0 * OFFICIAL_TRANSLATION_THRESHOLD, 0.0, 0.0]
    angle = 2.0 * OFFICIAL_ROTATION_THRESHOLD_RAD
    q_pred = [np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0]

    score = official_pose_score(t_pred, q_pred, gt_t, gt_q)

    assert np.isclose(score["translation_score"], 2.0 * OFFICIAL_TRANSLATION_THRESHOLD)
    assert np.isclose(score["rotation_score"], angle)
    assert batch.OFFICIAL_ROTATION_THRESHOLD_DEG == 0.169


def _pose(filename: str, offset: float = 0.0) -> dict:
    return {
        "filename": filename,
        "r_Vo2To_vbs_true": [offset, 0.0, 10.0],
        "q_vbs2tango_true": [1.0, 0.0, 0.0, 0.0],
    }


def test_saved_prediction_scoring_requires_exact_full_coverage(tmp_path):
    gt_path = tmp_path / "test.json"
    gt = [_pose("one.jpg"), _pose("two.jpg")]
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    complete = score_prediction_json([_pose("one.jpg", 1.0), _pose("two.jpg")], gt_path)
    assert complete["official_valid"] is True
    assert complete["official_avg_total_score"] == pytest.approx(0.05)
    assert complete["solved_only_official_avg_translation_score"] == pytest.approx(0.05)

    for predictions in (gt[:1], gt + gt[:1], gt + [_pose("extra.jpg")], []):
        metrics = score_prediction_json(predictions, gt_path)
        assert metrics["official_valid"] is False
        assert metrics["official_avg_total_score"] is None

    gt_path.write_text("[]", encoding="utf-8")
    assert score_prediction_json([], gt_path)["official_valid"] is False
    gt_path.write_text(json.dumps(gt + gt[:1]), encoding="utf-8")
    assert score_prediction_json(gt, gt_path)["official_valid"] is False


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_scorer_rejects_nonfinite_prediction_instead_of_ranking_it(bad_value):
    with pytest.raises(ValueError, match="finite"):
        official_pose_score([bad_value, 0, 10], [1, 0, 0, 0], [0, 0, 10], [1, 0, 0, 0])


def test_batch_reuses_inference_entry_and_writes_rankings_failures_and_curves(tmp_path, monkeypatch):
    project = tmp_path / "project"
    results = tmp_path / "results"
    config_dir = project / "configs"
    data_dir = tmp_path / "dataset" / "lightbox"
    data_dir.mkdir(parents=True)
    gt_json = data_dir / "public_test.json"
    gt_json.write_text(json.dumps([_pose("one.jpg"), _pose("two.jpg")]), encoding="utf-8")
    output = tmp_path / "score_output"
    checkpoints = []
    for iteration in range(3):
        config = config_dir / f"loop_niter_{iteration:04d}.json"
        _write_config(config, results, "lightbox")
        checkpoint = results / "configs" / config.name / "ckpt/init.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"placeholder")
        checkpoints.append(checkpoint)
    (checkpoints[0].parent / "ignored.pth").write_bytes(b"not-selected")
    calls = []
    real_run = batch.subprocess.run

    def fake_run(command, **kwargs):
        if len(command) < 2 or command[1] != str(project / "evaluate_target.py"):
            return real_run(command, **kwargs)
        calls.append(command)
        assert Path(command[1]) == project / "evaluate_target.py"
        assert command[command.index("--gt-json") + 1] == str(gt_json)
        assert command[command.index("--dataset-root") + 1] == str(data_dir.parent)
        assert command[command.index("--topk") + 1] == "9"
        assert kwargs["cwd"] == project
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        if checkpoint == checkpoints[2]:
            return SimpleNamespace(returncode=2)
        predictions = [_pose("one.jpg", 1.0), _pose("two.jpg")]
        if checkpoint == checkpoints[1]:
            predictions.pop()
        metrics = score_prediction_json(predictions, gt_json)
        metrics["failed_samples"] = (
            [{"filename": "two.jpg", "reason": "all_pose_strategies_failed"}]
            if len(predictions) == 1 else []
        )
        metrics_path = Path(command[command.index("--metrics-json") + 1])
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch, "PROJECT_ROOT", project)
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    exit_code = batch.main([
        "--results-dir", str(results),
        "--config-dir", str(config_dir),
        "--split", "lightbox",
        "--dataset-dir", str(data_dir),
        "--gt-json", str(gt_json),
        "--output-dir", str(output),
        "--topk", "9",
    ])

    assert exit_code == 1
    assert len(calls) == 3
    summary = json.loads((output / "official_scores_sorted.json").read_text())
    assert summary["total_checkpoints_found"] == 3
    assert [result["status"] for result in summary["results"]] == ["ok", "pose_failures", "error"]
    assert summary["results"][0]["gt_json_path"] == str(gt_json)
    assert summary["results"][1]["total_score"] is None
    assert summary["results"][1]["solved_total_score"] == pytest.approx(0.1)
    failures = json.loads((output / "failures.json").read_text())
    assert failures["failure_count"] == 2
    assert failures["failures"][0]["filename"] == "two.jpg"
    curve = json.loads((output / "official_scores_curve.json").read_text())
    assert curve["official_valid"] == [True, False, False]
    assert (output / "official_scores_curve.png").read_bytes().startswith(b"\x89PNG")
