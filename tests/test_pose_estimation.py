from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src import pose_estimation


class _CV2Proxy:
    def __init__(self, rvec, tvec, minimum_threshold=0.0):
        self.rvec = rvec
        self.tvec = tvec
        self.minimum_threshold = minimum_threshold
        self.ransac_calls = []

    def __getattr__(self, name):
        return getattr(cv2, name)

    def solvePnPRansac(self, object_points, image_points, camera, dist, **kwargs):
        self.ransac_calls.append(
            (len(object_points), float(kwargs["reprojectionError"]))
        )
        if float(kwargs["reprojectionError"]) < self.minimum_threshold:
            return False, None, None, None
        inliers = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
        return True, self.rvec.copy(), self.tvec.copy(), inliers

    def solvePnP(self, *args, **kwargs):
        return False, None, None


def _synthetic_inputs():
    world = np.array(
        [
            [-1.0, -0.7, -0.4], [-0.3, -0.8, 0.2], [0.8, -0.5, -0.1],
            [1.0, 0.6, 0.3], [0.2, 0.9, -0.5], [-0.8, 0.7, 0.5],
            [-0.7, -0.1, 0.8], [0.6, 0.2, 0.7], [0.0, 0.0, -0.9],
            [-0.2, 0.3, 0.1], [0.4, -0.2, -0.6],
        ],
        dtype=np.float32,
    )
    camera = np.array([[80.0, 0.0, 15.5], [0.0, 80.0, 15.5], [0.0, 0.0, 1.0]])
    rvec = np.array([[0.08], [-0.04], [0.03]], dtype=np.float64)
    tvec = np.array([[0.1], [-0.05], [6.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(world, rvec, tvec, camera, None)
    points_2d = projected.reshape(-1, 2).astype(np.float32)
    outputs = []
    for branch in range(2):
        heatmaps = np.zeros((1, len(world), 32, 32), dtype=np.float32)
        for keypoint in range(len(world)):
            heatmaps[0, keypoint, keypoint % 32, (keypoint * 3 + branch) % 32] = keypoint + 1
        outputs.append({"hm_c": heatmaps})
    runtime = SimpleNamespace(
        cv2=_CV2Proxy(rvec, tvec),
        F=SimpleNamespace(interpolate=lambda tensor, **kwargs: tensor),
        pose_utils=SimpleNamespace(
            heatmap_to_points=lambda tensor: points_2d[None, ...]
        ),
    )
    image = np.zeros((1, 3, 32, 32), dtype=np.float32)
    return outputs, image, world, camera.astype(np.float32), runtime


def test_shared_entrypoint_uses_official_search_defaults_and_config(monkeypatch):
    outputs, image, world, camera, runtime = _synthetic_inputs()
    calls = []

    def record_refinement(result, world_kpts, camera_matrix, config):
        calls.append(config)
        return {**result, "refinement": {"accepted": False}}

    monkeypatch.setattr("src.pose_refinement.refine_pose_lm", record_refinement)
    config = {
        "rows": 32,
        "cols": 32,
        "pose_threshold_reference_size": [64, 64],
        "pose_eval_search": "all_topks",
        "pose_refine_lm": True,
        "eval_reprojection_error": 2.0,
    }

    selected, reason = pose_estimation.estimate_pose(
        outputs, image, world, camera, config, runtime=runtime
    )

    assert selected is not None
    assert reason == selected["strategy"]
    assert calls and all(item is config for item in calls)
    # Config base threshold 2px is scaled by 32/64 before the solver sees it.
    assert runtime.cv2.ransac_calls[:2] == [(7, 1.0), (7, 1.0)]
    # all_topks compares rescue support counts at the same initial threshold.
    assert runtime.cv2.ransac_calls[2:10] == [
        (8, 1.0), (8, 1.0), (9, 1.0), (9, 1.0),
        (10, 1.0), (10, 1.0), (11, 1.0), (11, 1.0),
    ]


def test_shared_entrypoint_cli_overrides_take_precedence():
    outputs, image, world, camera, runtime = _synthetic_inputs()
    selected, reason = pose_estimation.estimate_pose(
        outputs,
        image,
        world,
        camera,
        {"rows": 32, "cols": 32, "eval_reprojection_error": 3.0},
        runtime=runtime,
        topk=6,
        reprojection_error=2.0,
        fallback_topks=[9],
        fallback_reprojection_errors=[4.0],
        min_inliers=5,
    )
    assert selected is not None
    assert reason == selected["strategy"]
    assert runtime.cv2.ransac_calls[0] == (6, 2.0)


def test_shared_entrypoint_runs_threshold_fallback_and_reports_total_failure():
    outputs, image, world, camera, runtime = _synthetic_inputs()
    runtime.cv2.minimum_threshold = 3.0
    selected, reason = pose_estimation.estimate_pose(
        outputs,
        image,
        world,
        camera,
        {"rows": 32, "cols": 32, "eval_reprojection_error": 1.0},
        runtime=runtime,
    )
    assert selected is not None
    assert reason == selected["strategy"]
    assert [threshold for _, threshold in runtime.cv2.ransac_calls[:3]] == [1.0, 1.0, 1.0]
    assert any(threshold == 3.0 for _, threshold in runtime.cv2.ransac_calls)

    outputs, image, world, camera, runtime = _synthetic_inputs()
    runtime.cv2.minimum_threshold = float("inf")
    selected, reason = pose_estimation.estimate_pose(
        outputs,
        image,
        world,
        camera,
        {"rows": 32, "cols": 32},
        runtime=runtime,
    )
    assert selected is None
    assert reason == "all_pose_strategies_failed"


def test_training_validation_and_target_evaluation_share_solver():
    root = Path(__file__).resolve().parents[1]
    training_source = (root / "src" / "pose_utils.py").read_text(encoding="utf-8")
    evaluation_source = (root / "evaluate_target.py").read_text(encoding="utf-8")
    tree = ast.parse(training_source)
    eval_loop = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "eval_loop"
    )
    calls = [
        node for node in ast.walk(eval_loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert any(node.func.id == "estimate_pose" for node in calls)
    assert not any(node.func.id == "solvePnPRansac" for node in calls)
    assert "estimate_pose(" in evaluation_source
    assert "eval_failed_pose_penalty" in training_source
    assert "speed_total + failed_count * failed_pose_penalty" in training_source
