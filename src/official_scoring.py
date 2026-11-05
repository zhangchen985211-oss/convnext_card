"""SPEED+ pose score, retained from the project's original competition scorer.

Rule reference: https://kelvins.esa.int/pose-estimation-2021/scoring/
"""

from __future__ import annotations

import json
import math
from typing import Sequence, Dict

import numpy as np

from src import speedscore

OFFICIAL_TRANSLATION_THRESHOLD = 0.002173
OFFICIAL_ROTATION_THRESHOLD_DEG = 0.169
OFFICIAL_ROTATION_THRESHOLD_RAD = math.radians(OFFICIAL_ROTATION_THRESHOLD_DEG)


def normalize_quaternion(q: Sequence[float]) -> np.ndarray:
    """把四元数归一化，避免非单位四元数把旋转误差算小。"""
    q_arr = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q_arr)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid quaternion norm")
    return q_arr / norm


def official_pose_score(
    t_pred: Sequence[float],
    q_pred: Sequence[float],
    t_gt: Sequence[float],
    q_gt: Sequence[float],
) -> Dict[str, float]:
    """按官方规则计算单张图片的姿态分数。

    注意：
    - 预测四元数和 GT 四元数只要顺序一致即可；
    - 当前模型链路里预测和 GT 都按 WXYZ 使用，因此这里不做重排；
    - 旋转分数最终使用 radians，而不是 degrees。
    """

    t_pred_arr = np.asarray(t_pred, dtype=np.float64).reshape(3)
    t_gt_arr = np.asarray(t_gt, dtype=np.float64).reshape(3)
    if not np.isfinite(t_pred_arr).all() or not np.isfinite(t_gt_arr).all():
        raise ValueError("translation vectors must contain only finite values")
    t_gt_norm = np.linalg.norm(t_gt_arr)
    if not np.isfinite(t_gt_norm) or t_gt_norm <= 0.0:
        raise ValueError("invalid ground-truth translation norm")

    q_pred_arr = normalize_quaternion(q_pred)
    q_gt_arr = normalize_quaternion(q_gt)

    # 平移项：预测平移误差除以 GT 距离，得到无量纲归一化误差。
    raw_translation = float(np.linalg.norm(t_pred_arr - t_gt_arr) / t_gt_norm)

    # q 和 -q 表示同一个旋转，因此取 dot 的绝对值；再裁剪到 [0, 1]
    # 避免浮点误差导致 acos 输入越界。
    dot = float(np.clip(abs(np.dot(q_pred_arr, q_gt_arr)), 0.0, 1.0))
    raw_rotation = float(2.0 * math.acos(dot))

    # 官方阈值是分项清零：平移项和旋转项互不影响。
    translation_score = (
        0.0
        if raw_translation < OFFICIAL_TRANSLATION_THRESHOLD
        else raw_translation
    )
    rotation_score = (
        0.0 if raw_rotation < OFFICIAL_ROTATION_THRESHOLD_RAD else raw_rotation
    )
    return {
        "translation_score": translation_score,
        "rotation_score": rotation_score,
        "total_score": translation_score + rotation_score,
        "raw_translation_error": raw_translation,
        "raw_rotation_error_rad": raw_rotation,
    }


def run_self_test() -> None:
    """不依赖 torch/kornia 的轻量自检，用来防止评分公式被改坏。"""
    def assert_close(name: str, actual: float, expected: float, tol: float = 1e-10) -> None:
        if abs(actual - expected) > tol:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")

    q_identity = [1.0, 0.0, 0.0, 0.0]
    t_gt = [1.0, 0.0, 0.0]

    score = official_pose_score(t_gt, q_identity, t_gt, q_identity)
    assert_close("same pose total", score["total_score"], 0.0)

    small_t = [1.0 + OFFICIAL_TRANSLATION_THRESHOLD * 0.5, 0.0, 0.0]
    score = official_pose_score(small_t, q_identity, t_gt, q_identity)
    assert_close("translation deadband", score["translation_score"], 0.0)

    large_t_delta = OFFICIAL_TRANSLATION_THRESHOLD * 2.0
    large_t = [1.0 + large_t_delta, 0.0, 0.0]
    score = official_pose_score(large_t, q_identity, t_gt, q_identity)
    assert_close("translation above threshold", score["translation_score"], large_t_delta)

    small_angle = OFFICIAL_ROTATION_THRESHOLD_RAD * 0.5
    q_small = [math.cos(small_angle / 2.0), math.sin(small_angle / 2.0), 0.0, 0.0]
    score = official_pose_score(t_gt, q_small, t_gt, q_identity)
    assert_close("rotation deadband", score["rotation_score"], 0.0)

    large_angle = OFFICIAL_ROTATION_THRESHOLD_RAD * 2.0
    q_large = [math.cos(large_angle / 2.0), math.sin(large_angle / 2.0), 0.0, 0.0]
    score = official_pose_score(t_gt, q_large, t_gt, q_identity)
    assert_close("rotation above threshold", score["rotation_score"], large_angle)

    score_a = official_pose_score(t_gt, q_large, t_gt, q_identity)
    score_b = official_pose_score(t_gt, [-v for v in q_large], t_gt, q_identity)
    assert_close("q sign invariance", score_a["rotation_score"], score_b["rotation_score"])

    score = official_pose_score(large_t, q_small, t_gt, q_identity)
    assert_close("independent rotation deadband", score["rotation_score"], 0.0)
    assert_close("independent translation score", score["translation_score"], large_t_delta)

    print("Official scoring self-test passed.")


def score_prediction_json(predictions, gt_json_path):
    """Score stored predictions; only exact, nonempty GT coverage is official-valid."""
    with open(gt_json_path, "r", encoding="utf-8") as fp:
        gt_items = json.load(fp)
    gt_map = {item["filename"]: item for item in gt_items}
    pred_map = {item["filename"]: item for item in predictions}
    scores = []
    official_scores = []
    missing = []
    for filename, gt_item in gt_map.items():
        pred = pred_map.get(filename)
        if pred is None:
            missing.append(filename)
            continue
        speed, speed_t, speed_r = speedscore.speed_score(
            np.array(pred["r_Vo2To_vbs_true"], dtype=np.float32),
            np.array(pred["q_vbs2tango_true"], dtype=np.float32),
            np.array(gt_item["r_Vo2To_vbs_true"], dtype=np.float32),
            np.array(gt_item["q_vbs2tango_true"], dtype=np.float32),
            applyThresh=False,
        )
        scores.append({
            "filename": filename,
            "speed": float(speed),
            "speed_t": float(speed_t),
            "speed_r": float(speed_r),
        })
        official_scores.append(official_pose_score(
            pred["r_Vo2To_vbs_true"], pred["q_vbs2tango_true"],
            gt_item["r_Vo2To_vbs_true"], gt_item["q_vbs2tango_true"],
        ))
    unexpected = sorted(set(pred_map) - set(gt_map))
    complete = (
        bool(gt_items)
        and len(gt_items) == len(gt_map)
        and not missing
        and not unexpected
        and len(predictions) == len(pred_map)
        and len(pred_map) == len(gt_map)
        and len(official_scores) == len(gt_map)
    )
    solved_translation = [item["translation_score"] for item in official_scores]
    solved_rotation = [item["rotation_score"] for item in official_scores]
    solved_total = [item["total_score"] for item in official_scores]
    return {
        "gt_count": len(gt_items),
        "pred_count": len(predictions),
        "matched_count": len(scores),
        "coverage": float(len(scores) / len(gt_items)) if gt_items else 0.0,
        "avg_speed": float(np.mean([s["speed"] for s in scores])) if scores else None,
        "avg_speed_t": float(np.mean([s["speed_t"] for s in scores])) if scores else None,
        "avg_speed_r": float(np.mean([s["speed_r"] for s in scores])) if scores else None,
        "missing_prediction_count": len(missing),
        "missing_prediction_examples": missing[:20],
        "unexpected_prediction_count": len(unexpected),
        "unexpected_prediction_examples": unexpected[:20],
        "per_sample_scores": scores,
        "official_valid": bool(complete),
        "official_coverage": float(len(official_scores) / len(gt_items)) if gt_items else 0.0,
        "official_avg_translation_score": (
            float(np.mean([item["translation_score"] for item in official_scores]))
            if complete else None
        ),
        "official_avg_rotation_score": (
            float(np.mean([item["rotation_score"] for item in official_scores]))
            if complete else None
        ),
        "official_avg_total_score": (
            float(np.mean([item["total_score"] for item in official_scores]))
            if complete else None
        ),
        "solved_only_official_avg_total_score": (
            float(np.mean(solved_total)) if solved_total else None
        ),
        "solved_only_official_avg_translation_score": (
            float(np.mean(solved_translation)) if solved_translation else None
        ),
        "solved_only_official_avg_rotation_score": (
            float(np.mean(solved_rotation)) if solved_rotation else None
        ),
    }
