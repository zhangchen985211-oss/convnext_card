"""Architecture-neutral keypoint-to-pose solving and fallback strategies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_TOPK = 7
DEFAULT_FALLBACK_TOPKS = [8, 9, 10, 11]
DEFAULT_FALLBACK_REPROJECTION_ERRORS = [1.0, 2.0, 3.0, 4.0]
DEFAULT_MIN_INLIERS = 4


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_branch_measurements(
    runtime: SimpleNamespace,
    sample_outputs: Sequence[Dict[str, Any]],
    image_tensor: Any,
) -> List[Dict[str, Any]]:
    """从 模型的多个输出分支中提取 2D 关键点和每个关键点的热图响应。"""
    measurements = []
    for branch_index, output in enumerate(sample_outputs):
        pred_mask = runtime.F.interpolate(
            output["hm_c"],
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_kpts = tensor_to_numpy(runtime.pose_utils.heatmap_to_points(pred_mask)[0]).astype(np.float32)
        responses = tensor_to_numpy(output["hm_c"])
        if responses.ndim == 4:
            responses = responses[0]
        responses = np.max(np.max(responses, axis=1), axis=1).astype(np.float32)
        measurements.append(
            {
                "name": f"branch{branch_index}",
                "branch_index": branch_index,
                "keypoints": pred_kpts,
                "responses": responses,
            }
        )
    return measurements


def build_fused_measurement(branches: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """构造双分支融合关键点：每个关键点取响应更高的那个分支坐标。"""
    if len(branches) < 2:
        return None
    keypoints = np.stack([branch["keypoints"] for branch in branches], axis=0)
    responses = np.stack([branch["responses"] for branch in branches], axis=0)
    best_branch = np.argmax(responses, axis=0)
    fused_keypoints = keypoints[best_branch, np.arange(keypoints.shape[1])]
    fused_responses = responses[best_branch, np.arange(responses.shape[1])]
    return {
        "name": "fused_best_response",
        "branch_index": None,
        "keypoints": fused_keypoints.astype(np.float32),
        "responses": fused_responses.astype(np.float32),
        "source_branches": best_branch.tolist(),
    }


def selected_points(
    measurement: Dict[str, Any],
    world_kpts: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """按热图响应从高到低选 topk 个 2D-3D 对应点。"""
    responses = measurement["responses"]
    keypoints = measurement["keypoints"]
    order = np.argsort(-responses)
    selected_index = order[: min(topk, len(order))]
    object_points = world_kpts[selected_index, :].astype(np.float32)
    image_points = keypoints[selected_index, :].astype(np.float32)
    response_sum = float(np.sum(responses[selected_index]))
    return selected_index, object_points, image_points, response_sum


def reprojection_stats(
    cv2_module: Any,
    rvecs: Any,
    tvecs: Any,
    object_points: np.ndarray,
    image_points: np.ndarray,
    k_mat: np.ndarray,
) -> Tuple[float, float]:
    """用零畸变重投影误差评价 PnP 候选解。"""
    projected, _ = cv2_module.projectPoints(
        object_points.astype(np.float32),
        np.asarray(rvecs, dtype=np.float64).reshape(3, 1),
        np.asarray(tvecs, dtype=np.float64).reshape(3, 1),
        k_mat,
        None,
    )
    projected = projected.reshape(-1, 2)
    residuals = np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)
    return float(np.mean(residuals)), float(np.max(residuals))


def minimum_depth(
    cv2_module: Any,
    rvecs: Any,
    tvecs: Any,
    object_points: np.ndarray,
) -> float:
    """检查航天器 3D 关键点投到相机坐标系后是否都在相机前方。"""
    rotation, _ = cv2_module.Rodrigues(np.asarray(rvecs, dtype=np.float64).reshape(3, 1))
    camera_points = (
        rotation @ object_points.astype(np.float64).reshape(-1, 3).T
        + np.asarray(tvecs, dtype=np.float64).reshape(3, 1)
    ).T
    return float(np.min(camera_points[:, 2]))


def validate_candidate(
    runtime: SimpleNamespace,
    candidate: Dict[str, Any],
    measurement: Dict[str, Any],
    world_kpts: np.ndarray,
    k_mat: np.ndarray,
    min_inliers: int,
    solver_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Check solver validity for full-coverage evaluation, not pseudo-label quality."""
    if not candidate.get("accepted"):
        return None
    if candidate.get("rvecs") is None or candidate.get("tvecs") is None:
        return None
    from src.pose_refinement import refine_pose_lm

    candidate = refine_pose_lm(
        {**candidate, "pred_kpts": measurement["keypoints"]},
        world_kpts, k_mat, config=solver_config,
    )
    rvecs = np.asarray(candidate.get("rvecs"), dtype=np.float64).reshape(3, 1)
    tvecs = np.asarray(candidate.get("tvecs"), dtype=np.float64).reshape(3, 1)
    if not np.all(np.isfinite(rvecs)) or not np.all(np.isfinite(tvecs)):
        return None

    selected_image = measurement["keypoints"][candidate["selected_keypoints"], :].astype(np.float32)
    selected_world = world_kpts[candidate["selected_keypoints"], :].astype(np.float32)
    mean_reproj, max_reproj = reprojection_stats(
        runtime.cv2,
        rvecs,
        tvecs,
        selected_world,
        selected_image,
        k_mat,
    )
    full_mean_reproj, full_max_reproj = reprojection_stats(
        runtime.cv2,
        rvecs,
        tvecs,
        world_kpts.astype(np.float32),
        measurement["keypoints"].astype(np.float32),
        k_mat,
    )
    if not np.isfinite(mean_reproj) or not np.isfinite(full_mean_reproj):
        return None

    min_depth = minimum_depth(runtime.cv2, rvecs, tvecs, world_kpts)
    if not np.isfinite(min_depth) or min_depth <= 0.0:
        return None

    inliers = candidate.get("inliers")
    inlier_count = 0 if inliers is None else int(np.asarray(inliers).reshape(-1).shape[0])
    support_count = len(candidate["selected_keypoints"])
    if (inliers is not None and inlier_count < min_inliers) or support_count < min_inliers:
        return None

    candidate = dict(candidate)
    candidate.update(
        {
            "rvecs": rvecs,
            "tvecs": tvecs,
            "inlier_count": inlier_count,
            "support_count": support_count,
            "ransac_verified": inliers is not None,
            "solver_success": True,
            "quality_passed": None,
            "mean_reprojection_error": mean_reproj,
            "max_reprojection_error": max_reproj,
            "full_mean_reprojection_error": full_mean_reproj,
            "full_max_reprojection_error": full_max_reproj,
            "min_depth": min_depth,
            "measurement_name": measurement["name"],
            "branch_index": measurement.get("branch_index"),
        }
    )
    return candidate


def solve_ransac_candidate(
    runtime: SimpleNamespace,
    measurement: Dict[str, Any],
    world_kpts: np.ndarray,
    k_mat: np.ndarray,
    topk: int,
    reprojection_error: float,
    strategy: str,
) -> Dict[str, Any]:
    """使用 solvePnPRansac + EPNP 求解一个候选 pose。"""
    selected_index, object_points, image_points, response_sum = selected_points(
        measurement,
        world_kpts,
        topk,
    )
    try:
        accepted, rvecs, tvecs, inliers = runtime.cv2.solvePnPRansac(
            object_points,
            image_points,
            k_mat,
            None,
            confidence=0.99,
            reprojectionError=float(reprojection_error),
            flags=runtime.cv2.SOLVEPNP_EPNP,
        )
    except runtime.cv2.error:
        accepted, rvecs, tvecs, inliers = False, None, None, None
    return {
        "accepted": bool(accepted),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "inliers": inliers,
        "response_sum": response_sum,
        "selected_keypoints": selected_index.tolist(),
        "topk": int(topk),
        "reprojection_error": float(reprojection_error),
        "method": "solvePnPRansac_EPNP",
        "strategy": strategy,
    }


def solve_direct_candidate(
    runtime: SimpleNamespace,
    measurement: Dict[str, Any],
    world_kpts: np.ndarray,
    k_mat: np.ndarray,
    method_flag: int,
    method_name: str,
) -> Dict[str, Any]:
    """最后兜底：不用 RANSAC，直接用全部关键点 solvePnP。"""
    selected_index, object_points, image_points, response_sum = selected_points(
        measurement,
        world_kpts,
        len(world_kpts),
    )
    try:
        accepted, rvecs, tvecs = runtime.cv2.solvePnP(
            object_points,
            image_points,
            k_mat,
            None,
            flags=method_flag,
        )
    except runtime.cv2.error:
        accepted, rvecs, tvecs = False, None, None
    return {
        "accepted": bool(accepted),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "inliers": None,
        "response_sum": response_sum,
        "selected_keypoints": selected_index.tolist(),
        "topk": int(len(world_kpts)),
        "reprojection_error": None,
        "method": method_name,
        "strategy": "direct_solvepnp_fallback",
    }


def candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Rank geometry before uncalibrated head peaks or threshold-dependent counts."""
    return (
        float(candidate.get("full_mean_reprojection_error", np.inf)),
        float(candidate.get("mean_reprojection_error", np.inf)),
        -float(candidate.get("inlier_count", 0)),
        -float(candidate.get("response_sum", 0.0)),
    )


def choose_best(candidates: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        return None
    valid.sort(key=candidate_sort_key)
    return valid[0]


def estimate_pose_with_fallback(
    runtime: SimpleNamespace,
    sample_outputs: Sequence[Dict[str, Any]],
    image_tensor: Any,
    world_kpts: np.ndarray,
    k_mat: np.ndarray,
    topk_start: int,
    fallback_topks: Sequence[int],
    fallback_reprojection_errors: Sequence[float],
    min_inliers: int,
    solver_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Coverage-oriented solver. Never use this permissive result as a pseudo label.

    all_topks compares the configured subset sizes at the initial pixel threshold
    before returning; later thresholds, fusion and direct PnP remain rescue tiers.
    """
    from src.pose_refinement import pixel_threshold_scale

    solver_config = solver_config or {}
    search = solver_config.get("pose_eval_search", "first_valid")
    if search not in {"first_valid", "all_topks"}:
        raise ValueError("pose_eval_search must be first_valid or all_topks")
    scale = pixel_threshold_scale(solver_config)
    errors = list(dict.fromkeys(float(value) * scale for value in fallback_reprojection_errors)) or [scale]
    if not all(np.isfinite(value) and value > 0 for value in errors):
        raise ValueError("RANSAC pixel thresholds must be finite and positive")
    branches = extract_branch_measurements(runtime, sample_outputs, image_tensor)
    if not branches:
        return None, "no_model_outputs"

    base_error = errors[0]

    def solve(measurement, topk, error, strategy):
        raw = solve_ransac_candidate(runtime, measurement, world_kpts, k_mat, topk, error, strategy)
        return validate_candidate(runtime, raw, measurement, world_kpts, k_mat, min_inliers, solver_config)

    strict_candidates = []
    for measurement in branches:
        strict_candidates.append(solve(measurement, topk_start, base_error, "strict_both_success"))
    if not (len(strict_candidates) >= 2 and all(item is not None for item in strict_candidates)):
        for item in strict_candidates:
            if item is not None:
                item["strategy"] = "single_branch_fallback"
    base_best = choose_best(strict_candidates)
    if search == "first_valid" and base_best is not None:
        return base_best, base_best["strategy"]

    topks = list(dict.fromkeys(int(value) for value in fallback_topks))
    if search == "all_topks":
        candidates = list(strict_candidates)
        for topk in topks:
            if topk != topk_start:
                candidates.extend(solve(item, topk, base_error, "expanded_topk_ransac") for item in branches)
        best = choose_best(candidates)
        if best is not None:
            return best, best["strategy"]

    # At each rescue threshold compare all configured subset sizes, rather than
    # letting the first solvable top-k suppress a better candidate in that tier.
    for reprojection_error in errors if search == "first_valid" else errors[1:]:
        expanded_candidates = []
        for topk in list(dict.fromkeys([topk_start] + topks)):
            for measurement in branches:
                expanded_candidates.append(solve(measurement, topk, reprojection_error, "expanded_topk_ransac"))
        best = choose_best(expanded_candidates)
        if best is not None:
            return best, "expanded_topk_ransac"

    fused = build_fused_measurement(branches)
    if fused is not None:
        fused_candidates = []
        for topk in (topk_start, len(world_kpts)):
            for reprojection_error in errors:
                raw = solve_ransac_candidate(
                    runtime,
                    fused,
                    world_kpts,
                    k_mat,
                    topk,
                    reprojection_error,
                    "fused_branch_ransac",
                )
                fused_candidates.append(
                    validate_candidate(runtime, raw, fused, world_kpts, k_mat, min_inliers, solver_config)
                )
        best = choose_best(fused_candidates)
        if best is not None:
            return best, "fused_branch_ransac"

    direct_measurements = list(branches)
    if fused is not None:
        direct_measurements.append(fused)
    direct_candidates = []
    for measurement in direct_measurements:
        for method_flag, method_name in (
            (runtime.cv2.SOLVEPNP_EPNP, "solvePnP_EPNP"),
            (runtime.cv2.SOLVEPNP_ITERATIVE, "solvePnP_ITERATIVE"),
        ):
            raw = solve_direct_candidate(
                runtime,
                measurement,
                world_kpts,
                k_mat,
                method_flag,
                method_name,
            )
            direct_candidates.append(
                validate_candidate(runtime, raw, measurement, world_kpts, k_mat, min_inliers, solver_config)
            )
    best = choose_best(direct_candidates)
    if best is not None:
        return best, "direct_solvepnp_fallback"

    return None, "all_pose_strategies_failed"

