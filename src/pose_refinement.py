"""Small NumPy/OpenCV helpers shared by pseudo-label and evaluation solvers."""

from __future__ import annotations

import cv2
import numpy as np


def pixel_threshold_scale(config):
    """Scale reference-image pixel thresholds; normalized metrics stay unchanged."""
    reference = config.get("pose_threshold_reference_size")
    if reference is None:
        return 1.0
    if np.isscalar(reference):
        reference = (float(reference), float(reference))
    reference = np.asarray(reference, dtype=np.float64).reshape(2)
    current = np.asarray([config["rows"], config["cols"]], dtype=np.float64)
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("pose_threshold_reference_size must contain positive dimensions")
    if not np.isfinite(current).all() or np.any(current <= 0):
        raise ValueError("rows and cols must be positive")
    ratios = current / reference
    if not np.isclose(ratios[0], ratios[1]):
        raise ValueError(
            "Pixel thresholds require isotropic resizing relative to "
            "pose_threshold_reference_size; restore reference coordinates for other aspect ratios"
        )
    return float(ratios[0])


def refine_pose_lm(result, world_kpts, camera_matrix, dist_coeffs=None, config=None):
    """Refine RANSAC inliers, keeping the original pose if any guard fails.

    OpenCV inliers index the selected subset, not the complete CAD point array.
    Copy the initial vectors because OpenCV may mutate them in place.
    """
    config = config or {}
    refined = dict(result)
    refined["refinement"] = {"attempted": False, "accepted": False, "reason": "disabled"}
    if not config.get("pose_refine_lm", False):
        return refined

    def keep(reason, **diagnostics):
        refined["refinement"] = {
            "attempted": reason not in {"not_ransac", "too_few_inliers", "unavailable"},
            "accepted": False,
            "reason": reason,
            **diagnostics,
        }
        return refined

    if not hasattr(cv2, "solvePnPRefineLM"):
        return keep("unavailable")
    if not result.get("accepted") or result.get("inliers") is None:
        return keep("not_ransac")
    selected = np.asarray(result.get("selected_keypoints", []), dtype=np.int64)
    local_inliers = np.unique(np.asarray(result["inliers"], dtype=np.int64).reshape(-1))
    if local_inliers.size < 4:
        return keep("too_few_inliers")
    if np.any(local_inliers < 0) or np.any(local_inliers >= selected.size):
        return keep("invalid_inlier_indices")
    world = np.asarray(world_kpts, dtype=np.float64).reshape(-1, 3)
    observed = np.asarray(result["pred_kpts"], dtype=np.float64).reshape(-1, 2)
    if np.any(selected < 0) or np.any(selected >= len(world)) or len(observed) != len(world):
        return keep("invalid_correspondences")
    indices = selected[local_inliers]
    holdout = np.setdiff1d(np.arange(len(world)), selected)
    r0 = np.asarray(result["rvecs"], dtype=np.float64).reshape(3, 1).copy()
    t0 = np.asarray(result["tvecs"], dtype=np.float64).reshape(3, 1).copy()
    camera = np.asarray(camera_matrix, dtype=np.float64)

    def residuals(rvec, tvec):
        projected, _ = cv2.projectPoints(world, rvec, tvec, camera, dist_coeffs)
        return np.linalg.norm(projected.reshape(-1, 2) - observed, axis=1)

    try:
        before = residuals(r0, t0)
        r1, t1 = cv2.solvePnPRefineLM(
            np.ascontiguousarray(world[indices]),
            np.ascontiguousarray(observed[indices]),
            camera,
            dist_coeffs,
            r0.copy(),
            t0.copy(),
        )
        if not np.isfinite(r1).all() or not np.isfinite(t1).all():
            return keep("non_finite_pose")
        rotation, _ = cv2.Rodrigues(r1)
        depths = (world @ rotation.T + t1.reshape(1, 3))[:, 2]
        after = residuals(r1, t1)
    except cv2.error:
        return keep("opencv_error")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return keep("non_finite_residuals")
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        return keep("non_positive_depth")
    min_depth = float(config.get("gate_min_depth", config.get("pseudo_min_depth", 0.1)))
    max_depth = float(config.get("gate_max_depth", config.get("pseudo_max_depth", 20.0)))
    if depths.min() < min_depth or depths.max() > max_depth:
        return keep("depth_out_of_range")
    threshold = result.get("reprojection_error")
    new_inliers = None
    if threshold is not None:
        new_inliers = np.flatnonzero(after[selected] <= float(threshold)).reshape(-1, 1)
        if len(new_inliers) < len(local_inliers):
            return keep("inlier_support_reduced")
    diagnostics = {
        "inlier_rmse_before": float(np.sqrt(np.mean(before[indices] ** 2))),
        "inlier_rmse_after": float(np.sqrt(np.mean(after[indices] ** 2))),
        "full_mean_before": float(before.mean()),
        "full_mean_after": float(after.mean()),
        "selected_mean_before": float(before[selected].mean()),
        "selected_mean_after": float(after[selected].mean()),
        "selected_max_before": float(before[selected].max()),
        "selected_max_after": float(after[selected].max()),
        "full_max_before": float(before.max()),
        "full_max_after": float(after.max()),
        "holdout_mean_before": float(before[holdout].mean()) if holdout.size else None,
        "holdout_mean_after": float(after[holdout].mean()) if holdout.size else None,
    }
    tolerance = float(config.get("pose_refine_guard_tolerance", 1e-6))
    if diagnostics["inlier_rmse_after"] > diagnostics["inlier_rmse_before"] + tolerance:
        return keep("inlier_residual_worsened", **diagnostics)
    if diagnostics["full_mean_after"] > diagnostics["full_mean_before"] + tolerance:
        return keep("full_residual_worsened", **diagnostics)
    for metric in ("selected_mean", "selected_max", "full_max"):
        if diagnostics[f"{metric}_after"] > diagnostics[f"{metric}_before"] + tolerance:
            return keep(f"{metric}_worsened", **diagnostics)
    if holdout.size and diagnostics["holdout_mean_after"] > diagnostics["holdout_mean_before"] + tolerance:
        return keep("holdout_residual_worsened", **diagnostics)
    refined.update({"rvecs": r1, "tvecs": t1})
    if new_inliers is not None:
        refined["inliers"] = new_inliers
    refined["refinement"] = {"attempted": True, "accepted": True, "reason": "accepted", **diagnostics}
    return refined
