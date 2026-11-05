"""NumPy geometry for source-only local appearance degradation sampling.

The information matrix measures local support of known synthetic keypoints.
It is a sampling proxy, not calibrated pose uncertainty or an occlusion model.
"""

from __future__ import annotations

import numpy as np


def projection_jacobian(camera_points, translation, camera_matrix, image_size, cad_scale):
    """Return [K,2,6] derivatives for [left object rotation, delta_t / CAD_scale].

    Rotation acts on R*X about the object's origin, not about the camera origin.
    Image coordinates are divided by [width-1,height-1]. Distortion is zero:
    callers must sample rectangles on the undistorted model image.
    """
    points = np.asarray(camera_points, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    height, width = image_size
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("camera_points must have shape [K,3]")
    if min(height, width) <= 1 or not np.isfinite(cad_scale) or cad_scale <= 0:
        raise ValueError("image dimensions and CAD scale must be positive")
    if not np.allclose(camera[2], [0, 0, 1]):
        raise ValueError("expected a pinhole camera matrix")
    if not np.isfinite(points).all() or not np.isfinite(translation).all() or np.any(points[:, 2] <= 0):
        raise ValueError("source camera points must be finite with positive depths")
    x, y, z = points.T
    dp = np.zeros((len(points), 2, 3), dtype=np.float64)
    dp[:, 0, 0] = camera[0, 0] / z
    dp[:, 0, 1] = camera[0, 1] / z
    dp[:, 0, 2] = -(camera[0, 0] * x + camera[0, 1] * y) / z**2
    dp[:, 1, 0] = camera[1, 0] / z
    dp[:, 1, 1] = camera[1, 1] / z
    dp[:, 1, 2] = -(camera[1, 0] * x + camera[1, 1] * y) / z**2
    dp[:, 0] /= width - 1
    dp[:, 1] /= height - 1
    rotated = points - translation
    skew = np.zeros((len(points), 3, 3), dtype=np.float64)
    rx, ry, rz = rotated.T
    skew[:, 0, 1], skew[:, 0, 2] = -rz, ry
    skew[:, 1, 0], skew[:, 1, 2] = rz, -rx
    skew[:, 2, 0], skew[:, 2, 1] = -ry, rx
    return np.concatenate((-dp @ skew, dp * float(cad_scale)), axis=-1)


def information_matrix(jacobian, weights):
    jacobian = np.asarray(jacobian, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if jacobian.shape != (len(weights), 2, 6):
        raise ValueError("Jacobian and keypoint weights must align")
    if not np.isfinite(jacobian).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Jacobian and nonnegative weights must be finite")
    return np.einsum("kij,kil,k->jl", jacobian, jacobian, weights)


def support_ratio(clean_information, degraded_information, rank_tolerance=1e-8):
    """Minimum generalized eigenvalue H_aug v = lambda H_clean v.

    A singular clean configuration returns None so the caller can use only
    global photometric changes instead of manufacturing support by damping H.
    """
    clean = np.asarray(clean_information, dtype=np.float64).reshape(6, 6)
    degraded = np.asarray(degraded_information, dtype=np.float64).reshape(6, 6)
    if not np.isfinite(clean).all() or not np.isfinite(degraded).all():
        return None
    values, vectors = np.linalg.eigh((clean + clean.T) / 2)
    if values[0] <= max(float(values[-1]) * rank_tolerance, 1e-14):
        return None
    inverse_sqrt = (vectors / np.sqrt(values)) @ vectors.T
    relative = inverse_sqrt @ degraded @ inverse_sqrt
    return float(np.clip(np.linalg.eigvalsh((relative + relative.T) / 2)[0], 0, 1))


def region_observation_mask(keypoints, rectangle, patch_radius=0.0):
    """Conservative patch/rectangle overlap; separate from dataset visibility."""
    points = np.asarray(keypoints, dtype=np.float64)
    x0, y0, x1, y1 = rectangle
    radius = float(patch_radius)
    if points.ndim != 2 or points.shape[1] != 2 or x1 <= x0 or y1 <= y0 or radius < 0:
        raise ValueError("invalid keypoints, rectangle or patch radius")
    overlap = ((points[:, 0] + radius >= x0) & (points[:, 0] - radius < x1)
               & (points[:, 1] + radius >= y0) & (points[:, 1] - radius < y1))
    return ~overlap & np.isfinite(points).all(axis=1)


def affected_valid_keypoint_counts(rectangles, keypoints, valid_keypoints, patch_radius=0.0):
    """Count valid in-frame point neighborhoods touched by each image-space patch."""
    points = np.asarray(keypoints, dtype=np.float64)
    valid = np.asarray(valid_keypoints, dtype=bool).reshape(-1)
    if points.shape != (len(valid), 2):
        raise ValueError("keypoints and validity mask must align")
    return [int(np.count_nonzero(valid & ~region_observation_mask(points, rect, patch_radius)))
            for rect in rectangles]


def supported_regions(rectangles, keypoints, jacobian, base_weights, *, min_ratio=0.35,
                      min_keypoints=6, patch_radius=4.0):
    """Return valid candidate indices and diagnostics without choosing easiest."""
    if not 0 <= min_ratio <= 1 or min_keypoints < 4:
        raise ValueError("invalid source geometry thresholds")
    weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    clean = information_matrix(jacobian, weights)
    if np.count_nonzero(weights) < min_keypoints or support_ratio(clean, clean) is None:
        return [], {"reason": "clean_geometry_degenerate", "ratios": []}
    accepted, ratios = [], []
    for index, rectangle in enumerate(rectangles):
        observed = region_observation_mask(keypoints, rectangle, patch_radius)
        degraded_weights = weights * observed
        ratio = support_ratio(clean, information_matrix(jacobian, degraded_weights))
        ratios.append(ratio)
        if np.count_nonzero(degraded_weights) >= min_keypoints and ratio is not None and ratio >= min_ratio:
            accepted.append(index)
    return accepted, {"reason": "supported" if accepted else "no_supported_region", "ratios": ratios}
