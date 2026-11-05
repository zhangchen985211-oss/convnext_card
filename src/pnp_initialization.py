"""Checked OpenCV initialization for the differentiable training PnP branch."""

from __future__ import annotations

import cv2
import numpy as np


def checked_pnp_initialization(points_2d, world, camera):
    points = np.ascontiguousarray(points_2d, dtype=np.float64).reshape(-1, 2)
    world = np.ascontiguousarray(world, dtype=np.float64).reshape(-1, 3)
    camera = np.ascontiguousarray(camera, dtype=np.float64).reshape(3, 3)
    if len(points) != len(world) or len(points) < 6:
        return None, "insufficient_correspondences"
    if not all(np.isfinite(item).all() for item in (points, world, camera)):
        return None, "non_finite_input"
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-6) < 2:
        return None, "collapsed_image_points"
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world, points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE,
            confidence=0.9999, reprojectionError=3.0,
        )
        if not success or rvec is None or tvec is None or inliers is None or len(inliers) < 6:
            return None, "ransac_failed"
        success, rvec, tvec = cv2.solvePnP(
            world, points, camera, None, rvec=rvec, tvec=tvec,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or rvec is None or tvec is None:
            return None, "refinement_failed"
        pose = np.concatenate((rvec.reshape(3), tvec.reshape(3)))
        if not np.isfinite(pose).all():
            return None, "non_finite_pose"
        xyz = world @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(3)
        if not np.all(xyz[:, 2] > 1e-6):
            return None, "non_positive_depth"
        _, jacobian = cv2.projectPoints(world, rvec, tvec, camera, None)
        values = np.linalg.svd(jacobian[:, :6], compute_uv=False)
        if values[-1] <= values[0] * 1e-8:
            return None, "degenerate_pose_jacobian"
        return pose.astype(np.float32), "accepted"
    except cv2.error:
        return None, "opencv_error"
