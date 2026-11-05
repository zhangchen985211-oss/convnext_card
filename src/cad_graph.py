"""CAD keypoint graph utilities used by ConvNeXt-CARD."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def build_knn_graph(points_3d: np.ndarray, knn: int = 2):
    points = np.asarray(points_3d, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape [K,3]")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    scale = float(np.median(distances[distances > 0]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    adjacency = np.eye(len(points), dtype=np.float32)
    for i in range(len(points)):
        neighbors = np.argsort(distances[i])[1 : knn + 1]
        adjacency[i, neighbors] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    normalized = distances / scale
    return adjacency, normalized


def build_speedplus_graph(points_3d: np.ndarray):
    """Use the documented SPEED+ 11-keypoint spacecraft topology."""
    points = np.asarray(points_3d, dtype=np.float32)
    if points.shape != (11, 3):
        raise ValueError(f"expected SPEED+ points [11,3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("CAD points must contain only finite values")
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 7), (1, 4), (2, 5), (3, 6),
        (6, 8), (4, 9), (5, 10),
    )
    adjacency = np.eye(11, dtype=np.float32)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    nonzero = distances[distances > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    distances = distances / max(scale, 1e-8)
    return adjacency, distances


def load_kpts_mat(path: str | Path) -> np.ndarray:
    """Read the 11 CAD points in either MATLAB orientation; return [11,3]."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to read kpts.mat") from exc
    data = loadmat(str(path))
    if "corners" in data:
        points = np.asarray(data["corners"], dtype=np.float32)
    else:
        candidates = [
            np.asarray(value, dtype=np.float32)
            for key, value in data.items()
            if not key.startswith("__")
            and np.asarray(value).shape in ((3, 11), (11, 3))
        ]
        if len(candidates) != 1:
            raise KeyError(
                f"expected an unambiguous 11x3 `corners` array in {path}; "
                f"found {len(candidates)} matching arrays"
            )
        points = candidates[0]
    if points.shape == (3, 11):
        points = points.T
    elif points.shape != (11, 3):
        raise ValueError(f"expected CAD points [3,11] or [11,3] in {path}, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"CAD points in {path} must contain only finite values")
    return np.ascontiguousarray(points, dtype=np.float32)


def save_graph(path: str | Path, points_3d: np.ndarray, adjacency, distances):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = [
        [int(i), int(j)]
        for i in range(len(points_3d))
        for j in range(i + 1, len(points_3d))
        if adjacency[i, j] > 0
    ]
    payload = {
        "num_keypoints": int(len(points_3d)),
        "points_3d": np.asarray(points_3d, dtype=float).tolist(),
        "edges": edges,
        "adjacency": np.asarray(adjacency, dtype=float).tolist(),
        "distances": np.asarray(distances, dtype=float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_graph(path: str | Path, expected_points: np.ndarray | None = None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("num_keypoints") != 11:
        raise ValueError("SPEED CAD graph must declare exactly 11 keypoints")
    points = np.asarray(payload["points_3d"], dtype=np.float32)
    adjacency = np.asarray(payload["adjacency"], dtype=np.float32)
    distances = np.asarray(payload["distances"], dtype=np.float32)
    if points.shape != (11, 3):
        raise ValueError("SPEED CAD graph points must have shape [11,3]")
    if adjacency.shape != (11, 11) or distances.shape != (11, 11):
        raise ValueError("CAD graph matrices must have shape [11,11]")
    if not all(np.isfinite(array).all() for array in (points, adjacency, distances)):
        raise ValueError("CAD graph arrays must contain only finite values")
    if expected_points is not None:
        expected = np.asarray(expected_points, dtype=np.float32)
        if expected.shape != (11, 3) or not np.isfinite(expected).all():
            raise ValueError("expected CAD points must be finite with shape [11,3]")
        if not np.allclose(points, expected, rtol=1e-6, atol=1e-6):
            raise ValueError(f"CAD graph points in {path} do not match the active kpts.mat")
    expected_adjacency, expected_distances = build_speedplus_graph(points)
    if not np.array_equal(adjacency, expected_adjacency):
        raise ValueError(f"CAD graph topology in {path} does not match the Route B topology")
    if not np.allclose(distances, expected_distances, rtol=1e-6, atol=1e-6):
        raise ValueError(f"CAD graph distances in {path} do not match its 3D points")
    return points, adjacency, distances
