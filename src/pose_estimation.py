"""Shared full-coverage pose protocol for validation and public evaluation.

This entry point deliberately reuses the official evaluator's solver and defaults.
It must not be used to accept training pseudo labels, which have stricter gates.
Deep-learning dependencies are loaded only when a runtime is not supplied.
"""

from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=1)
def _default_runtime():
    from src.pose_runtime import import_runtime_dependencies

    return import_runtime_dependencies()


def estimate_pose(
    sample_outputs,
    image_tensor,
    world_kpts,
    k_mat,
    config=None,
    *,
    runtime=None,
    topk=None,
    reprojection_error=None,
    fallback_topks=None,
    fallback_reprojection_errors=None,
    min_inliers=None,
):
    """Return ``(selected_pose, reason)`` using the shared evaluation protocol.

    Explicit arguments override configuration. The base pixel threshold otherwise
    comes from ``eval_reprojection_error``; search, LM refinement and reference
    resolution are forwarded unchanged in ``config``. Defaults (top-k 7, rescue
    top-k 8/9/10/11, thresholds 1/2/3/4) have one source in the official solver.
    A supplied runtime may use NumPy/OpenCV adapters without importing torch.
    """
    from src.pose_solver import (
        DEFAULT_FALLBACK_REPROJECTION_ERRORS,
        DEFAULT_FALLBACK_TOPKS,
        DEFAULT_MIN_INLIERS,
        DEFAULT_TOPK,
        estimate_pose_with_fallback,
    )

    config = config or {}
    base_error = float(
        reprojection_error
        if reprojection_error is not None
        else config.get("eval_reprojection_error", DEFAULT_FALLBACK_REPROJECTION_ERRORS[0])
    )
    if not math.isfinite(base_error) or base_error <= 0:
        raise ValueError("Evaluation reprojection error must be finite and positive")
    configured_errors = (
        DEFAULT_FALLBACK_REPROJECTION_ERRORS
        if fallback_reprojection_errors is None
        else fallback_reprojection_errors
    )
    # Preserve the public evaluator's order: the requested base threshold first,
    # followed by the configured rescue thresholds without a repeated base.
    errors = [base_error] + [float(value) for value in configured_errors if float(value) != base_error]
    if runtime is None:
        runtime = _default_runtime()
    return estimate_pose_with_fallback(
        runtime,
        sample_outputs,
        image_tensor,
        world_kpts,
        k_mat,
        DEFAULT_TOPK if topk is None else int(topk),
        DEFAULT_FALLBACK_TOPKS if fallback_topks is None else fallback_topks,
        errors,
        DEFAULT_MIN_INLIERS if min_inliers is None else int(min_inliers),
        solver_config=config,
    )
