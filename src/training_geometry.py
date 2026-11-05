"""Differentiable PnP with explicit failed-sample handling for route training."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from BPnP import BPnP
from src.pnp_initialization import checked_pnp_initialization


def bpnp_reprojection_loss(points, target, world, camera, image_size, visible=None):
    """Bad initializations contribute zero PnP loss, preserving other supervision.

    BPnP receives only finite, visible, in-frame correspondences and a valid
    initial pose. The same subset is used for initialization and supervision.
    Each sample is refined independently so a failed sample cannot suppress the
    geometry loss and gradients of other valid samples in the same batch.
    """
    height, width = image_size
    if points.ndim != 3 or points.shape[-1] != 2 or target.shape != points.shape:
        raise ValueError("points and target must have equal [B,K,2] shape")
    if world.shape != (points.shape[1], 3):
        raise ValueError("world must have shape [K,3]")
    if visible is not None and visible.shape != points.shape[:2]:
        raise ValueError("visible must have shape [B,K]")
    if min(height, width) <= 1:
        raise ValueError("image dimensions must be greater than one")
    scale = points.new_tensor([max(width - 1, 1), max(height - 1, 1)])
    valid_points = torch.isfinite(target).all(dim=-1) & torch.isfinite(points).all(dim=-1)
    valid_points &= (target[..., 0] >= 0) & (target[..., 0] <= width - 1)
    valid_points &= (target[..., 1] >= 0) & (target[..., 1] <= height - 1)
    if visible is not None:
        valid_points &= visible.to(device=points.device, dtype=torch.bool)
    indices, projections, selected_masks, losses = [], [], [], []
    stats = {"total": len(points), "valid": 0, "failed": 0, "reasons": {}}
    camera_numpy = camera.detach().cpu().numpy()
    def failed(reason):
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

    for index, mask in enumerate(valid_points):
        if int(mask.sum().item()) < 6:
            failed("insufficient_visible_correspondences")
            continue
        observed = points[index, mask]
        selected_world = world[mask]
        selected_world_numpy = selected_world.detach().cpu().numpy()
        pose, reason = checked_pnp_initialization(
            observed.detach().cpu().numpy(), selected_world_numpy, camera_numpy
        )
        if pose is None:
            failed(reason)
            continue
        initial = torch.as_tensor(pose[None], device=points.device, dtype=points.dtype)
        try:
            rt = BPnP.BPnP_fast.apply(observed.unsqueeze(0), selected_world, camera, initial)
        except (cv2.error, ValueError):
            failed("bpnp_refinement_failed")
            continue
        if not bool(torch.isfinite(rt).all()):
            failed("bpnp_non_finite")
            continue
        refined = rt.detach().cpu().numpy()[0]
        xyz = selected_world_numpy @ cv2.Rodrigues(refined[:3])[0].T + refined[3:]
        if not np.all(xyz[:, 2] > 1e-6):
            failed("bpnp_non_positive_depth")
            continue
        projected = BPnP.batch_project(rt, selected_world, camera)
        if not bool(torch.isfinite(projected).all()):
            failed("bpnp_projection_non_finite")
            continue
        indices.append(index)
        projections.append(projected)
        selected_masks.append(mask)
        # Clipping the projection would suppress the gradient needed to bring
        # an incorrect pose back into the image. Only the target validity mask
        # controls which correspondences contribute.
        projected_norm = projected[0] / scale
        target_norm = target[index, mask] / scale
        predicted_norm = observed / scale
        loss = torch.nn.functional.mse_loss(projected_norm, target_norm)
        losses.append(loss + torch.nn.functional.mse_loss(projected_norm, predicted_norm))
    # Do not manufacture a valid pose for an invalid sample.
    zero = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0).mul(0.0).sum()
    projected_all = target.detach().clone()
    if not indices:
        stats["failed"] = stats["total"]
        return zero, projected_all, stats
    stats["valid"] = len(indices)
    stats["failed"] = stats["total"] - stats["valid"]
    loss = torch.stack(losses).mean()
    for index, mask, projected in zip(indices, selected_masks, projections):
        projected_all[index, mask] = projected[0].detach()
    return loss, projected_all, stats
