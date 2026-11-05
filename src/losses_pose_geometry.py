"""Small, independently testable losses for the ConvNeXt-CARD route."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def heatmap_mse(pred: Tensor, target: Tensor, visible: Optional[Tensor] = None):
    if visible is None:
        return F.mse_loss(pred, target)
    mask = visible.to(device=pred.device, dtype=pred.dtype).reshape(-1, pred.shape[1], 1, 1)
    numerator = ((pred - target).square() * mask).sum()
    denominator = mask.sum() * pred.shape[-2] * pred.shape[-1]
    return numerator / denominator.clamp_min(1.0)


def coarse_fine_consistency(
    coarse_coords: Tensor, fine_coords: Tensor
) -> Tensor:
    if coarse_coords.shape != fine_coords.shape:
        raise ValueError("coarse and fine coordinates must have equal shape")
    return F.smooth_l1_loss(coarse_coords, fine_coords)


def cad_edge_loss(
    points_3d_pred: Tensor,
    cad_points_3d: Tensor,
    edges: Tensor,
) -> Tensor:
    if points_3d_pred.shape != cad_points_3d.shape:
        raise ValueError("predicted and CAD points must have equal shape")
    edge_index = edges.to(device=points_3d_pred.device, dtype=torch.long)
    pred_edges = points_3d_pred[:, edge_index[:, 0]] - points_3d_pred[:, edge_index[:, 1]]
    cad_edges = cad_points_3d[:, edge_index[:, 0]] - cad_points_3d[:, edge_index[:, 1]]
    pred_lengths = pred_edges.norm(dim=-1)
    cad_lengths = cad_edges.norm(dim=-1)
    pred_scale = pred_lengths.detach().median(dim=-1, keepdim=True).values.clamp_min(1e-6)
    cad_scale = cad_lengths.detach().median(dim=-1, keepdim=True).values.clamp_min(1e-6)
    return F.smooth_l1_loss(pred_lengths / pred_scale, cad_lengths / cad_scale)


