"""CAD-conditioned keypoint relation decoding for ConvNeXt-CARD."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CADRelationAttention(nn.Module):
    """Sparse self-attention with CAD-distance relative bias."""

    def __init__(self, dim: int, num_heads: int = 4, hidden_dim: int = 32):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.distance_bias = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_heads)
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, tokens: Tensor, adjacency: Tensor, distances: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != adjacency.shape[0]:
            raise ValueError("tokens must be [B,K,C] and match CAD graph size")
        bsz, points, _ = tokens.shape
        residual = tokens
        qkv = self.qkv(self.norm(tokens)).reshape(
            bsz, points, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self.distance_bias(
            distances.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        )
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        graph_mask = adjacency.to(device=tokens.device, dtype=torch.bool)
        logits = (logits + bias).masked_fill(
            ~graph_mask.unsqueeze(0).unsqueeze(0), -1e4
        )
        updated = torch.matmul(torch.softmax(logits, dim=-1), v)
        updated = updated.transpose(1, 2).reshape(bsz, points, self.dim)
        updated = residual + self.out(updated)
        return updated + self.ffn(updated)


class CADRelationDecoder(nn.Module):
    """Refine FPN features around coarse keypoint locations."""

    def __init__(
        self,
        feature_dim: int,
        token_dim: int,
        adjacency: Tensor,
        distances: Tensor,
        num_heads: int = 4,
        gaussian_sigma: float = 2.0,
    ):
        super().__init__()
        if adjacency.shape != distances.shape:
            raise ValueError("adjacency and distances must have equal shape")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("CAD graph must be square")
        self.num_points = adjacency.shape[0]
        self.gaussian_sigma = float(gaussian_sigma)
        self.register_buffer("adjacency", adjacency.float(), persistent=True)
        self.register_buffer("distances", distances.float(), persistent=True)
        self.token_in = nn.Linear(feature_dim, token_dim)
        self.attention = CADRelationAttention(token_dim, num_heads=num_heads)
        self.token_out = nn.Linear(token_dim, feature_dim)
        self.feature_norm = nn.GroupNorm(8, feature_dim)
        self.feature_out = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )

    @staticmethod
    def _soft_argmax(heatmap: Tensor) -> Tensor:
        bsz, points, height, width = heatmap.shape
        probs = torch.softmax(heatmap.flatten(2), dim=-1)
        ys = torch.linspace(0, 1, height, device=heatmap.device, dtype=heatmap.dtype)
        xs = torch.linspace(0, 1, width, device=heatmap.device, dtype=heatmap.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        x = (probs * xx.reshape(1, 1, -1)).sum(-1)
        y = (probs * yy.reshape(1, 1, -1)).sum(-1)
        return torch.stack((x, y), dim=-1).reshape(bsz, points, 2)

    def _extract_tokens(self, feature: Tensor, heatmap: Tensor):
        coords = self._soft_argmax(heatmap)
        grid = coords.mul(2.0).sub(1.0).unsqueeze(2)
        tokens = F.grid_sample(
            feature, grid, mode="bilinear", align_corners=True
        ).squeeze(-1).transpose(1, 2)
        return tokens, coords

    def _scatter_tokens(self, feature: Tensor, tokens: Tensor, coords: Tensor):
        _, _, height, width = feature.shape
        ys = torch.linspace(0, 1, height, device=feature.device, dtype=feature.dtype)
        xs = torch.linspace(0, 1, width, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).reshape(1, 1, height, width, 2)
        delta = grid - coords.unsqueeze(2).unsqueeze(2)
        sigma = max(self.gaussian_sigma / max(height, width), 1e-4)
        weights = torch.exp(-(delta.square().sum(-1)) / (2 * sigma * sigma))
        weights = weights / (weights.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        values = self.token_out(tokens).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        # weights: [B,K,H,W] -> [B,1,K,H,W], values: [B,C,K,1,1].
        # The previous ordering broadcast the batch/channel dimensions incorrectly.
        return feature + (values * weights.unsqueeze(1)).sum(dim=2)

    def forward(self, feature: Tensor, coarse_heatmap: Tensor):
        if coarse_heatmap.shape[1] != self.num_points:
            raise ValueError("heatmap channel count does not match CAD graph")
        tokens, coords = self._extract_tokens(feature, coarse_heatmap)
        tokens = self.attention(
            self.token_in(tokens), self.adjacency, self.distances
        )
        refined = self._scatter_tokens(feature, tokens, coords)
        return self.feature_out(self.feature_norm(refined)), tokens, coords
