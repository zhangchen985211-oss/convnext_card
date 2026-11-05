"""ConvNeXt-CARD model for the independent strong-backbone route."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cad_relation_decoder import CADRelationDecoder


class MultiScaleFPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, out_channels, 1) for channels in in_channels]
        )
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                )
                for _ in in_channels
            ]
        )

    def forward(self, features: Sequence[Tensor]) -> Dict[str, Tensor]:
        if len(features) != len(self.lateral):
            raise ValueError("feature count does not match FPN configuration")
        outputs: List[Optional[Tensor]] = [None] * len(features)
        top = None
        for index in reversed(range(len(features))):
            current = self.lateral[index](features[index])
            if top is not None:
                current = current + F.interpolate(
                    top, size=current.shape[-2:], mode="bilinear", align_corners=False
                )
            outputs[index] = self.refine[index](current)
            top = outputs[index]
        return {f"P{index + 2}": value for index, value in enumerate(outputs)}


class PoseHead(nn.Module):
    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
        )
        self.hm = nn.Conv2d(in_channels, num_keypoints, 1)
        self.depth = nn.Conv2d(in_channels, num_keypoints, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        hidden = self.body(x)
        return {"hm_c": self.hm(hidden), "depth": self.depth(hidden)}


class ConvNeXtFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        out_indices: Iterable[int] = (0, 1, 2, 3),
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "ConvNeXt-CARD requires timm; install the A6000 environment first"
            ) from exc
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.feature_channels = tuple(self.backbone.feature_info.channels())

    def forward(self, x: Tensor) -> List[Tensor]:
        return list(self.backbone(x))


class ConvNeXtCARD(nn.Module):
    """ConvNeXt + FPN + CAD-conditioned relational decoder."""

    def __init__(
        self,
        num_keypoints: int = 11,
        pretrained: bool = True,
        model_name: str = "convnext_tiny",
        decoder_channels: int = 128,
        adjacency: Optional[Tensor] = None,
        distances: Optional[Tensor] = None,
        relation_heads: int = 4,
        legacy_output: bool = False,
    ):
        super().__init__()
        if adjacency is None or distances is None:
            adjacency = torch.eye(num_keypoints)
            distances = torch.zeros_like(adjacency)
        if adjacency.shape != (num_keypoints, num_keypoints):
            raise ValueError("CAD adjacency must have [K,K] shape")
        self.num_keypoints = num_keypoints
        self.legacy_output = legacy_output
        self.encoder = ConvNeXtFeatureEncoder(model_name, pretrained)
        self.fpn = MultiScaleFPN(self.encoder.feature_channels, decoder_channels)
        self.coarse_head = PoseHead(decoder_channels, num_keypoints)
        self.relation_decoder = CADRelationDecoder(
            decoder_channels,
            decoder_channels,
            adjacency,
            distances,
            relation_heads,
        )
        self.fine_head = PoseHead(decoder_channels, num_keypoints)

    def forward(self, x: Tensor):
        pyramid = self.fpn(self.encoder(x))
        coarse = self.coarse_head(pyramid["P3"])
        refined, relation_tokens, coarse_coords = self.relation_decoder(
            pyramid["P2"], coarse["hm_c"]
        )
        fine = self.fine_head(refined)
        output = {
            "coarse": coarse,
            "fine": fine,
            "relation_tokens": relation_tokens,
            "coarse_coords": coarse_coords,
        }
        return [coarse, fine] if self.legacy_output else output


def get_convnext_card(config, adjacency=None, distances=None):
    return ConvNeXtCARD(
        num_keypoints=int(config.get("num_keypoints", 11)),
        pretrained=bool(config.get("pretrained", True)),
        model_name=config.get("backbone", "convnext_tiny"),
        decoder_channels=int(config.get("decoder_channels", 128)),
        adjacency=adjacency,
        distances=distances,
        relation_heads=int(config.get("relation_heads", 4)),
        legacy_output=bool(config.get("legacy_output", False)),
    )
