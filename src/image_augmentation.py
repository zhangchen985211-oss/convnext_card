"""Image-only, pixel-domain augmentation, with optional source geometry support.

Torch is imported lazily so geometry/configuration checks work without a training
environment. Heatmaps, camera parameters, poses and source visibility are never
modified here. Returned masks only describe the artificial local degradation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.augmentation_geometry import (
    affected_valid_keypoint_counts, projection_jacobian,
    region_observation_mask, supported_regions,
)


def pixel_augmentation_enabled(config):
    return bool(config.get("image_augmentation", {}).get("enabled", False))


def seed_training(config):
    import random
    import torch

    seed = int(config.get("seed", 3407))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible RNG is not a promise of bitwise deterministic CUDA kernels.
    return seed


class ImageOnlyAugmenter:
    def __init__(self, config):
        self.config = dict(config.get("image_augmentation", {}))
        self.enabled = bool(self.config.get("enabled", False))
        self.local = dict(self.config.get("local", {}))
        self.geometry = dict(self.local.get("geometry", {}))
        for key, default in (("probability", 1.0), ("blur_probability", 0.1), ("noise_probability", 0.5)):
            self._probability(self.config.get(key, default), key)
        self._probability(self.local.get("probability", 0.0), "local.probability")
        for key, default in (("gain", [0.65, 1.45]), ("gamma", [0.75, 1.35]),
                             ("contrast", [0.8, 1.2]), ("noise_std", [0.0, 0.015])):
            lo, hi = self.config.get(key, default)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi) or (key != "noise_std" and lo == 0):
                raise ValueError(f"invalid image_augmentation.{key} range")
        lo, hi = self.local.get("area", [0.02, 0.12])
        if not 0 < lo <= hi < 1:
            raise ValueError("local.area must satisfy 0 < min <= max < 1")
        lo, hi = self.local.get("aspect", [0.5, 2.0])
        if not 0 < lo <= hi or not math.isfinite(hi):
            raise ValueError("invalid local.aspect")
        lo, hi = self.local.get("strength", [0.5, 0.9])
        if not 0 <= lo <= hi <= 1:
            raise ValueError("invalid local.strength")
        if int(self.local.get("candidates", 4)) < 1:
            raise ValueError("local.candidates must be >= 1")
        radius = float(self.local.get("patch_radius", 4.0))
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("local.patch_radius must be finite and nonnegative")
        if not 0 <= float(self.geometry.get("min_support_ratio", 0.35)) <= 1:
            raise ValueError("geometry.min_support_ratio must be in [0,1]")
        if int(self.geometry.get("min_keypoints", 6)) < 4:
            raise ValueError("geometry.min_keypoints must be >= 4")

    @staticmethod
    def _probability(value, name):
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be in [0,1]")

    @staticmethod
    def _draw(reference, low=0.0, high=1.0, generator=None):
        import torch
        return low + (high - low) * float(torch.rand((), device=reference.device, generator=generator).item())

    def _range(self, image, key, default, generator):
        return self._draw(image, *self.config.get(key, default), generator=generator)

    def _rectangles(self, image, generator):
        height, width = image.shape[-2:]
        rectangles = []
        for _ in range(int(self.local.get("candidates", 4))):
            area = self._draw(image, *self.local.get("area", [0.02, 0.12]), generator=generator) * height * width
            aspect = math.exp(self._draw(image, *np.log(self.local.get("aspect", [0.5, 2.0])), generator=generator))
            rect_w = min(width, max(1, round(math.sqrt(area * aspect))))
            rect_h = min(height, max(1, round(math.sqrt(area / aspect))))
            left = min(width - rect_w, int(self._draw(image, 0, width - rect_w + 1, generator)))
            top = min(height - rect_h, int(self._draw(image, 0, height - rect_h + 1, generator)))
            rectangles.append((left, top, left + rect_w, top + rect_h))
        return rectangles

    def __call__(self, images, *, keypoints=None, visible=None, camera_points=None,
                 translation=None, camera_matrix=None, cad_scale=None,
                 source_geometry=False, generator=None):
        import torch
        import torch.nn.functional as F

        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("expected floating point images [B,3,H,W] in pixel domain")
        batch, _, height, width = images.shape
        masks = torch.zeros((batch, 1, height, width), device=images.device, dtype=torch.bool)
        observations = None if keypoints is None else torch.ones(keypoints.shape[:2], device=images.device, dtype=torch.bool)
        stats = {"photometric_applied": 0, "local_applied": 0, "geometry_fallback": 0,
                 "geometry_skipped_target": 0, "support_ratios": [], "affected_valid_keypoints": [],
                 "background_only_regions": 0, "background_only_candidates": 0,
                 "local_no_keypoint_candidate": 0, "fallback_reasons": {}}
        if not self.enabled:
            return images, observations, masks, stats
        result = images.clone()
        for index in range(batch):
            image = result[index:index + 1]
            if self._draw(image, generator=generator) < float(self.config.get("probability", 1.0)):
                gain = self._range(image, "gain", [0.65, 1.45], generator)
                gamma = self._range(image, "gamma", [0.75, 1.35], generator)
                contrast = self._range(image, "contrast", [0.8, 1.2], generator)
                center = image.mean(dim=(-2, -1), keepdim=True)
                image = ((image - center) * contrast + center).clamp(0, 1)
                image = (image.pow(gamma) * gain).clamp(0, 1)
                if self._draw(image, generator=generator) < float(self.config.get("blur_probability", 0.1)):
                    sigma = self._draw(image, 0.4, 1.2, generator)
                    axis = torch.arange(-3, 4, device=image.device, dtype=image.dtype)
                    kernel = torch.exp(-axis.square() / (2 * sigma**2))
                    kernel = kernel / kernel.sum()
                    kernel = (kernel[:, None] * kernel[None, :]).expand(3, 1, 7, 7)
                    image = F.conv2d(F.pad(image, (3, 3, 3, 3), mode="replicate"), kernel, groups=3)
                if self._draw(image, generator=generator) < float(self.config.get("noise_probability", 0.5)):
                    std = self._range(image, "noise_std", [0, 0.015], generator)
                    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
                    image = (image + noise * std).clamp(0, 1)
                stats["photometric_applied"] += 1

            apply_local = self._draw(image, generator=generator) < float(self.local.get("probability", 0.0))
            if apply_local and self.local.get("source_only", False) and not source_geometry:
                stats["geometry_skipped_target"] += 1
                apply_local = False
            if apply_local:
                candidates = self._rectangles(image, generator)
                diagnostic = None
                points = None if keypoints is None else keypoints[index].detach().cpu().numpy()
                if points is None:
                    valid = list(range(len(candidates)))
                    valid_points = None
                else:
                    valid_points = np.isfinite(points).all(axis=1)
                    valid_points &= (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
                    valid_points &= (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
                    if visible is not None:
                        valid_points &= visible[index].detach().cpu().numpy().astype(bool)
                    patch_radius = float(self.local.get("patch_radius", 4.0)) * min(height, width) / 512
                    hit_counts = affected_valid_keypoint_counts(candidates, points, valid_points, patch_radius)
                    valid = [candidate_id for candidate_id, count in enumerate(hit_counts) if count > 0]
                    stats["background_only_candidates"] += sum(count == 0 for count in hit_counts)
                if self.geometry.get("enabled", False):
                    if not source_geometry:
                        raise ValueError("geometry filtering requires labeled synthetic source data")
                    if any(item is None for item in (keypoints, camera_points, translation, camera_matrix, cad_scale)):
                        raise ValueError("source geometry filtering requires points, translation, K and CAD scale")
                    camera = camera_matrix[min(index, len(camera_matrix) - 1)].detach().cpu().numpy()
                    xyz = camera_points[index].detach().cpu().numpy()
                    origin = translation[index].detach().cpu().numpy()
                    base = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
                    base &= (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
                    if visible is not None:
                        base &= visible[index].detach().cpu().numpy().astype(bool)
                    try:
                        jacobian = projection_jacobian(xyz, origin, camera, (height, width), cad_scale)
                        valid, diagnostic = supported_regions(
                            candidates, points, jacobian, base,
                            min_ratio=float(self.geometry.get("min_support_ratio", 0.35)),
                            min_keypoints=int(self.geometry.get("min_keypoints", 6)),
                            patch_radius=float(self.local.get("patch_radius", 4.0)) * min(height, width) / 512,
                        )
                        # A local degradation must touch at least one valid
                        # source point; support-preserving background patches
                        # do not provide the intended geometry challenge.
                        valid = [candidate_id for candidate_id in valid
                                 if points is None or hit_counts[candidate_id] > 0]
                        if not valid and diagnostic["reason"] == "supported":
                            diagnostic = {"reason": "no_keypoint_region", "ratios": diagnostic["ratios"]}
                    except ValueError:
                        valid, diagnostic = [], {"reason": "invalid_source_geometry", "ratios": []}
                if valid:
                    chosen = valid[min(len(valid) - 1, int(self._draw(image, 0, len(valid), generator)))]
                    left, top, right, bottom = candidates[chosen]
                    strength = self._draw(image, *self.local.get("strength", [0.5, 0.9]), generator=generator)
                    fill = 0.0 if self._draw(image, generator=generator) < 0.5 else 1.0
                    image = image.clone()
                    image[..., top:bottom, left:right] = image[..., top:bottom, left:right] * (1 - strength) + fill * strength
                    masks[index, :, top:bottom, left:right] = True
                    if observations is not None:
                        observed = region_observation_mask(points, candidates[chosen], float(self.local.get("patch_radius", 4.0)) * min(height, width) / 512)
                        observations[index] = torch.as_tensor(observed, device=images.device)
                        affected = int(np.count_nonzero(valid_points & ~observed))
                        stats["affected_valid_keypoints"].append(affected)
                        stats["background_only_regions"] += int(affected == 0)
                    stats["local_applied"] += 1
                    if diagnostic is not None:
                        stats["support_ratios"].append(diagnostic["ratios"][chosen])
                else:
                    if points is not None and not any(hit_counts):
                        stats["local_no_keypoint_candidate"] += 1
                    if self.geometry.get("enabled", False):
                        stats["geometry_fallback"] += 1
                        reason = diagnostic["reason"]
                        stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + 1
            result[index:index + 1] = image.clamp(0, 1)
        return result, observations, masks, stats


def save_augmentation_preview(directory, original, augmented, region_masks, stats, *, step, bgr=False):
    """Save one before/after/mask PNG plus numeric diagnostics per selected step."""
    from PIL import Image

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    def rgb(tensor):
        array = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        if bgr:
            array = array[..., ::-1]
        return (array * 255).round().astype(np.uint8)
    before, after = rgb(original[0]), rgb(augmented[0])
    mask = region_masks[0, 0].detach().cpu().numpy()
    overlay = after.copy()
    overlay[mask] = (0.5 * overlay[mask] + np.array([127.5, 0, 0])).astype(np.uint8)
    Image.fromarray(np.concatenate((before, after, overlay), axis=1)).save(destination / f"step_{step:06d}.png")
    (destination / f"step_{step:06d}.json").write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8")
