"""Input transforms shared by Route B training, pseudo-labeling, and evaluation."""

from __future__ import annotations

import kornia.augmentation as K
import cv2
from torchvision import transforms


def _to_rgb_if_bgr(image):
    # The shared dataset applies the same transform to the image and its
    # multi-channel heatmap. Only convert ordinary BGR images; leave heatmaps
    # untouched so ToTensor can preserve all 11 keypoint channels.
    if getattr(image, "ndim", 0) == 3 and image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def build_input_transforms(config, training: bool):
    """Use ConvNeXt's ImageNet normalization and keep geometry augmentation-free."""
    mean = config.get("input_mean", (0.485, 0.456, 0.406))
    std = config.get("input_std", (0.229, 0.224, 0.225))
    to_tensor = transforms.Compose(
        [
            transforms.Lambda(_to_rgb_if_bgr),
            transforms.ToTensor(),
            transforms.Resize((config["rows"], config["cols"])),
        ]
    )
    ops = [K.Normalize(mean=mean, std=std)]
    if training and not config.get("image_augmentation", {}).get("enabled", False):
        ops.extend(
            [
                K.RandomGaussianBlur(kernel_size=(15, 15), sigma=(0.8, 0.8), p=0.1),
                K.RandomGaussianNoise(p=1.0, std=0.005),
            ]
        )
    return to_tensor, K.AugmentationSequential(*ops)
