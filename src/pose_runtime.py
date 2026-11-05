"""Lazily load runtime dependencies shared by pose inference and training."""

from __future__ import annotations

from types import SimpleNamespace

def import_runtime_dependencies() -> SimpleNamespace:
    """延迟导入训练和推理所需依赖，便于几何工具在轻量环境下单独导入。"""
    try:
        import cv2
        import kornia.augmentation as K
        import scipy.io
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from tqdm import tqdm

        from loaders import speedplus_segmentation_precomputed as speedplus
        from src import pose_utils
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Run the project training "
            "environment with torch, kornia, scipy, opencv, torchvision, and tqdm."
        ) from exc

    return SimpleNamespace(
        cv2=cv2,
        K=K,
        scipy_io=scipy.io,
        torch=torch,
        F=F,
        DataLoader=DataLoader,
        Dataset=Dataset,
        transforms=transforms,
        tqdm=tqdm,
        speedplus=speedplus,
        pose_utils=pose_utils,
    )

