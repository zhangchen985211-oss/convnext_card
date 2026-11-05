"""Dataset and camera helpers for labeled target-split evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from loaders import speedplus_segmentation_precomputed as speedplus


class JsonPoseDataset(Dataset):
    """Read images and poses from a labeled SPEED+ split JSON."""

    def __init__(self, dataset_dir, gt_json_path, transform_input):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.gt_json_path = Path(gt_json_path)
        self.transform_input = transform_input
        with self.gt_json_path.open("r", encoding="utf-8") as fp:
            gt_items = json.load(fp)
        self.items = [
            {
                "filename": item["filename"],
                "q": np.array(item["q_vbs2tango_true"], dtype=np.float32),
                "r": np.array(item["r_Vo2To_vbs_true"], dtype=np.float32),
            }
            for item in gt_items
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image_path = self.image_dir / item["filename"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return {
            "image": self.transform_input(image),
            "filename": item["filename"],
            "q0": item["q"],
            "r0": item["r"],
        }


def load_scaled_camera(dataset_root, rows, cols):
    """Read camera calibration and scale intrinsics to model input size."""
    cam = speedplus.Camera(str(Path(dataset_root)))
    k_mat = cam.K.copy()
    k_mat[0, :] *= cols / 1920.0
    k_mat[1, :] *= rows / 1200.0
    return k_mat.astype(np.float32), np.asarray(cam.dcoef, dtype=np.float32)
