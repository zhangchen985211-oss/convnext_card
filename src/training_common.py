"""Shared SPEED+ data-loading and metric helpers for route B."""
from __future__ import annotations

import torch

from loaders import speedplus_segmentation_precomputed as speedplus


def pnp_loss_weight(config, epoch):
    warmup_epochs = max(0, int(config.get("geometry_warmup_epochs", 0)))
    ramp_epochs = max(0, int(config.get("geometry_ramp_epochs", 0)))
    pnp_target = float(config.get("lambda_pnp", 0.5))

    if epoch < warmup_epochs:
        scale = 0.0
    elif ramp_epochs > 0 and epoch < warmup_epochs + ramp_epochs:
        scale = float(epoch - warmup_epochs + 1) / float(ramp_epochs)
    else:
        scale = 1.0
    return pnp_target * scale


def normalize_val_metrics(val_metrics):
    if isinstance(val_metrics, dict):
        valid_score = val_metrics.get("valid_score", val_metrics.get("score", 1e16))
        valid_score_t = val_metrics.get("valid_score_t", val_metrics.get("score_t", 1e16))
        valid_score_r = val_metrics.get("valid_score_r", val_metrics.get("score_r", 1e16))
        penalized_score = val_metrics.get("penalized_score", valid_score)
        total_count = val_metrics.get("total_count", val_metrics.get("valid_count", 0))
        valid_count = val_metrics.get("valid_count", total_count)
        failed_count = val_metrics.get("failed_count", max(total_count - valid_count, 0))
        success_rate = val_metrics.get("success_rate", valid_count / max(total_count, 1))
        return {
            "penalized_score": penalized_score,
            "valid_score": valid_score,
            "valid_score_t": valid_score_t,
            "valid_score_r": valid_score_r,
            "success_rate": success_rate,
            "valid_count": valid_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "has_penalized_score": "penalized_score" in val_metrics,
        }

    valid_score, valid_score_t, valid_score_r = val_metrics
    return {
        "penalized_score": valid_score,
        "valid_score": valid_score,
        "valid_score_t": valid_score_t,
        "valid_score_r": valid_score_r,
        "success_rate": 1.0,
        "valid_count": 0,
        "failed_count": 0,
        "total_count": 0,
        "has_penalized_score": False,
    }


def dataset_id_from_config(config, split):
    for key in (
        f"{split}_dataset_id",
        "dataset_id",
        "style_dataset_id",
        "source_dataset_id",
    ):
        dataset_id = config.get(key, "")
        if dataset_id:
            return dataset_id
    return "synthetic"


def build_loaders(config, tforms):
    if config["isloop"]:
        if config["split_submission"] == "sunlamp":
            split = "sunlamp_train"
        elif config["split_submission"] == "lightbox":
            split = "lightbox_train"
        else:
            raise ValueError("UDA loop only supports sunlamp/lightbox")
        train_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(
            split=split,
            speed_root=config["root_dir"],
            transform_input=tforms,
            config=config,
        )
    else:
        train_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(
            split="train",
            speed_root=config["root_dir"],
            transform_input=tforms,
            config=config,
            dataset_id=dataset_id_from_config(config, "train"),
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=bool(config.get("pin_memory", True)),
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty: check labels, batch_size and drop_last")
    if config["isloop"]:
        return train_loader, None

    val_dataset_id = config.get("val_dataset_id", dataset_id_from_config(config, "train"))
    val_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(
        split="validation",
        speed_root=config["root_dir"],
        transform_input=tforms,
        config=config,
        dataset_id=val_dataset_id,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
        pin_memory=bool(config.get("pin_memory", True)),
    )
    return train_loader, val_loader
