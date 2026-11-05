"""Standalone ConvNeXt-CARD trainer for the SPEED+ project.

It keeps the existing dataset, camera, BPnP and official validation protocol,
while replacing only the feature encoder and pose decoder. The model is trained
with visible-keypoint heatmap supervision, differentiable PnP reprojection and
coarse/fine CAD-conditioned coordinate consistency.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_common import build_loaders, dataset_id_from_config, normalize_val_metrics, pnp_loss_weight  # noqa: E402
from src import pose_utils  # noqa: E402
from src.image_augmentation import ImageOnlyAugmenter, save_augmentation_preview, seed_training  # noqa: E402
from src.run_paths import (  # noqa: E402
    config_output_dir,
    mark_training_complete,
    normalize_config_paths,
    parse_config_iteration,
    project_config_arg,
    replace_config_iteration,
)
from src.training_geometry import bpnp_reprojection_loss  # noqa: E402
from src.checkpoints import load_model_checkpoint  # noqa: E402

from models.poseconvnext import get_convnext_card  # noqa: E402
from src.cad_graph import (  # noqa: E402
    build_speedplus_graph,
    load_graph,
    load_kpts_mat,
    save_graph,
)
from src.losses_pose_geometry import (  # noqa: E402
    cad_edge_loss,
    coarse_fine_consistency,
    heatmap_mse,
)
from src.input_transforms import build_input_transforms  # noqa: E402


def _load_graph(config):
    points = load_kpts_mat(Path(config["root_dir"]) / "kpts.mat")
    graph_path = config.get("cad_graph") or config.get("graph_path")
    if graph_path and Path(graph_path).exists():
        points, adjacency, distances = load_graph(graph_path, expected_points=points)
        return points, adjacency, distances
    adjacency, distances = build_speedplus_graph(points)
    if graph_path:
        save_graph(graph_path, points, adjacency, distances)
    return points, adjacency, distances


def _resolve_loop_checkpoint(config_path, config):
    if not config.get("isloop", False):
        return config.get("path_pretrain", "")
    current = parse_config_iteration(config_path)
    previous = replace_config_iteration(config_path, max(current - 1, 0))
    return str(config_output_dir(config, previous, PROJECT_ROOT) / "ckpt" / "init.pth")


def _build_optimizer(config, model):
    backbone, decoder, heads = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            backbone.append(parameter)
        elif name.startswith(("coarse_head.", "fine_head.")):
            heads.append(parameter)
        else:
            decoder.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(config.get("backbone_lr", 1e-5))},
            {"params": decoder, "lr": float(config.get("decoder_lr", 1e-4))},
            {"params": heads, "lr": float(config.get("head_lr", config.get("decoder_lr", 1e-4)))},
        ],
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def _load_checkpoint(model, path, device):
    if not path:
        return
    load_model_checkpoint(model, path, device)


def _normalized_coordinates(points, height, width):
    scale = points.new_tensor([max(width - 1, 1), max(height - 1, 1)])
    return points / scale


def _train_epoch(model, loader, optimizer, scaler, config, geometry, cad_points_numpy, aug_intensity, device, writer, epoch, image_augmenter=None):
    model.train(True)
    k_mat, dist, world = geometry
    # lambda_pnp is the final coefficient; apply the warm-up/ramp once.
    pnp_weight = pnp_loss_weight(config, epoch)
    running = 0.0
    accumulation = max(int(config.get("gradient_accumulation_steps", 1)), 1)
    optimizer.zero_grad(set_to_none=True)
    source_geometry = not config.get("isloop", False) and dataset_id_from_config(config, "train").startswith("synthetic")
    cad_scale = float(((cad_points_numpy.max(axis=0) - cad_points_numpy.min(axis=0)) ** 2).sum() ** 0.5)
    for step, data in enumerate(tqdm(loader, ncols=80)):
        images = data["image"].to(device)
        heatmap_gt = data["heatmap"].to(device)
        kpts_gt = data["kpts_2Dim"].to(device)
        visible = data["visible_kpts"].to(device)
        keypoints_3d_cam = data["kpts_3Dcam"].to(device)
        if image_augmenter is not None and image_augmenter.enabled:
            raw_images = pose_utils.undistort_for_model(images, k_mat, dist)
            pixel_images, _, region_masks, augmentation_stats = image_augmenter(
                raw_images, keypoints=kpts_gt, visible=visible,
                camera_points=keypoints_3d_cam.transpose(1, 2), translation=data["r0"].to(device),
                camera_matrix=k_mat, cad_scale=cad_scale, source_geometry=source_geometry,
            )
            images = aug_intensity(pixel_images)
            if writer is not None and epoch == int(config.get("start_epoch", 0)) and step < int(config["image_augmentation"].get("preview_count", 2)):
                save_augmentation_preview(
                    Path(writer.log_dir) / "augmentations", raw_images, pixel_images,
                    region_masks, augmentation_stats, step=step,
                )
            if writer is not None and step % int(config.get("save_tensorboard", 100)) == 0:
                for name in ("local_applied", "geometry_fallback", "geometry_skipped_target",
                             "background_only_candidates", "local_no_keypoint_candidate"):
                    writer.add_scalar(f"Augmentation/{name}", augmentation_stats[name], epoch * len(loader) + step)
                writer.add_scalar("Augmentation/affected_valid_keypoints", sum(augmentation_stats["affected_valid_keypoints"]), epoch * len(loader) + step)
                if augmentation_stats["support_ratios"]:
                    writer.add_scalar("Augmentation/min_support_ratio", min(augmentation_stats["support_ratios"]), epoch * len(loader) + step)
        else:
            images = pose_utils.preprocess_for_model(images, aug_intensity, k_mat, dist, config)
        with torch.cuda.amp.autocast(enabled=bool(config.get("amp", True))):
            output = model(images)
            coarse, fine = output["coarse"], output["fine"]
            coarse_hm = F.interpolate(coarse["hm_c"], size=heatmap_gt.shape[-2:], mode="bilinear", align_corners=False)
            fine_hm = F.interpolate(fine["hm_c"], size=heatmap_gt.shape[-2:], mode="bilinear", align_corners=False)
            target_scale = float(config.get("heatmap_target_scale", 100.0))
            loss_scale = float(config.get("heatmap_loss_scale", 0.1))
            scaled_target = heatmap_gt * target_scale
            heatmap_loss = loss_scale * (
                0.4 * heatmap_mse(coarse_hm, scaled_target, visible)
                + 0.6 * heatmap_mse(fine_hm, scaled_target, visible)
            )
        # Keep soft-argmax and all geometry outside AMP. Casting only the
        # BPnP inputs is insufficient while autocast remains enabled.
        with torch.cuda.amp.autocast(enabled=False):
            kpts_pred = pose_utils.heatmap_to_points(fine_hm.float())
            pnp_loss = torch.zeros((), device=device)
            if config.get("activate_lpnp", True) and pnp_weight > 0:
                # BPnP is kept in float32 because its custom backward is not
                # guaranteed to support half precision on every CUDA build.
                pnp_loss, _, pnp_stats = bpnp_reprojection_loss(
                    kpts_pred.float(), kpts_gt.float(), world[0].float(), k_mat[0].float(),
                    (config["rows"], config["cols"]), visible=visible,
                )
                if writer is not None and step % int(config.get("save_tensorboard", 100)) == 0:
                    writer.add_scalar("Geometry/PnP_failed", pnp_stats["failed"], epoch * len(loader) + step)
            fine_coords = _normalized_coordinates(kpts_pred, config["rows"], config["cols"])
            # Both stages use the same upsampling and coordinate extraction
            # convention; native feature-grid endpoints have a different origin.
            coarse_kpts = pose_utils.heatmap_to_points(coarse_hm.float())
            coarse_coords = _normalized_coordinates(coarse_kpts, config["rows"], config["cols"])
            relation_loss = coarse_fine_consistency(coarse_coords, fine_coords)
            depth_map = F.interpolate(
                fine["depth"].float(), size=(config["rows"], config["cols"]), mode="bilinear", align_corners=False
            )
            # Supervise the predicted camera-space z values from the labeled
            # synthetic poses. The root dataset emits [B,3,K] camera points.
            gt_depth = keypoints_3d_cam[:, 2, :].float()
            gt_coords = _normalized_coordinates(kpts_gt, config["rows"], config["cols"])
            predicted_depth_values = []
            for keypoint_id in range(depth_map.shape[1]):
                depth_channel = depth_map[:, keypoint_id : keypoint_id + 1]
                point = gt_coords[:, keypoint_id : keypoint_id + 1]
                sampled = pose_utils.point_sample(depth_channel, point).reshape(-1)
                predicted_depth_values.append(sampled)
            predicted_depth = F.softplus(torch.stack(predicted_depth_values, dim=1)) + 1e-4
            depth_mask = visible.bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
            if depth_mask.any():
                depth_loss = F.smooth_l1_loss(
                    torch.log(predicted_depth[depth_mask]),
                    torch.log(gt_depth[depth_mask]),
                )
            else:
                depth_loss = torch.zeros((), device=device)
            sampled_depth_values = []
            for keypoint_id in range(depth_map.shape[1]):
                depth_channel = depth_map[:, keypoint_id : keypoint_id + 1]
                point = fine_coords[:, keypoint_id : keypoint_id + 1]
                sampled = pose_utils.point_sample(depth_channel, point).reshape(-1)
                sampled_depth_values.append(sampled)
            sampled_depth = torch.stack(sampled_depth_values, dim=1)
            sampled_depth = F.softplus(sampled_depth).unsqueeze(-1) + 1e-4
            fx = k_mat[:, 0, 0].view(-1, 1)
            fy = k_mat[:, 1, 1].view(-1, 1)
            cx = k_mat[:, 0, 2].view(-1, 1)
            cy = k_mat[:, 1, 2].view(-1, 1)
            xyz = torch.cat(
                [
                    (kpts_pred[..., 0:1] - cx.unsqueeze(-1)) / fx.unsqueeze(-1) * sampled_depth,
                    (kpts_pred[..., 1:2] - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * sampled_depth,
                    sampled_depth,
                ],
                dim=-1,
            )
            edges = torch.nonzero(
                torch.triu(model.relation_decoder.adjacency > 0, diagonal=1),
                as_tuple=False,
            )
            cad_points = torch.as_tensor(cad_points_numpy, device=device, dtype=xyz.dtype).unsqueeze(0).expand(xyz.shape[0], -1, -1)
            cad_loss = cad_edge_loss(xyz, cad_points, edges) if edges.numel() else torch.zeros((), device=device)
            total = float(config.get("lambda_hm", 1.0)) * heatmap_loss.float() + pnp_weight * pnp_loss
            total = total + float(config.get("lambda_depth", 0.2)) * depth_loss
            total = total + float(config.get("lambda_stage_cons", 0.1)) * relation_loss
            total = total + float(config.get("lambda_cad", 0.1)) * cad_loss
        # The last accumulation group may contain fewer micro-batches.
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        scaler.scale(total / group_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += float(total.detach().item())
        if writer is not None and step % int(config.get("save_tensorboard", 100)) == 0:
            writer.add_scalar("Train/total", running / (step + 1), epoch * len(loader) + step)
            writer.add_scalar("Train/heatmap", float(heatmap_loss.detach().item()), epoch * len(loader) + step)
            writer.add_scalar("Train/depth", float(depth_loss.detach().item()), epoch * len(loader) + step)
            writer.add_scalar("Train/relation", float(relation_loss.detach().item()), epoch * len(loader) + step)
            writer.add_scalar("Train/cad", float(cad_loss.detach().item()), epoch * len(loader) + step)
    return running / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt-CARD training")
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    args.cfg = project_config_arg(args.cfg, PROJECT_ROOT)
    config = normalize_config_paths(pose_utils.load_config(args.cfg), PROJECT_ROOT)
    if config.get("legacy_output", False):
        raise ValueError("Training requires legacy_output=false; legacy list output is for inference adapters")
    if int(config.get("start_epoch", 0)) != 0:
        raise ValueError(
            "Exact epoch resume is not implemented: keep start_epoch=0. "
            "Use path_pretrain in a new source experiment, or --start-iteration/--restart-iteration for UDA."
        )
    if config.get("save_optimizer", False):
        raise ValueError("save_optimizer=true is not supported; checkpoints currently contain model weights only")
    seed_training(config)
    if not 0 <= int(config.get("start_epoch", 0)) < int(config["total_epochs"]):
        raise ValueError("require 0 <= start_epoch < total_epochs")
    device = torch.device(config.get("device", "cuda"))
    points, adjacency, distances = _load_graph(config)
    checkpoint_path = _resolve_loop_checkpoint(args.cfg, config)
    if checkpoint_path:
        # Full route checkpoints already contain the encoder weights. Avoid
        # re-downloading timm weights during later UDA rounds.
        config["pretrained"] = False
    model = get_convnext_card(
        config,
        adjacency=torch.from_numpy(adjacency),
        distances=torch.from_numpy(distances),
    ).to(device)
    _load_checkpoint(model, checkpoint_path, device)
    output_dir = config_output_dir(config, args.cfg, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    completion_marker = output_dir / "training_complete.json"
    if completion_marker.exists():
        raise FileExistsError(
            f"This run is already complete: {completion_marker}. Use a new experiment config/output path; "
            "the UDA scheduler removes the marker only for an explicit round restart."
        )
    writer = tensorboard.writer.SummaryWriter(output_dir / "logs")
    transforms_input, aug_intensity = build_input_transforms(config, training=True)
    image_augmenter = ImageOnlyAugmenter(config)
    _, aug_intensity_val = build_input_transforms(config, training=False)
    train_loader, val_loader = build_loaders(config, transforms_input)
    optimizer = _build_optimizer(config, model)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("amp", True)))
    geometry = (
        pose_utils.get_kmat_scaled(config, device),
        pose_utils.get_coefs(config, device),
        pose_utils.get_world_kpts(config, device),
    )
    best = float("inf")
    for epoch in range(int(config.get("start_epoch", 0)), int(config["total_epochs"])):
        freeze_epochs = int(config.get("freeze_backbone_epochs", 0))
        backbone_trainable = epoch >= freeze_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(backbone_trainable)
        loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            config,
            geometry,
            points,
            aug_intensity,
            device,
            writer,
            epoch,
            image_augmenter,
        )
        print(f"[Route B] epoch={epoch} train_loss={loss:.6f}")
        if not config.get("isloop", False):
            model.eval()
            with torch.no_grad():
                model.legacy_output = True
                metrics = normalize_val_metrics(
                    pose_utils.eval_loop(
                        val_loader,
                        aug_intensity_val,
                        model,
                        geometry[2],
                        geometry[0],
                        geometry[1],
                        device,
                        config,
                    )
                )
                model.legacy_output = False
            score = metrics["penalized_score"]
            writer.add_scalar("Validation/score", score, epoch)
            if score < best:
                best = score
                torch.save(model.state_dict(), ckpt_dir / f"best_epoch_{epoch}_{score:.6f}.pth")
            if epoch % int(config.get("save_epoch", 5)) == 0:
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pth")
        if config.get("isloop", False):
            torch.save(model.state_dict(), ckpt_dir / "init.pth")
    torch.save(model.state_dict(), ckpt_dir / "last.pth")
    mark_training_complete(
        output_dir, config, args.cfg,
        parent_checkpoint=checkpoint_path if config.get("isloop", False) else None,
        project_root=PROJECT_ROOT,
    )
    writer.close()


if __name__ == "__main__":
    main()
