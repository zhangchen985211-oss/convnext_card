import json
import os
import re
import shutil
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from BPnP import BPnP
from loaders import speedplus_segmentation_precomputed as speedplus
from src import speedscore
from src import utils as base_utils
from src.cad_graph import load_kpts_mat
from src.pose_estimation import estimate_pose
from src.pose_refinement import pixel_threshold_scale, refine_pose_lm
from src.image_augmentation import pixel_augmentation_enabled


load_config = base_utils.load_config
tensor_to_cvmat = base_utils.tensor_to_cvmat
tensor_to_mat = base_utils.tensor_to_mat
draw_keypoints = base_utils.draw_keypoints
heatmap_to_points = base_utils.heatmap_to_points
point_sample = base_utils.point_sample


def _device(device):
    return torch.device(device)


def get_world_kpts(config, device):
    world_kpts = load_kpts_mat(os.path.join(config["root_dir"], "kpts.mat"))
    return (
        torch.from_numpy(world_kpts)
        .unsqueeze(0)
        .repeat([config["batch_size"], 1, 1])
        .to(_device(device))
        .float()
    )


def get_coefs(config, device):
    cam = speedplus.Camera(config["root_dir"])
    return (
        torch.from_numpy(np.array(cam.dcoef, dtype=np.float32))
        .unsqueeze(0)
        .repeat([config["batch_size"], 1])
        .to(_device(device))
        .float()
    )


def get_kmat_scaled(config, device):
    cam = speedplus.Camera(config["root_dir"])
    k = cam.K.copy()
    k[0, :] *= config["cols"] / 1920
    k[1, :] *= config["rows"] / 1200
    return (
        torch.from_numpy(k.astype(np.float32))
        .unsqueeze(0)
        .repeat([config["batch_size"], 1, 1])
        .to(_device(device))
        .float()
    )


def camera_for_batch(k_mat_input, dist_coefs, batch_size):
    """把相机参数裁剪/复制到当前 batch，保证 undistort 的 batch 维一致。"""
    if k_mat_input.shape[0] == batch_size:
        k_batch = k_mat_input
    elif k_mat_input.shape[0] > batch_size:
        k_batch = k_mat_input[:batch_size]
    else:
        k_batch = k_mat_input[:1].repeat(batch_size, 1, 1)

    if dist_coefs.shape[0] == batch_size:
        d_batch = dist_coefs
    elif dist_coefs.shape[0] > batch_size:
        d_batch = dist_coefs[:batch_size]
    else:
        d_batch = dist_coefs[:1].repeat(batch_size, 1)
    return k_batch, d_batch


def undistort_for_model(image, k_mat_input, dist_coefs):
    """Apply the same lens-undistortion transform used by training and inference."""
    k_batch, d_batch = camera_for_batch(k_mat_input, dist_coefs, image.shape[0])
    return kornia.geometry.undistort_image(image, k_batch, d_batch)


def preprocess_for_model(image, normalize, k_mat_input, dist_coefs, config):
    """Deterministic inference path matching the selected training input order."""
    if pixel_augmentation_enabled(config):
        return normalize(undistort_for_model(image, k_mat_input, dist_coefs))
    # Preserve the exact order used by existing checkpoints when disabled.
    return undistort_for_model(normalize(image), k_mat_input, dist_coefs)


def get_pnp_dist_coeffs_after_undistort(cam_or_dist=None):
    """Undistorted images must use zero distortion coefficients during PnP."""
    return None


def _writer_level_ids(input_dict):
    return sorted(
        key[1]
        for key in input_dict
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "heatmap"
    )


def update_writer(writer, input_dict, iteration, config):
    level_ids = _writer_level_ids(input_dict)
    if not level_ids:
        return

    first_level = level_ids[0]
    final_level = level_ids[-1]

    with torch.no_grad():
        writer.add_scalar(
            "Total Train Loss ",
            sum(input_dict["total_loss", level_id] for level_id in level_ids),
            iteration,
        )
        writer.add_scalar(
            "Heatmap Loss ",
            sum(input_dict["loss_hm", level_id] for level_id in level_ids),
            iteration,
        )
        if ("loss_pnp", first_level) in input_dict:
            writer.add_scalar(
                "PnP Loss ",
                sum(input_dict["loss_pnp", level_id] for level_id in level_ids),
                iteration,
            )
        if ("loss_3d", first_level) in input_dict:
            writer.add_scalar(
                "3D  Loss ",
                sum(input_dict["loss_3d", level_id] for level_id in level_ids),
                iteration,
            )
            writer.add_scalar(
                "Rotation  Loss ",
                sum(input_dict["rot_err", level_id] for level_id in level_ids),
                iteration,
            )
            writer.add_scalar(
                "Translation  Loss ",
                sum(input_dict["tra_err", level_id] for level_id in level_ids),
                iteration,
            )

        image = tensor_to_cvmat(input_dict["img_source"].sum(1, True)[0])
        image = draw_keypoints(
            image,
            tensor_to_mat(input_dict["kpts_gt"][0]),
            color=(0, 255, 0),
        )
        image = cv2.resize(image, (200, 200))

        hm_gt = cv2.resize(
            tensor_to_cvmat(input_dict["heatmap_gt"].sum(1, True)[0]),
            (200, 200),
        )

        hm_c_0 = tensor_to_cvmat(input_dict["heatmap", first_level].sum(1, True)[0])
        if ("kpts_pnp", first_level) in input_dict:
            hm_c_0 = draw_keypoints(
                hm_c_0,
                tensor_to_mat(input_dict["kpts_pnp", first_level][0]),
                color=(0, 255, 255),
            )
        hm_c_0 = cv2.resize(hm_c_0, (200, 200))

        hm_c_1 = tensor_to_cvmat(input_dict["heatmap", final_level].sum(1, True)[0])
        if ("kpts_pnp", final_level) in input_dict:
            hm_c_1 = draw_keypoints(
                hm_c_1,
                tensor_to_mat(input_dict["kpts_pnp", final_level][0]),
                color=(0, 255, 255),
            )
        hm_c_1 = cv2.resize(hm_c_1, (200, 200))

        kpts_3d_1 = None
        kpts_3d_2 = None
        if ("poses_3d", first_level) in input_dict:
            kpts_3d_1 = BPnP.batch_project(
                input_dict["poses_3d", first_level],
                input_dict["kpts_world"],
                input_dict["k_mat_input"],
            )
        if ("poses_3d", final_level) in input_dict:
            kpts_3d_2 = BPnP.batch_project(
                input_dict["poses_3d", final_level],
                input_dict["kpts_world"],
                input_dict["k_mat_input"],
            )

        depth_0 = tensor_to_cvmat(input_dict["depth", first_level].sum(1, True)[0])
        if kpts_3d_1 is not None:
            depth_0 = draw_keypoints(
                depth_0,
                tensor_to_mat(kpts_3d_1[0]),
                color=(0, 255, 255),
            )
        depth_0 = cv2.resize(depth_0, (200, 200))

        depth_1 = tensor_to_cvmat(input_dict["depth", final_level].sum(1, True)[0])
        if kpts_3d_2 is not None:
            depth_1 = draw_keypoints(
                depth_1,
                tensor_to_mat(kpts_3d_2[0]),
                color=(0, 255, 255),
            )
        depth_1 = cv2.resize(depth_1, (200, 200))

        a = np.vstack((image, hm_gt))
        b = np.vstack((hm_c_0, hm_c_1))
        c = np.vstack((depth_0, depth_1))
        d = np.hstack((a, b, c))
        writer.add_image(
            "GT -- Heatmap + PnP -- Depth + 3D",
            d,
            iteration,
            dataformats="HWC",
        )


def extract_niter(path_text):
    match = re.search(r"niter_(\d+)", str(path_text))
    return int(match.group(1)) if match else 0


def dynamic_reprojection_error(path_text):
    return 2.0 if extract_niter(path_text) <= 40 else 1.0


def _responses_from_mask(mask_tensor):
    mask_np = tensor_to_mat(mask_tensor)
    if mask_np.ndim == 4:
        mask_np = mask_np[0]
    return np.max(np.max(mask_np, axis=1), axis=1)


PSEUDO_STRATEGY_STAGED_QUALITY = "staged_quality_fallback"

PSEUDO_QUALITY_DEFAULTS = {
    "pseudo_min_depth": 0.1,
    "pseudo_max_depth": 20.0,
    "pseudo_max_selected_mean_reprojection": 1.5,
    "pseudo_max_selected_reprojection": 3.0,
    "pseudo_max_full_mean_reprojection": 8.0,
    "pseudo_max_full_reprojection": 20.0,
    "pseudo_strict_min_inliers": 5,
    "pseudo_single_min_inliers": 6,
    "pseudo_expanded_min_inliers": 7,
    "pseudo_fused_min_inliers": 7,
    "pseudo_allow_direct_solvepnp": False,
}

PSEUDO_QUALITY_COUNTER_STRATEGIES = (
    "strict_both_success",
    "expanded_topk_ransac",
    "single_branch_fallback",
    "fused_branch_ransac",
)


def pose_result_to_quaternion(result, device):
    rvec = torch.from_numpy(np.squeeze(result["rvecs"])).unsqueeze(0).to(device)
    quat = kornia.geometry.angle_axis_to_quaternion(
        rvec,
        kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ,
    ).unsqueeze(0)
    return tensor_to_mat(quat[0])


def solve_measurement_pose(
    pred_kpts,
    responses,
    world_kpts,
    k_mat_input,
    cam,
    topk,
    reprojection_error,
    selected_keypoints=None,
    config=None,
):
    pred_kpts = np.asarray(pred_kpts, dtype=np.float32)
    responses = np.asarray(responses, dtype=np.float32)
    world_kpts = np.asarray(world_kpts, dtype=np.float32)
    k_mat_input = np.asarray(k_mat_input, dtype=np.float32)

    if selected_keypoints is None:
        index_max = np.argsort(-responses)
        selected_index = index_max[:topk]
    else:
        selected_index = np.asarray(selected_keypoints, dtype=np.int64)
        topk = len(selected_index)

    result = {
        "accepted": False,
        "rvecs": None,
        "tvecs": None,
        "inliers": None,
        "response_sum": 0.0,
        "selected_keypoints": selected_index.astype(int).tolist(),
        "topk": int(len(selected_index)),
        "pred_kpts": pred_kpts,
        "responses": responses,
        "reprojection_error": float(reprojection_error),
    }
    valid = (
        world_kpts.ndim == 2 and world_kpts.shape[1] == 3
        and pred_kpts.shape == (len(world_kpts), 2)
        and responses.shape == (len(world_kpts),)
        and k_mat_input.shape == (3, 3)
        and selected_index.ndim == 1 and len(selected_index) >= 4
        and len(np.unique(selected_index)) == len(selected_index)
        and np.all(selected_index >= 0) and np.all(selected_index < len(world_kpts))
        and np.isfinite(world_kpts).all() and np.isfinite(pred_kpts).all()
        and np.isfinite(responses).all() and np.isfinite(k_mat_input).all()
        and np.isfinite(reprojection_error) and reprojection_error > 0
    )
    if not valid:
        result["solver_reject_reason"] = "invalid_correspondences"
        return result
    result["response_sum"] = float(np.sum(responses[selected_index]))
    try:
        accepted, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(world_kpts[selected_index]),
            np.ascontiguousarray(pred_kpts[selected_index]),
            k_mat_input,
            get_pnp_dist_coeffs_after_undistort(cam),
            confidence=0.99,
            reprojectionError=float(reprojection_error),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        result["solver_reject_reason"] = "opencv_error"
        return result
    result.update(accepted=bool(accepted), rvecs=rvecs, tvecs=tvecs, inliers=inliers)
    return refine_pose_lm(
        result, world_kpts, k_mat_input, get_pnp_dist_coeffs_after_undistort(cam), config
    )


def solve_branch_pose(
    output_entry,
    image_shape,
    world_kpts,
    k_mat_input,
    cam,
    topk,
    reprojection_error,
    config=None,
):
    pred_mask = output_entry["hm_c"]
    pred_mask = F.interpolate(
        pred_mask,
        size=image_shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    pred_kpts = tensor_to_mat(heatmap_to_points(pred_mask)[0])
    responses = _responses_from_mask(output_entry["hm_c"])
    return solve_measurement_pose(
        pred_kpts,
        responses,
        world_kpts,
        k_mat_input,
        cam,
        topk,
        reprojection_error,
        config=config,
    )


def _project_points(world_kpts, rvecs, tvecs, k_mat_input, cam):
    projected, _ = cv2.projectPoints(
        np.asarray(world_kpts, dtype=np.float32),
        np.asarray(rvecs, dtype=np.float64),
        np.asarray(tvecs, dtype=np.float64),
        np.asarray(k_mat_input, dtype=np.float64),
        get_pnp_dist_coeffs_after_undistort(cam),
    )
    return np.squeeze(projected, axis=1)


def _pose_depths(world_kpts, rvecs, tvecs):
    rot_mat, _ = cv2.Rodrigues(np.asarray(rvecs, dtype=np.float64))
    points_camera = np.asarray(world_kpts, dtype=np.float64) @ rot_mat.T
    points_camera += np.asarray(tvecs, dtype=np.float64).reshape(1, 3)
    return points_camera[:, 2]


def _inlier_count(inliers):
    if inliers is None:
        return 0
    return int(np.asarray(inliers).reshape(-1).shape[0])


def _quality_config(config):
    quality = dict(PSEUDO_QUALITY_DEFAULTS)
    for key in quality:
        if key in config:
            quality[key] = config[key]
    scale = pixel_threshold_scale(config)
    for key in (
        "pseudo_max_selected_mean_reprojection", "pseudo_max_selected_reprojection",
        "pseudo_max_full_mean_reprojection", "pseudo_max_full_reprojection",
    ):
        quality[key] = float(quality[key]) * scale
    return quality


def _finite_pose(result):
    rvecs = result.get("rvecs")
    tvecs = result.get("tvecs")
    if rvecs is None or tvecs is None:
        return False
    return bool(np.all(np.isfinite(rvecs)) and np.all(np.isfinite(tvecs)))


def _candidate_metrics(result, world_kpts, k_mat_input, cam):
    metrics = {
        "inliers": _inlier_count(result.get("inliers")),
        "selected_mean_reprojection": None,
        "selected_max_reprojection": None,
        "full_mean_reprojection": None,
        "full_max_reprojection": None,
        "min_depth": None,
        "max_depth": None,
        "finite_pose": _finite_pose(result),
    }
    pred_kpts = result.get("pred_kpts")
    if not result.get("accepted") or pred_kpts is None or not metrics["finite_pose"]:
        return metrics

    projected = _project_points(
        world_kpts,
        result["rvecs"],
        result["tvecs"],
        k_mat_input,
        cam,
    )
    errors = np.linalg.norm(projected - np.asarray(pred_kpts), axis=1)
    selected_keypoints = np.asarray(result.get("selected_keypoints", []), dtype=np.int64)
    selected_errors = errors[selected_keypoints] if selected_keypoints.size else errors
    depths = _pose_depths(world_kpts, result["rvecs"], result["tvecs"])

    metrics.update(
        {
            "selected_mean_reprojection": float(np.mean(selected_errors)),
            "selected_max_reprojection": float(np.max(selected_errors)),
            "full_mean_reprojection": float(np.mean(errors)),
            "full_max_reprojection": float(np.max(errors)),
            "min_depth": float(np.min(depths)),
            "max_depth": float(np.max(depths)),
        }
    )
    return metrics


def _metrics_are_finite(metrics, keys):
    return all(
        metrics[key] is not None and np.isfinite(metrics[key])
        for key in keys
    )


def _passes_quality_gate(result, world_kpts, k_mat_input, cam, config, min_inliers):
    quality = _quality_config(config)
    metrics = _candidate_metrics(result, world_kpts, k_mat_input, cam)
    reject_reason = None
    finite_metric_keys = (
        "selected_mean_reprojection",
        "selected_max_reprojection",
        "full_mean_reprojection",
        "full_max_reprojection",
        "min_depth",
        "max_depth",
    )

    if not result.get("accepted"):
        reject_reason = "pnp_failed"
    elif not metrics["finite_pose"]:
        reject_reason = "non_finite_pose"
    elif not _metrics_are_finite(metrics, finite_metric_keys):
        reject_reason = "non_finite_metrics"
    elif metrics["inliers"] < int(min_inliers):
        reject_reason = "too_few_inliers"
    elif (
        metrics["min_depth"] is None
        or metrics["min_depth"] < float(quality["pseudo_min_depth"])
        or metrics["max_depth"] > float(quality["pseudo_max_depth"])
    ):
        reject_reason = "depth_out_of_range"
    elif metrics["selected_mean_reprojection"] > float(
        quality["pseudo_max_selected_mean_reprojection"]
    ):
        reject_reason = "selected_mean_reprojection_too_high"
    elif metrics["selected_max_reprojection"] > float(
        quality["pseudo_max_selected_reprojection"]
    ):
        reject_reason = "selected_max_reprojection_too_high"
    elif metrics["full_mean_reprojection"] > float(
        quality["pseudo_max_full_mean_reprojection"]
    ):
        reject_reason = "full_mean_reprojection_too_high"
    elif metrics["full_max_reprojection"] > float(
        quality["pseudo_max_full_reprojection"]
    ):
        reject_reason = "full_max_reprojection_too_high"

    metrics["quality_passed"] = reject_reason is None
    metrics["reject_reason"] = reject_reason
    return metrics


def _diagnostic_from_result(
    filename,
    niter,
    strategy,
    result,
    metrics,
    reprojection_error,
    accepted=False,
):
    return {
        "filename": filename,
        "niter": niter,
        "strategy": strategy,
        "accepted": bool(accepted),
        "reject_reason": metrics.get("reject_reason"),
        "quality_passed": metrics.get("quality_passed"),
        "topk": result.get("topk"),
        "branch_id": result.get("selected_index"),
        "inliers": metrics.get("inliers"),
        "selected_keypoints": result.get("selected_keypoints"),
        "response_sum": result.get("response_sum"),
        "selected_mean_reprojection": metrics.get("selected_mean_reprojection"),
        "selected_max_reprojection": metrics.get("selected_max_reprojection"),
        "full_mean_reprojection": metrics.get("full_mean_reprojection"),
        "full_max_reprojection": metrics.get("full_max_reprojection"),
        "min_depth": metrics.get("min_depth"),
        "max_depth": metrics.get("max_depth"),
        "finite_pose": metrics.get("finite_pose"),
        "dynamic_reprojection_error": reprojection_error,
        "refinement": result.get("refinement"),
    }


def _write_diagnostic(fp, diagnostic):
    json.dump(diagnostic, fp, ensure_ascii=False)
    fp.write("\n")


def _is_selected_candidate(candidate, selected):
    return (
        candidate.get("strategy") == selected.get("strategy")
        and candidate.get("selected_index") == selected.get("selected_index")
        and candidate.get("topk") == selected.get("topk")
    )


def _choose_quality_candidate(candidates, world_kpts, k_mat_input, cam, config, min_inliers):
    evaluated = []
    for candidate in candidates:
        metrics = _passes_quality_gate(
            candidate,
            world_kpts,
            k_mat_input,
            cam,
            config,
            min_inliers,
        )
        evaluated.append((candidate, metrics))

    passed = [(candidate, metrics) for candidate, metrics in evaluated if metrics["quality_passed"]]
    if not passed:
        return None, evaluated
    passed.sort(key=lambda item: (
        item[1]["full_mean_reprojection"],
        item[1]["selected_mean_reprojection"],
        -item[1]["inliers"],
        -item[0].get("response_sum", 0.0),
    ))
    return passed[0], evaluated


def _increment_quality_counter(stats, strategy, metrics):
    candidate_key = f"{strategy}_candidates"
    passed_key = f"{strategy}_quality_passed"
    if candidate_key in stats:
        stats[candidate_key] += 1
    if passed_key in stats and metrics.get("quality_passed"):
        stats[passed_key] += 1


def _update_quality_candidate_stats(stats, selected):
    selected_counted = False
    for strategy, candidate, metrics in selected.get("diagnostics", []):
        _increment_quality_counter(stats, strategy, metrics)
        if selected.get("accepted") and _is_selected_candidate(candidate, selected):
            selected_counted = True

    if selected.get("accepted") and not selected_counted:
        _increment_quality_counter(
            stats,
            selected.get("strategy"),
            selected.get("quality_metrics", {}),
        )


def _fused_branch_result(branch_results, world_kpts, k_mat_input, cam, reprojection_error, config=None):
    pred_kpts = np.mean([item["pred_kpts"] for item in branch_results], axis=0)
    responses = np.mean([item["responses"] for item in branch_results], axis=0)
    selected_keypoints = np.argsort(-responses)[:9]
    result = solve_measurement_pose(
        pred_kpts,
        responses,
        world_kpts,
        k_mat_input,
        cam,
        topk=9,
        reprojection_error=reprojection_error,
        selected_keypoints=selected_keypoints,
        config=config,
    )
    result["selected_index"] = "fused"
    result["strategy"] = "fused_branch_ransac"
    return result


def _stats_for_staged_pseudo(niter, reprojection_error, enabled_strategies):
    stats = {
        "pseudo_strategy": PSEUDO_STRATEGY_STAGED_QUALITY,
        "niter": niter,
        "enabled_strategies": enabled_strategies,
        "strict_both_success": 0,
        "expanded_topk_ransac": 0,
        "single_branch_fallback": 0,
        "fused_branch_ransac": 0,
        "rejected_samples": 0,
        "written_samples": 0,
        "strict_quality_rejected": 0,
        "only_branch0_success": 0,
        "only_branch1_success": 0,
        "both_failed": 0,
        "reprojection_error": reprojection_error,
    }
    for strategy in PSEUDO_QUALITY_COUNTER_STRATEGIES:
        stats[f"{strategy}_candidates"] = 0
        stats[f"{strategy}_quality_passed"] = 0
    return stats


def _enabled_quality_strategies(niter, config=None):
    config = config or {}
    strategies = ["strict_both_success"]
    for key, default, name in (
        ("pseudo_expanded_start_round", 9, "expanded_topk_ransac"),
        ("pseudo_single_start_round", 12, "single_branch_fallback"),
        ("pseudo_fused_start_round", 14, "fused_branch_ransac"),
    ):
        start = config.get(key, default)
        if start is not None and niter >= int(start):
            strategies.append(name)
    return strategies


def select_pose_staged_quality(
    outputs,
    image_tensor,
    world_kpts,
    k_mat_input,
    cam,
    config,
    niter,
    reprojection_error,
):
    branch_results = [
        solve_branch_pose(
            output,
            image_tensor.shape,
            world_kpts,
            k_mat_input,
            cam,
            topk=7,
            reprojection_error=reprojection_error,
            config=config,
        )
        for output in outputs
    ]
    accepted_flags = [item["accepted"] for item in branch_results]
    response_sums = [item["response_sum"] for item in branch_results]
    diagnostics = []
    quality = _quality_config(config)

    strict_candidates = []
    for branch_id, result in enumerate(branch_results):
        candidate = dict(result)
        candidate.update(selected_index=branch_id, response_sums=response_sums,
                         strategy="strict_both_success")
        strict_candidates.append(candidate)
    selected_pair, strict_evaluated = _choose_quality_candidate(
        strict_candidates, world_kpts, k_mat_input, cam, config,
        quality["pseudo_strict_min_inliers"],
    )
    diagnostics.extend((candidate["strategy"], candidate, metrics)
                       for candidate, metrics in strict_evaluated)
    # Strict means every branch passes the quality gate. A solver success flag
    # alone is not quality evidence; a lone good branch uses the later tier.
    if len(branch_results) >= 2 and all(metrics["quality_passed"] for _, metrics in strict_evaluated):
        selected, metrics = selected_pair
        selected.update(accepted=True, quality_metrics=metrics, diagnostics=diagnostics)
        return selected
    if all(accepted_flags):
        strict_failure = "strict_quality_rejected"
    elif accepted_flags == [True, False]:
        strict_failure = "only_branch0_success"
    elif accepted_flags == [False, True]:
        strict_failure = "only_branch1_success"
    else:
        strict_failure = "both_failed"

    expanded_results = []
    enabled_strategies = _enabled_quality_strategies(niter, config)
    if "expanded_topk_ransac" in enabled_strategies:
        expanded_results = [
            solve_branch_pose(
                output,
                image_tensor.shape,
                world_kpts,
                k_mat_input,
                cam,
                topk=9,
                reprojection_error=reprojection_error,
                config=config,
            )
            for output in outputs
        ]
        expanded_response_sums = [item["response_sum"] for item in expanded_results]
        expanded_candidates = []
        for branch_id, result in enumerate(expanded_results):
            candidate = dict(result)
            candidate.update(
                {
                    "selected_index": branch_id,
                    "response_sums": expanded_response_sums,
                    "strategy": "expanded_topk_ransac",
                }
            )
            expanded_candidates.append(candidate)

        selected_pair, expanded_evaluated = _choose_quality_candidate(
            expanded_candidates,
            world_kpts,
            k_mat_input,
            cam,
            config,
            quality["pseudo_expanded_min_inliers"],
        )
        diagnostics.extend(
            (candidate["strategy"], candidate, metrics)
            for candidate, metrics in expanded_evaluated
        )
        if selected_pair is not None:
            selected, metrics = selected_pair
            selected["accepted"] = True
            selected["quality_metrics"] = metrics
            selected["diagnostics"] = diagnostics
            selected["strict_failure"] = strict_failure
            return selected

    if "single_branch_fallback" in enabled_strategies:
        candidates = []
        for branch_id, result in enumerate(branch_results):
            candidate = dict(result)
            candidate.update(
                {
                    "selected_index": branch_id,
                    "response_sums": response_sums,
                    "strategy": "single_branch_fallback",
                }
            )
            candidates.append(candidate)
        selected_pair, evaluated = _choose_quality_candidate(
            candidates,
            world_kpts,
            k_mat_input,
            cam,
            config,
            quality["pseudo_single_min_inliers"],
        )
        diagnostics.extend(
            (candidate["strategy"], candidate, metrics) for candidate, metrics in evaluated
        )
        if selected_pair is not None:
            selected, metrics = selected_pair
            selected["accepted"] = True
            selected["quality_metrics"] = metrics
            selected["diagnostics"] = diagnostics
            selected["strict_failure"] = strict_failure
            return selected

    if "fused_branch_ransac" in enabled_strategies:
        fuse_inputs = expanded_results if expanded_results else branch_results
        fused_result = _fused_branch_result(
            fuse_inputs,
            world_kpts,
            k_mat_input,
            cam,
            reprojection_error,
            config=config,
        )
        metrics = _passes_quality_gate(
            fused_result,
            world_kpts,
            k_mat_input,
            cam,
            config,
            quality["pseudo_fused_min_inliers"],
        )
        diagnostics.append((fused_result["strategy"], fused_result, metrics))
        if metrics["quality_passed"]:
            fused_result["accepted"] = True
            fused_result["quality_metrics"] = metrics
            fused_result["diagnostics"] = diagnostics
            fused_result["strict_failure"] = strict_failure
            return fused_result

    return {
        "accepted": False,
        "selected_index": None,
        "response_sums": response_sums,
        "strategy": "quality_rejected",
        "branch_results": branch_results,
        "strict_failure": strict_failure,
        "diagnostics": diagnostics,
    }


def select_pose_from_outputs(
    outputs,
    image_tensor,
    world_kpts,
    k_mat_input,
    cam,
    topk=7,
    reprojection_error=1.0,
    strict_consensus=True,
):
    """双分支 PnP 共识：默认两个分支都成功才接受，完全贴近 hourglass 逻辑。"""
    branch_results = [
        solve_branch_pose(
            output,
            image_tensor.shape,
            world_kpts,
            k_mat_input,
            cam,
            topk,
            reprojection_error,
        )
        for output in outputs
    ]
    accepted_flags = [item["accepted"] for item in branch_results]
    if all(accepted_flags):
        response_sums = [item["response_sum"] for item in branch_results]
        selected_index = int(np.argmax(response_sums))
        selected = dict(branch_results[selected_index])
        selected.update(
            {
                "accepted": True,
                "selected_index": selected_index,
                "response_sums": response_sums,
                "strategy": "both_success_pick_high_response",
            }
        )
        return selected

    if not strict_consensus and any(accepted_flags):
        selected_index = accepted_flags.index(True)
        selected = dict(branch_results[selected_index])
        selected.update(
            {
                "accepted": True,
                "selected_index": selected_index,
                "response_sums": [item["response_sum"] for item in branch_results],
                "strategy": "single_success_fallback",
            }
        )
        return selected

    return {
        "accepted": False,
        "selected_index": None,
        "response_sums": [item["response_sum"] for item in branch_results],
        "strategy": "pnp_failed_or_strict_rejected",
        "branch_results": branch_results,
    }


def eval_loop(val_loader, aug_intensity_val, model, kpts_world, k_mat_input, dist_coefs, device, config):
    speed_total = speed_t_total = speed_r_total = 0.0
    valid_count = 0
    failed_count = 0
    total_count = 0
    failed_pose_penalty = float(config.get("eval_failed_pose_penalty", 1e6))
    kpts_world_np = tensor_to_mat(kpts_world[0])
    k_mat_np = tensor_to_mat(k_mat_input[0])

    for data in tqdm(val_loader, ncols=50):
        img = data["image"].to(device)
        img = preprocess_for_model(img, aug_intensity_val, k_mat_input, dist_coefs, config)
        out = model(img)

        for index in range(img.shape[0]):
            total_count += 1
            selected, _ = estimate_pose(
                [{"hm_c": branch["hm_c"][index : index + 1]} for branch in out],
                img[index : index + 1],
                kpts_world_np,
                k_mat_np,
                config,
            )
            if selected is None:
                failed_count += 1
                continue

            q_gt = tensor_to_mat(data["q0"][index])
            t_gt = tensor_to_mat(data["r0"][index])
            q_est = pose_result_to_quaternion(selected, device)
            speed, speed_t, speed_r = speedscore.speed_score(
                selected["tvecs"], q_est, t_gt, q_gt, applyThresh=False
            )
            speed_total += speed
            speed_t_total += speed_t
            speed_r_total += speed_r
            valid_count += 1

    if failed_count:
        print(f"[Pose Eval] PnP failed samples: {failed_count}")
    if valid_count == 0:
        print("[Pose Eval] All samples failed PnP; returning a large penalty score.")
        return {
            "score": 1e16,
            "score_t": 1e16,
            "score_r": 1e16,
            "valid_score": 1e16,
            "valid_score_t": 1e16,
            "valid_score_r": 1e16,
            "penalized_score": 1e16,
            "success_rate": 0.0,
            "valid_count": 0,
            "failed_count": failed_count,
            "total_count": total_count,
        }
    denom = valid_count
    valid_score = speed_total / denom
    valid_score_t = speed_t_total / denom
    valid_score_r = speed_r_total / denom
    penalized_score = (
        speed_total + failed_count * failed_pose_penalty
    ) / max(total_count, 1)
    success_rate = valid_count / max(total_count, 1)
    return {
        "score": penalized_score,
        "score_t": valid_score_t,
        "score_r": valid_score_r,
        "valid_score": valid_score,
        "valid_score_t": valid_score_t,
        "valid_score_r": valid_score_r,
        "penalized_score": penalized_score,
        "success_rate": success_rate,
        "valid_count": valid_count,
        "failed_count": failed_count,
        "total_count": total_count,
    }


def generate_json_loop(
    test_loader,
    aug_intensity_test,
    model,
    kpts_world,
    k_mat_input,
    dist_coefs,
    device,
    config,
    path_json,
):
    cam = speedplus.Camera(config["root_dir"])
    kpts_world_np = tensor_to_mat(kpts_world[0])
    k_mat_np = tensor_to_mat(k_mat_input[0])
    reprojection_error = float(config.get(
        "pseudo_reprojection_error", dynamic_reprojection_error(path_json)
    )) * pixel_threshold_scale(config)
    niter = extract_niter(path_json)
    use_staged_quality = (
        config.get("pseudo_strategy") == PSEUDO_STRATEGY_STAGED_QUALITY
    )
    enabled_strategies = _enabled_quality_strategies(niter, config)
    stats = {
        "both_success": 0,
        "only_branch0_success": 0,
        "only_branch1_success": 0,
        "both_failed": 0,
        "written_samples": 0,
        "reprojection_error": reprojection_error,
    }
    if use_staged_quality:
        stats = _stats_for_staged_pseudo(niter, reprojection_error, enabled_strategies)
    print(
        f"\n[Pose Pseudo] NITER={niter}, "
        f"reprojectionError={reprojection_error}\n"
    )
    if use_staged_quality:
        print("[Pose Pseudo] strategy:", PSEUDO_STRATEGY_STAGED_QUALITY)
        print("[Pose Pseudo] enabled strategies:", ", ".join(enabled_strategies))

    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = str(Path(path_json).with_name("pseudo_label_diagnostics.jsonl"))
    diagnostics_fp = None
    if use_staged_quality:
        diagnostics_fp = open(diagnostics_path, "w", encoding="utf-8")

    try:
        with open(path_json, "w", encoding="utf-8") as fp:
            for data in tqdm(test_loader, ncols=50):
                img = data["image"].to(device)
                img = preprocess_for_model(img, aug_intensity_test, k_mat_input, dist_coefs, config)
                outputs = model(img)
                for batch_index in range(img.shape[0]):
                    sample_outputs = [
                        {
                            "hm_c": output["hm_c"][batch_index : batch_index + 1],
                            "depth": output["depth"][batch_index : batch_index + 1],
                        }
                        for output in outputs
                    ]
                    filename = data["y"][batch_index]
                    if use_staged_quality:
                        selected = select_pose_staged_quality(
                            sample_outputs,
                            img[batch_index : batch_index + 1],
                            kpts_world_np,
                            k_mat_np,
                            cam,
                            config,
                            niter,
                            reprojection_error,
                        )
                        strict_failure = selected.get("strict_failure")
                        if strict_failure in stats:
                            stats[strict_failure] += 1
                        _update_quality_candidate_stats(stats, selected)

                        if selected["accepted"]:
                            strategy = selected["strategy"]
                            stats[strategy] += 1
                            metrics = selected.get("quality_metrics", {})
                            _write_diagnostic(
                                diagnostics_fp,
                                _diagnostic_from_result(
                                    filename,
                                    niter,
                                    strategy,
                                    selected,
                                    metrics,
                                    reprojection_error,
                                    accepted=True,
                                ),
                            )
                            for diag_strategy, candidate, diag_metrics in selected.get(
                                "diagnostics", []
                            ):
                                if _is_selected_candidate(candidate, selected):
                                    continue
                                _write_diagnostic(
                                    diagnostics_fp,
                                    _diagnostic_from_result(
                                        filename,
                                        niter,
                                        diag_strategy,
                                        candidate,
                                        diag_metrics,
                                        reprojection_error,
                                        accepted=False,
                                    ),
                                )
                        else:
                            stats["rejected_samples"] += 1
                            diagnostics = selected.get("diagnostics", [])
                            if diagnostics:
                                for strategy, candidate, metrics in diagnostics:
                                    _write_diagnostic(
                                        diagnostics_fp,
                                        _diagnostic_from_result(
                                            filename,
                                            niter,
                                            strategy,
                                            candidate,
                                            metrics,
                                            reprojection_error,
                                            accepted=False,
                                        ),
                                    )
                            else:
                                _write_diagnostic(
                                    diagnostics_fp,
                                    {
                                        "filename": filename,
                                        "niter": niter,
                                        "strategy": "strict_both_success",
                                        "accepted": False,
                                        "reject_reason": "strict_failed_fallback_locked",
                                        "quality_passed": None,
                                        "topk": 7,
                                        "branch_id": None,
                                        "inliers": None,
                                        "selected_keypoints": None,
                                        "response_sum": None,
                                        "selected_mean_reprojection": None,
                                        "selected_max_reprojection": None,
                                        "full_mean_reprojection": None,
                                        "full_max_reprojection": None,
                                        "min_depth": None,
                                        "max_depth": None,
                                        "finite_pose": None,
                                        "dynamic_reprojection_error": reprojection_error,
                                    },
                                )
                    else:
                        selected = select_pose_from_outputs(
                            sample_outputs,
                            img[batch_index : batch_index + 1],
                            kpts_world_np,
                            k_mat_np,
                            cam,
                            topk=7,
                            reprojection_error=reprojection_error,
                            strict_consensus=True,
                        )
                        flags = [
                            item["accepted"]
                            for item in selected.get("branch_results", [])
                        ]
                        if selected["accepted"]:
                            stats["both_success"] += 1
                        elif flags == [True, False]:
                            stats["only_branch0_success"] += 1
                        elif flags == [False, True]:
                            stats["only_branch1_success"] += 1
                        else:
                            stats["both_failed"] += 1

                    if not selected["accepted"]:
                        continue

                    q_est = pose_result_to_quaternion(selected, device)
                    sample = {
                        "q_vbs2tango_true": q_est.tolist(),
                        "r_Vo2To_vbs_true": np.squeeze(selected["tvecs"].T).tolist(),
                        "filename": filename,
                    }
                    json.dump(sample, fp)
                    fp.write("\n")
                    stats["written_samples"] += 1
    finally:
        if diagnostics_fp is not None:
            diagnostics_fp.close()

    stats_path = str(Path(path_json).with_name("pseudo_label_stats.json"))
    if use_staged_quality:
        stats["total_samples"] = stats["written_samples"] + stats["rejected_samples"]
        stats["acceptance_rate"] = stats["written_samples"] / max(stats["total_samples"], 1)
    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print("[Pose Pseudo] stats:", stats)
    print("[Pose Pseudo] stats saved to:", stats_path)
    if use_staged_quality:
        print("[Pose Pseudo] diagnostics saved to:", diagnostics_path)
    return stats


def test_loop(val_loader, aug_intensity_val, model, kpts_world, k_mat_input, dist_coefs, device, config):
    cam = speedplus.Camera(config["root_dir"])
    speed_total = speed_t_total = speed_r_total = 0.0
    valid_count = 0
    failed_count = 0
    kpts_world_np = tensor_to_mat(kpts_world[0])
    k_mat_np = tensor_to_mat(k_mat_input[0])

    for data in tqdm(val_loader, ncols=50):
        img = data["image"].to(device)
        q_gt = tensor_to_mat(data["q0"][0])
        t_gt = tensor_to_mat(data["r0"][0])
        img = preprocess_for_model(img, aug_intensity_val, k_mat_input, dist_coefs, config)
        selected = select_pose_from_outputs(
            model(img),
            img,
            kpts_world_np,
            k_mat_np,
            cam,
            topk=7,
            reprojection_error=1.0,
            strict_consensus=True,
        )
        if not selected["accepted"]:
            failed_count += 1
            continue
        q_est = pose_result_to_quaternion(selected, device)
        speed, speed_t, speed_r = speedscore.speed_score(
            selected["tvecs"], q_est, t_gt, q_gt, applyThresh=False
        )
        speed_total += speed
        speed_t_total += speed_t
        speed_r_total += speed_r
        valid_count += 1
    if failed_count:
        print(f"[Pose Test] PnP failed samples: {failed_count}")
    if valid_count == 0:
        print("[Pose Test] All samples failed PnP; returning a large penalty score.")
        return 1e16, 1e16, 1e16
    denom = valid_count
    return speed_total / denom, speed_t_total / denom, speed_r_total / denom


def _parse_json_samples(filedata, input_file):
    if not filedata.strip():
        return []

    decoder = json.JSONDecoder()
    samples = []
    index = 0
    length = len(filedata)

    while index < length:
        while index < length and (filedata[index].isspace() or filedata[index] == ","):
            index += 1
        if index >= length:
            break
        try:
            item, next_index = decoder.raw_decode(filedata, index)
        except json.JSONDecodeError as exc:
            line = filedata.count("\n", 0, exc.pos) + 1
            line_start = filedata.rfind("\n", 0, exc.pos) + 1
            column = exc.pos - line_start + 1
            raise ValueError(
                f"Invalid pseudo-label JSON in {input_file} at line {line}, "
                f"column {column}: {exc.msg}"
            ) from exc

        if isinstance(item, list):
            samples.extend(item)
        elif isinstance(item, dict):
            samples.append(item)
        else:
            raise ValueError(
                f"Invalid pseudo-label JSON in {input_file}: expected object or array, "
                f"got {type(item).__name__}"
            )
        index = next_index

    return samples


def format_and_copy_json(previous_config, project_root=None):
    if project_root is None:
        config = load_config(previous_config)
        run_dir = os.path.join(config["path_results"], previous_config)
    else:
        from src.run_paths import config_output_dir, normalize_config_paths, project_config_arg
        config_arg = project_config_arg(previous_config, project_root)
        config = load_config(os.path.join(project_root, config_arg))
        config = normalize_config_paths(config, project_root)
        run_dir = str(config_output_dir(config, config_arg, project_root))
    print("Changing the format to the labels")
    input_file = os.path.join(run_dir, "train.json")
    output_file = os.path.join(run_dir, "train_clean.json")
    print("\t --input file: ", input_file)
    print("\t --output file: ", output_file)

    with open(input_file, "r", encoding="utf-8") as file:
        filedata = file.read()

    samples = _parse_json_samples(filedata, input_file)
    if not samples:
        raise ValueError(
            f"No accepted pseudo labels in {input_file}; inspect the gate diagnostics "
            "before regenerating maps or starting the next training round."
        )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)

    split_submission = config["split_submission"]
    if split_submission == "sunlamp":
        dst = os.path.join(config["root_dir"], "sunlamp_train", "train.json")
    elif split_submission == "lightbox":
        dst = os.path.join(config["root_dir"], "lightbox_train", "train.json")
    else:
        raise ValueError("Pose loop only supports sunlamp/lightbox split_submission")

    print("copying labels to: ", dst)
    shutil.copyfile(output_file, dst)
