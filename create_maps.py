#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建关键点热图脚本

该脚本用于为数据集创建关键点热图，主要功能包括：
1. 加载相机参数和数据集标签
2. 根据位姿信息计算关键点在图像上的投影位置
3. 为每个关键点创建高斯热图
4. 保存热图到指定目录

使用多进程加速处理过程，适用于大规模数据集。
"""

import numpy as np  # 导入NumPy库用于数值计算
import json  # 导入JSON库用于处理JSON文件
import cv2
import os  # 导入操作系统相关功能
from tqdm import tqdm  # 导入进度条库
import argparse  # 导入命令行参数解析库
from multiprocessing import Pool  # 导入多进程池
from multiprocessing import freeze_support  # 导入多进程支持
from pathlib import Path

from src.cad_graph import load_kpts_mat
from src.run_paths import normalize_config_paths, project_config_arg

PROJECT_ROOT = Path(__file__).resolve().parent


def run_multiprocessing(func, i, n_processors, config):
    """
    运行多进程处理并收集结果
    
    Args:
        func: 要执行的函数
        i: 函数参数列表
        n_processors: 进程数量
    
    Returns:
        list: 函数执行结果列表
    """
    with Pool(processes=n_processors, initializer=initialize_worker, initargs=(config,)) as pool:
        return list(tqdm(pool.imap(func, i), total=len(i), desc="Heatmaps"))


class Camera:
    """
    相机参数访问的实用类
    """
    def __init__(self, speed_root):
        """
        初始化相机参数
        
        Args:
            speed_root (str): 数据集根目录路径
        """
        with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
            camera_params = json.load(f)
        
        # 焦距 [米]
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        
        # 像素数量
        self.nu = camera_params['Nu']  # 水平方向像素数
        self.nv = camera_params['Nv']  # 垂直方向像素数
        
        # 像素间距 [米/像素]
        self.ppx = camera_params['ppx']  # 水平像素间距
        self.ppy = camera_params['ppy']  # 垂直像素间距
        
        # 像素单位的焦距
        self.fpx = self.fx / self.ppx  # 水平焦距 [像素]
        self.fpy = self.fy / self.ppy  # 垂直焦距 [像素]
        
        # 相机矩阵
        self.k = camera_params['cameraMatrix']
        self.K = np.array(self.k)  # 相机内参矩阵
        
        # 畸变系数
        self.dcoef = camera_params['distCoeffs']


def quat2dcm(q):
    """
    将四元数转换为方向余弦矩阵
    改编自PyNav库
    
    Args:
        q (numpy.ndarray): 四元数 [q0, q1, q2, q3]
    
    Returns:
        numpy.ndarray: 3x3方向余弦矩阵
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("pose quaternion must contain four finite values")
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("pose quaternion must have non-zero norm")
    q = q / norm

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # 初始化方向余弦矩阵
    dcm = np.zeros((3, 3))

    # 计算方向余弦矩阵的元素
    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    """
    绘制2D高斯热图
    改编自 https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    
    Args:
        img: 输入图像
        pt: 关键点坐标
        sigma: 高斯分布的标准差
        type: 热图类型，'Gaussian'或'Cauchy'
    
    Returns:
        tuple: (绘制热图后的图像, 关键点是否在图像内)
    """
    pt = np.asarray(pt, dtype=np.float64).reshape(-1)
    if pt.shape != (2,) or not np.isfinite(pt).all() or not np.isfinite(sigma) or sigma <= 0:
        return img, False

    # 检查高斯分布是否有部分在图像边界内
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]  # 左上角坐标
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]  # 右下角坐标
    
    # 如果高斯分布大部分在图像外，返回原图像
    if (ul[0] >= 0.95*img.shape[1] or ul[1] >= 0.95*img.shape[0] or br[0] < 10 or br[1] < 10):
        return img, False
    
    # 生成高斯分布
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2  # 高斯分布中心
    
    # 生成高斯分布，中心值为1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
    
    # 计算高斯分布在图像内的范围
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    
    # 计算图像内的范围
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    
    # 将高斯分布绘制到图像上
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, True


def initialize_worker(config):
    """Read metadata in spawned workers; never delete maps during import."""
    global image_root, kptsmap_root, labels, kpts, cam
    speed_root = config["root_dir"]
    split = config["split_submission"]
    if split in ("sunlamp", "lightbox"):
        dataset_id = f"{split}_train"
    elif split == "synthetic":
        dataset_id = config.get("train_dataset_id") or config.get("source_dataset_id") or "synthetic"
    else:
        raise ValueError(f"Unsupported split_submission: {split}")
    image_root = os.path.join(speed_root, dataset_id, "images")
    kptsmap_root = os.path.join(speed_root, dataset_id, "kptsmap")
    with open(os.path.join(speed_root, dataset_id, "train.json"), encoding="utf-8") as stream:
        label_list = json.load(stream)
    labels = {label["filename"]: {"q": label["q_vbs2tango_true"], "r": label["r_Vo2To_vbs_true"]} for label in label_list}
    kpts_path = Path(speed_root) / "kpts.mat"
    kpts = load_kpts_mat(kpts_path).T
    cam = Camera(speed_root)
    return list(labels)


def save_map(sample_id):
    """
    为单个样本创建并保存关键点热图
    
    Args:
        sample_id (str): 样本文件名
    """
    # 读取图像
    img_name = os.path.join(image_root, sample_id)
    image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {img_name}")
    if image.shape[:2] != (1200, 1920):
        raise ValueError(f"Expected 1200x1920 SPEED+ image, got {image.shape[:2]}: {img_name}")

    # 获取位姿标签（四元数和平移向量）
    q_return, r = labels[sample_id]['q'], labels[sample_id]['r']

    # 将四元数转换为方向余弦矩阵
    q = quat2dcm(q_return)

    # 转换平移向量
    translation = np.asarray(r, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError(f"invalid translation label for {sample_id}")
    r2 = np.expand_dims(translation, axis=1)
    r2 = q @ (r2)

    # 计算关键点在相机坐标系中的位置
    tmp = q.T @ (kpts + r2)
    # Project only finite points in front of the camera. Invalid/behind-camera
    # points remain invisible instead of producing NaN heatmaps.
    kpts_im = np.zeros((kpts.shape[1], 2), dtype=np.float64)
    positive_depth = np.isfinite(tmp).all(axis=0) & (tmp[2, :] > 1e-6)
    if positive_depth.any():
        projected = cam.K @ (tmp[:, positive_depth] / tmp[2, positive_depth])
        projected = projected[:2, :].T
        positive_indices = np.flatnonzero(positive_depth)
        finite_projection = np.isfinite(projected).all(axis=1)
        kpts_im[positive_indices[finite_projection]] = projected[finite_projection]
        positive_depth[positive_indices[~finite_projection]] = False
    
    # 注释掉的代码：限制关键点坐标在图像范围内
    # kpts_im[kpts_im[:,0] >= 1920,0] = 1920-1
    # kpts_im[kpts_im[:,0] <= 0   ,0] = 0
    # kpts_im[kpts_im[:,1] >= 1200,1] = 1200-1
    # kpts_im[kpts_im[:,1] <= 0   ,1] = 0
    
    # 初始化热图和可见性标记
    pil_drawkpts = np.zeros((1200, 1920, 11), dtype=np.float32)  # 11个关键点的热图
    vis = np.zeros((11), dtype=bool)  # 关键点可见性标记
    
    # 为每个关键点创建热图
    for i, pt in enumerate(kpts_im):
        if not positive_depth[i]:
            continue
        # 绘制高斯热图
        aux, vis[i] = draw_labelmap(pil_drawkpts[:, :, i], pt, 15.0, type='Gaussian')
        # 如果关键点在图像内，归一化热图
        if vis[i]:
            aux = ((aux - np.min(aux)) / (np.max(aux) - np.min(aux)))
            pil_drawkpts[:, :, i] = aux

    # 过滤可见的关键点
    filt_kpts_im = kpts_im[vis, :]
    filt_kpts = kpts[:, vis]

    # 保存热图
    savename = os.path.join(kptsmap_root, Path(sample_id).stem)
    np.savez_compressed(savename, pil_drawkpts, filt_kpts_im, filt_kpts, vis)


def main():
    parser = argparse.ArgumentParser(description="Create SPEED+ keypoint heatmaps")
    parser.add_argument("--cfg", "--config", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    config_arg = project_config_arg(args.cfg, PROJECT_ROOT)
    with open(PROJECT_ROOT / config_arg, encoding="utf-8") as stream:
        config = normalize_config_paths(json.load(stream), PROJECT_ROOT)
    sample_ids = initialize_worker(config)
    if not sample_ids:
        raise ValueError("No labels available; inspect pseudo-label gate diagnostics")
    Path(kptsmap_root).mkdir(parents=True, exist_ok=True)
    # Cleanup belongs to one parent, never an imported/spawned worker.
    if config["split_submission"] in ("sunlamp", "lightbox"):
        for old_map in Path(kptsmap_root).glob("*.npz"):
            old_map.unlink()
    run_multiprocessing(save_map, sample_ids, max(1, args.num_workers), config)


if __name__ == "__main__":
    freeze_support()
    main()
