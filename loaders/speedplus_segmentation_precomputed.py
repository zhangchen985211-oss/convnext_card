#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
航天器位姿估计数据集加载器

该模块提供了用于加载和处理航天器位姿估计数据集的功能，包括：
1. 处理JSON格式的数据集文件
2. 四元数到方向余弦矩阵的转换
3. 点投影到图像帧
4. 相机参数管理
5. 数据集检查和可视化
6. PyTorch DataLoader兼容的数据集类
"""

import numpy as np  # 导入NumPy库用于数值计算
import json  # 导入JSON库用于处理JSON文件
import os  # 导入操作系统相关功能
from PIL import Image  # 导入PIL库用于图像处理
from matplotlib import pyplot as plt  # 导入Matplotlib库用于可视化

from torch.utils.data import Dataset  # 导入PyTorch的Dataset类
import cv2  # 导入OpenCV库用于图像处理

from src.cad_graph import load_kpts_mat


def _require_existing_path(path, description):
    """
    检查路径是否存在，不存在则抛出更明确的错误。

    Args:
        path (str): 需要检查的路径
        description (str): 路径含义说明
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到{description}: {path}")


def process_json_dataset(root_dir):
    """
    处理JSON格式的数据集文件
    
    Args:
        root_dir (str): 数据集根目录路径
    
    Returns:
        tuple: 包含两个元素的元组
            - partitions (dict): 不同数据分割的文件名列表
            - labels (dict): 训练数据的标签（四元数和平移向量）
    """
    # 加载合成训练数据的标签
    with open(os.path.join(root_dir, 'synthetic', 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    # 加载合成验证数据的标签
    with open(os.path.join(root_dir, 'synthetic', 'validation.json'), 'r') as f:
        test_image_list = json.load(f)

    # 加载sunlamp测试数据的标签
    with open(os.path.join(root_dir, 'sunlamp', 'test.json'), 'r') as f:
        sunlamp_image_list = json.load(f)

    # 加载lightbox测试数据的标签
    with open(os.path.join(root_dir, 'lightbox', 'test.json'), 'r') as f:
        lightbox_image_list = json.load(f)

    # 初始化数据分割和标签字典
    partitions = {'validation': [], 'train': [], 'sunlamp': [], 'lightbox': []}
    labels = {}

    # 处理训练数据
    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {
            'q': image_ann['q_vbs2tango_true'],  # 四元数
            'r': image_ann['r_Vo2To_vbs_true']   # 平移向量
        }

    # 处理验证数据
    for image in test_image_list:
        partitions['validation'].append(image['filename'])

    # 处理sunlamp测试数据
    for image in sunlamp_image_list:
        partitions['sunlamp'].append(image['filename'])

    # 处理lightbox测试数据
    for image in lightbox_image_list:
        partitions['lightbox'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):
    """
    将四元数转换为方向余弦矩阵
    改编自PyNav库
    
    Args:
        q (numpy.ndarray): 四元数 [q0, q1, q2, q3]
    
    Returns:
        numpy.ndarray: 3x3方向余弦矩阵
    """
    # 归一化四元数
    q = q / np.linalg.norm(q)

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


def project(q, r):
    """
    将点投影到图像帧以绘制坐标轴
    
    Args:
        q (numpy.ndarray): 四元数
        r (numpy.ndarray): 平移向量
    
    Returns:
        tuple: 投影后的x和y坐标
    """
    # 卫星坐标系中的参考点，用于绘制坐标轴
    p_axes = np.array([
        [0, 0, 0, 1],  # 原点
        [1, 0, 0, 1],  # x轴
        [0, 1, 0, 1],  # y轴
        [0, 0, 1, 1]   # z轴
    ])
    points_body = np.transpose(p_axes)

    # 转换到相机坐标系
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # 获取齐次坐标
    points_camera_frame = p_cam / p_cam[2]

    # 投影到图像平面
    points_image_plane = Camera.K.dot(points_camera_frame)
    x0, y0 = (points_image_plane[0], points_image_plane[1])
    
    # 应用畸变
    dist = Camera.dcoef
    
    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
    x1 = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
    y1 = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0
    
    # 计算最终像素坐标
    x = Camera.K[0, 0] * x1 + Camera.K[0, 2]
    y = Camera.K[1, 1] * y1 + Camera.K[1, 2]
    
    return x, y


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
        # 加载相机参数
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


class SatellitePoseEstimationDataset:
    """
    数据集检查类：方便访问单张图像及其对应的位姿 ground truth 数据
    """

    def __init__(self, root_dir='datasets/'):
        """
        初始化数据集
        
        Args:
            root_dir (str): 数据集根目录路径
        """
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):
        """
        加载图像为PIL图像
        
        Args:
            i (int): 图像索引
            split (str): 数据分割，可选值：'train', 'validation', 'sunlamp_train', 'lightbox_train'
        
        Returns:
            PIL.Image: 加载的图像
        """
        img_name = self.partitions[split][i]
        
        # 根据分割类型构建图像路径
        if split == 'train':
            img_name = os.path.join(self.root_dir, 'synthetic', 'images', img_name)
        elif split == 'validation':
            img_name = os.path.join(self.root_dir, 'synthetic', 'images', img_name)
        elif split == 'sunlamp_train':
            img_name = os.path.join(self.root_dir, 'sunlamp_train', 'images', img_name)
        elif split == 'lightbox_train':
            img_name = os.path.join(self.root_dir, 'lightbox_train', 'images', img_name)
        else:
            print()
            # 可以在这里添加错误处理
        
        # 打开并返回图像
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):
        """
        获取图像的位姿标签
        
        Args:
            i (int): 图像索引
        
        Returns:
            tuple: 四元数和平移向量
        """
        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):
        """
        可视化图像，并在训练图像上投影ground truth位姿的坐标轴
        
        Args:
            i (int): 图像索引
            partition (str): 数据分割，默认为'train'
            ax (matplotlib.axes.Axes): 绘图轴，默认为None
        """
        if ax is None:
            ax = plt.gca()
        
        # 加载图像
        img = self.get_image(i)
        ax.imshow(img)

        # 只为训练数据绘制位姿坐标轴
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya = project(q, r)
            # 绘制坐标轴
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')  # x轴（红色）
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')  # y轴（绿色）
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')  # z轴（蓝色）

        return


class PyTorchSatellitePoseEstimationDataset(Dataset):
    """
    可与PyTorch DataLoader一起使用的SPEED数据集类
    """

    def __init__(
        self,
        split='train',
        speed_root='datasets/',
        transform_input=None,
        transform_gt=None,
        config=None,
        dataset_id=None,
    ):
        """
        初始化PyTorch数据集
        
        Args:
            split (str): 数据分割，可选值：'train', 'validation', 'sunlamp', 'lightbox', 'sunlamp_train', 'lightbox_train'
            speed_root (str): 数据集根目录路径
            transform_input (callable): 输入图像的变换函数
            transform_gt (callable):  ground truth的变换函数
            config (dict): 配置字典
            dataset_id (str | None): 当 split 为 train/validation 时，真实读取的数据集子目录名。
                默认为 synthetic，也可指定 synthetic_sunlamp_style / synthetic_lightbox_style 等。
        """
        # 验证分割类型
        if split not in {'train', 'validation', 'sunlamp', 'lightbox', 'sunlamp_train', 'lightbox_train'}:
            raise ValueError('无效的分割类型，必须是\'train\'、\'validation\'、\'sunlamp\'或\'lightbox\'')

        # 根据分割类型设置图像和掩码根目录
        if split in {'train', 'validation'}:
            supervised_dataset_id = dataset_id or 'synthetic'
            dataset_root = os.path.join(speed_root, supervised_dataset_id)
            label_path = os.path.join(dataset_root, split + '.json')
            self.image_root = os.path.join(dataset_root, 'images')
            self.mask_root = os.path.join(dataset_root, 'kptsmap')

            _require_existing_path(dataset_root, f"监督数据集目录 `{supervised_dataset_id}`")
            _require_existing_path(label_path, f"监督标注文件 `{supervised_dataset_id}/{split}.json`")
            _require_existing_path(self.image_root, f"监督图像目录 `{supervised_dataset_id}/images`")
            if split == 'train':
                _require_existing_path(self.mask_root, f"监督热图目录 `{supervised_dataset_id}/kptsmap`")

            with open(label_path, 'r') as f:
                label_list = json.load(f)

        elif split in {'sunlamp_train', 'lightbox_train'}:
            self.image_root = os.path.join(speed_root, split, 'images')
            self.mask_root = os.path.join(speed_root, split, 'kptsmap')
            label_path = os.path.join(speed_root, split, 'train.json')

            _require_existing_path(self.image_root, f"训练图像目录 `{split}/images`")
            _require_existing_path(self.mask_root, f"训练热图目录 `{split}/kptsmap`")
            _require_existing_path(label_path, f"训练标注文件 `{split}/train.json`")

            with open(label_path, 'r') as f:
                label_list = json.load(f)

        else:
            self.image_root = os.path.join(speed_root, split, 'images')
            label_path = os.path.join(speed_root, split, 'test.json')

            _require_existing_path(self.image_root, f"测试图像目录 `{split}/images`")
            _require_existing_path(label_path, f"测试标注文件 `{split}/test.json`")

            with open(label_path, 'r') as f:
                label_list = json.load(f)

        # 解析输入
        self.sample_ids = [label['filename'] for label in label_list]
        self.train = (split == 'train') or (split == 'sunlamp_train') or (split == 'lightbox_train')
        self.validation = (split == 'validation')

        # 加载标签（仅用于训练和验证）
        if self.train or self.validation:
            self.labels = {
                label['filename']: {
                    'q': label['q_vbs2tango_true'],  # 四元数
                    'r': label['r_Vo2To_vbs_true']   # 平移向量
                }
                for label in label_list
            }

        # 加载资源
        self.kpts = load_kpts_mat(os.path.join(speed_root, "kpts.mat")).T
        self.cam = Camera(speed_root)  # 相机参数
        self.K = self.cam.K

        # 根据配置调整相机矩阵
        self.K[0, :] *= ((config["cols"])/1920)
        self.K[1, :] *= ((config["rows"])/1200)

        # 网络输入张量的变换
        self.transform_input = transform_input
        self.col_factor_input = ((config["cols"])/1920)
        self.row_factor_input = ((config["rows"])/1200)

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 数据集样本数量
        """
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
        
        Returns:
            dict: 包含图像和标签的字典
        """
        # 加载图像
        sample_id = self.sample_ids[idx]

        img_name = os.path.join(self.image_root, sample_id)
        pil_image = cv2.imread(img_name)
        torch_image = self.transform_input(pil_image)

        y = sample_id

        # 对于验证，只加载ground truth位姿
        if self.validation:
            q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

        # 对于训练，需要更多数据
        if self.train:
            q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

            # 在计算真实关键点位置之前，需要一些变换
            q = quat2dcm(q0)
            r = np.expand_dims(np.array(r0), axis=1)
            r = q @ (r)

            # 航天器关键点放置在相机前方
            kpts_cam = q.T @ (self.kpts + r)

            # 将关键点投影到相机
            kpts_im = self.K @ (kpts_cam / kpts_cam[2, :])
            kpts_im = np.transpose(kpts_im[:2, :])

            # 加载热图
            heatmap_id = sample_id.split(".jpg")[0] + ".npz"
            data_loaded = np.load(os.path.join(self.mask_root, heatmap_id), allow_pickle=True)
            
            pil_heatmap = data_loaded.f.arr_0.astype(np.float32)
            visible_kpts = data_loaded.f.arr_3.astype(np.bool_)
        
            torch_heatmap = self.transform_input(pil_heatmap)

        # 构建样本字典
        sample = dict()
        
        if self.train:
            sample["image"] = torch_image
            sample["heatmap"] = torch_heatmap
            sample["kpts_3Dcam"] = kpts_cam.astype(np.float32)
            sample["kpts_2Dim"] = kpts_im.astype(np.float32)
            sample["visible_kpts"] = visible_kpts
            sample["q0"] = np.array(q0).astype(np.float32)
            sample["r0"] = np.array(r0).astype(np.float32)
            sample["y"] = y

        if self.validation:
            sample["image"] = torch_image
            sample["q0"] = np.array(q0).astype(np.float32)
            sample["r0"] = np.array(r0).astype(np.float32)
            sample["y"] = y

        else:
            sample["image"] = torch_image
            sample["y"] = y

        return sample
