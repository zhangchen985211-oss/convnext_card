#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPnP (Perspective-n-Point) 算法实现

该模块实现了可反向传播的 PnP 算法，用于从 2D 图像点和 3D 空间点估计相机位姿。
主要包含以下类和函数：
1. BPnP - 支持批量输入的 2D 点，所有 2D 点集对应于一个公共的 3D 点集
2. BPnP_m3d - 支持批量输入的 2D 点和 3D 点，其中第 i 个 2D 点集对应于第 i 个 3D 点集
3. BPnP_fast - BPnP 的高效版本，通过忽略系数图的高阶导数来节省运行时间
4. get_coefs - 计算系数
5. batch_project - 批量投影 3D 点到 2D 平面
"""

import torch  # 导入PyTorch库
import cv2 as cv  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import kornia as kn  # 导入Kornia库用于计算机视觉操作


class BPnP(torch.autograd.Function):
    """
    可反向传播的 PnP 算法
    
    输入：
    - pts2d: 2D 关键点坐标，大小为 [batch_size, num_keypoints, 2]
    - pts3d: 3D 关键点坐标，大小为 [num_keypoints, 3]
    - K: 相机内参矩阵，大小为 [3, 3]
    
    输出：
    - P_6d: 6自由度位姿，大小为 [batch_size, 6]，其中每行的前3个元素是角轴旋转向量（欧拉向量），
      后3个元素是平移向量。
    
    注意：
    此 BPnP 函数假设小批量中的所有 2D 点集都对应于一个公共的 3D 点集。
    对于 pts3d 也是小批量的情况，请使用 BPnP_m3d 类。
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        """
        前向传播函数
        
        Args:
            ctx: 上下文对象，用于保存反向传播所需的张量
            pts2d: 2D 关键点坐标
            pts3d: 3D 关键点坐标
            K: 相机内参矩阵
            ini_pose: 初始位姿估计（可选）
        
        Returns:
            P_6d: 6自由度位姿
        """
        bs = pts2d.size(0)  # 批量大小
        n = pts2d.size(1)  # 关键点数量
        device = pts2d.device  # 设备（CPU或GPU）
        
        # 将张量转换为NumPy数组，用于OpenCV函数
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.detach().cpu())
        
        # 初始化输出位姿
        P_6d = torch.zeros(bs, 6, device=device)

        # 对每个样本执行PnP
        for i in range(bs):
            # 准备2D点数据
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n, 1, 2))
            
            # 使用RANSAC获取初始位姿估计
            if ini_pose is None:
                _, rvec0, T0, _ = cv.solvePnPRansac(
                    objectPoints=pts3d_np, 
                    imagePoints=pts2d_i_np, 
                    cameraMatrix=K_np, 
                    distCoeffs=None, 
                    flags=cv.SOLVEPNP_ITERATIVE, 
                    confidence=0.9999, 
                    reprojectionError=3
                )
            else:
                # 使用提供的初始位姿
                rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
            
            # 使用初始估计进行精确位姿计算
            _, rvec, T = cv.solvePnP(
                objectPoints=pts3d_np, 
                imagePoints=pts2d_i_np, 
                cameraMatrix=K_np, 
                distCoeffs=None, 
                flags=cv.SOLVEPNP_ITERATIVE, 
                useExtrinsicGuess=True, 
                rvec=rvec0, 
                tvec=T0
            )
            
            # 将结果转换为PyTorch张量
            angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).view(1, 3)
            T = torch.tensor(T, device=device, dtype=torch.float).view(1, 3)
            P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)

        # 保存反向传播所需的张量
        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数
        
        Args:
            ctx: 上下文对象，包含前向传播保存的张量
            grad_output: 来自上一层的梯度
        
        Returns:
            各输入参数的梯度
        """
        # 从上下文中获取保存的张量
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6  # 位姿参数数量

        # 初始化梯度
        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        # 对每个样本计算梯度
        for i in range(bs):
            # 初始化雅可比矩阵
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2*n, device=device)
            J_fz = torch.zeros(m, 3*n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            # 启用梯度计算
            torch.set_grad_enabled(True)
            
            # 准备可求导的张量
            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().view(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().view(-1).detach().requires_grad_()
            K_flat = K.clone().view(-1).detach().requires_grad_()

            # 计算雅可比矩阵的每一行
            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    # 清零之前的梯度
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                # 将角轴旋转向量转换为旋转矩阵
                R = kn.geometry.angle_axis_to_rotation_matrix(P_6d_flat[0:m-3].view(1, 3))

                # 构建投影矩阵
                P = torch.cat((R[0, 0:3, 0:3].view(3, 3), P_6d_flat[m-3:m].view(3, 1)), dim=-1)
                KP = torch.mm(K_flat.view(3, 3), P)
                
                # 准备点数据
                pts2d_i = pts2d_flat.view(n, 2).transpose(0, 1)
                pts3d_i = torch.cat((pts3d_flat.view(n, 3), torch.ones(n, 1, device=device)), dim=-1).t()
                
                # 投影3D点到2D平面
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2, :].view(1, n)

                # 计算残差
                r = pts2d_i * Si - proj_i[0:2, :]
                
                # 获取系数
                coefs = get_coefs(P_6d_flat.view(1, 6), pts3d_flat.view(n, 3), K_flat.view(3, 3))
                coef = coefs[:, :, j].transpose(0, 1)  # 大小: [2, n]
                
                # 计算标量函数
                fj = (coef * r).sum()
                
                # 反向传播计算梯度
                fj.backward()
                
                # 保存雅可比矩阵的元素
                J_fy[j, :] = P_6d_flat.grad.clone()
                J_fx[j, :] = pts2d_flat.grad.clone()
                J_fz[j, :] = pts3d_flat.grad.clone()
                J_fK[j, :] = K_flat.grad.clone()

            # 计算雅可比矩阵的逆
            inv_J_fy = torch.inverse(J_fy)

            # 计算位姿对各输入的雅可比矩阵
            J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            J_yK = (-1) * torch.mm(inv_J_fy, J_fK)

            # 计算梯度
            grad_x[i] = grad_output[i].view(1, m).mm(J_yx).view(n, 2)
            grad_z += grad_output[i].view(1, m).mm(J_yz).view(n, 3)
            grad_K += grad_output[i].view(1, m).mm(J_yK).view(3, 3)

        # 返回各输入的梯度，None表示ini_pose没有梯度
        return grad_x, grad_z, grad_K, None


class BPnP_m3d(torch.autograd.Function):
    """
    BPnP_m3d 支持 3D 关键点的批量输入，其中第 i 个 2D 关键点集对应于第 i 个 3D 关键点集。
    
    输入：
    - pts2d: 2D 关键点坐标，大小为 [batch_size, num_keypoints, 2]
    - pts3d: 3D 关键点坐标，大小为 [batch_size, num_keypoints, 3]
    - K: 相机内参矩阵，大小为 [3, 3]
    
    输出：
    - P_6d: 6自由度位姿，大小为 [batch_size, 6]，其中每行的前3个元素是角轴旋转向量（欧拉向量），
      后3个元素是平移向量。
    
    注意：
    对于小批量中的所有 2D 点集都对应于一个公共的 3D 点集的情况，请使用 BPnP 类。
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        """
        前向传播函数
        
        Args:
            ctx: 上下文对象，用于保存反向传播所需的张量
            pts2d: 2D 关键点坐标
            pts3d: 3D 关键点坐标
            K: 相机内参矩阵
            ini_pose: 初始位姿估计（可选）
        
        Returns:
            P_6d: 6自由度位姿
        """
        bs = pts2d.size(0)  # 批量大小
        n = pts2d.size(1)  # 关键点数量
        device = pts2d.device  # 设备（CPU或GPU）
        
        # 将相机内参转换为NumPy数组
        K_np = np.array(K.detach().cpu())
        
        # 初始化输出位姿
        P_6d = torch.zeros(bs, 6, device=device)

        # 对每个样本执行PnP
        for i in range(bs):
            # 准备2D和3D点数据
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n, 1, 2))
            pts3d_i_np = np.ascontiguousarray(pts3d[i].detach().cpu()).reshape((n, 3))
            
            # 使用RANSAC获取初始位姿估计
            if ini_pose is None:
                _, rvec0, T0, _ = cv.solvePnPRansac(
                    objectPoints=pts3d_i_np, 
                    imagePoints=pts2d_i_np, 
                    cameraMatrix=K_np, 
                    distCoeffs=None, 
                    flags=cv.SOLVEPNP_ITERATIVE, 
                    confidence=0.9999, 
                    reprojectionError=1
                )
            else:
                # 使用提供的初始位姿
                rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
            
            # 使用初始估计进行精确位姿计算
            _, rvec, T = cv.solvePnP(
                objectPoints=pts3d_i_np, 
                imagePoints=pts2d_i_np, 
                cameraMatrix=K_np, 
                distCoeffs=None, 
                flags=cv.SOLVEPNP_ITERATIVE, 
                useExtrinsicGuess=True, 
                rvec=rvec0, 
                tvec=T0
            )
            
            # 将结果转换为PyTorch张量
            angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).view(1, 3)
            T = torch.tensor(T, device=device, dtype=torch.float).view(1, 3)
            P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)

        # 保存反向传播所需的张量
        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数
        
        Args:
            ctx: 上下文对象，包含前向传播保存的张量
            grad_output: 来自上一层的梯度
        
        Returns:
            各输入参数的梯度
        """
        # 从上下文中获取保存的张量
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6  # 位姿参数数量

        # 初始化梯度
        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        # 对每个样本计算梯度
        for i in range(bs):
            # 初始化雅可比矩阵
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2*n, device=device)
            J_fz = torch.zeros(m, 3*n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            # 启用梯度计算
            torch.set_grad_enabled(True)
            
            # 准备可求导的张量
            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().view(-1).detach().requires_grad_()
            pts3d_flat = pts3d[i].clone().view(-1).detach().requires_grad_()
            K_flat = K.clone().view(-1).detach().requires_grad_()

            # 计算雅可比矩阵的每一行
            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    # 清零之前的梯度
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                # 将角轴旋转向量转换为旋转矩阵
                R = kn.angle_axis_to_rotation_matrix(P_6d_flat[0:m-3].view(1, 3))

                # 构建投影矩阵
                P = torch.cat((R[0, 0:3, 0:3].view(3, 3), P_6d_flat[m-3:m].view(3, 1)), dim=-1)
                KP = torch.mm(K_flat.view(3, 3), P)
                
                # 准备点数据
                pts2d_i = pts2d_flat.view(n, 2).transpose(0, 1)
                pts3d_i = torch.cat((pts3d_flat.view(n, 3), torch.ones(n, 1, device=device)), dim=-1).t()
                
                # 投影3D点到2D平面
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2, :].view(1, n)

                # 计算残差
                r = pts2d_i * Si - proj_i[0:2, :]
                
                # 获取系数
                coefs = get_coefs(P_6d_flat.view(1, 6), pts3d_flat.view(n, 3), K_flat.view(3, 3))
                coef = coefs[:, :, j].transpose(0, 1)  # 大小: [2, n]
                
                # 计算标量函数
                fj = (coef * r).sum()
                
                # 反向传播计算梯度
                fj.backward()
                
                # 保存雅可比矩阵的元素
                J_fy[j, :] = P_6d_flat.grad.clone()
                J_fx[j, :] = pts2d_flat.grad.clone()
                J_fz[j, :] = pts3d_flat.grad.clone()
                J_fK[j, :] = K_flat.grad.clone()

            # 计算雅可比矩阵的逆
            inv_J_fy = torch.inverse(J_fy)

            # 计算位姿对各输入的雅可比矩阵
            J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            J_yK = (-1) * torch.mm(inv_J_fy, J_fK)

            # 计算梯度
            grad_x[i] = grad_output[i].view(1, m).mm(J_yx).view(n, 2)
            grad_z[i] = grad_output[i].view(1, m).mm(J_yz).view(n, 3)
            grad_K += grad_output[i].view(1, m).mm(J_yK).view(3, 3)

        # 返回各输入的梯度，None表示ini_pose没有梯度
        return grad_x, grad_z, grad_K, None


class BPnP_fast(torch.autograd.Function):
    """
    BPnP_fast 是 BPnP 类的高效版本，通过忽略系数图的高阶导数来节省运行时间。
    这会牺牲可忽略的梯度精度，但节省大量运行时间。
    
    输入：
    - pts2d: 2D 关键点坐标，大小为 [batch_size, num_keypoints, 2]
    - pts3d: 3D 关键点坐标，大小为 [num_keypoints, 3]
    - K: 相机内参矩阵，大小为 [3, 3]
    
    输出：
    - P_6d: 6自由度位姿，大小为 [batch_size, 6]，其中每行的前3个元素是角轴旋转向量（欧拉向量），
      后3个元素是平移向量。
    
    注意：
    此 BPnP 函数假设小批量中的所有 2D 点集都对应于一个公共的 3D 点集。
    对于 pts3d 也是小批量的情况，请使用 BPnP_m3d 类。
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        """
        前向传播函数
        
        Args:
            ctx: 上下文对象，用于保存反向传播所需的张量
            pts2d: 2D 关键点坐标
            pts3d: 3D 关键点坐标
            K: 相机内参矩阵
            ini_pose: 初始位姿估计（可选）
        
        Returns:
            P_6d: 6自由度位姿
        """
        bs = pts2d.size(0)  # 批量大小
        n = pts2d.size(1)  # 关键点数量
        device = pts2d.device  # 设备（CPU或GPU）
        
        # 将张量转换为NumPy数组，用于OpenCV函数
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.detach().cpu())
        
        # 初始化输出位姿
        P_6d = torch.zeros(bs, 6, device=device)

        # 对每个样本执行PnP
        for i in range(bs):
            # 准备2D点数据
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n, 1, 2))
            
            # 使用RANSAC获取初始位姿估计
            if ini_pose is None:
                _, rvec0, T0, _ = cv.solvePnPRansac(
                    objectPoints=pts3d_np, 
                    imagePoints=pts2d_i_np, 
                    cameraMatrix=K_np, 
                    distCoeffs=None, 
                    flags=cv.SOLVEPNP_ITERATIVE, 
                    confidence=0.9999, 
                    reprojectionError=3
                )
            else:
                # 使用提供的初始位姿
                rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
            
            # 使用初始估计进行精确位姿计算
            success, rvec, T = cv.solvePnP(
                objectPoints=pts3d_np, 
                imagePoints=pts2d_i_np, 
                cameraMatrix=K_np, 
                distCoeffs=None, 
                flags=cv.SOLVEPNP_ITERATIVE, 
                useExtrinsicGuess=True, 
                rvec=rvec0, 
                tvec=T0
            )
            
            if not success or rvec is None or T is None:
                raise ValueError("BPnP refinement failed")
            if not np.isfinite(rvec).all() or not np.isfinite(T).all():
                raise ValueError("BPnP refinement returned a non-finite pose")
            xyz = pts3d_np @ cv.Rodrigues(rvec)[0].T + np.asarray(T).reshape(3)
            if not np.all(xyz[:, 2] > 1e-6):
                raise ValueError("BPnP refinement returned non-positive depths")

            # 将结果转换为PyTorch张量
            angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).view(1, 3)
            T = torch.tensor(T, device=device, dtype=torch.float).view(1, 3)
            P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)

        # 保存反向传播所需的张量
        ctx.save_for_backward(pts2d, P_6d, pts3d, K)
        return P_6d

    @staticmethod
    @torch.enable_grad()
    def backward(ctx, grad_output):
        """
        反向传播函数
        
        Args:
            ctx: 上下文对象，包含前向传播保存的张量
            grad_output: 来自上一层的梯度
        
        Returns:
            各输入参数的梯度
        """
        # 从上下文中获取保存的张量
        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6  # 位姿参数数量

        # 初始化梯度
        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        # 对每个样本计算梯度
        for i in range(bs):
            # 初始化雅可比矩阵
            J_fy = torch.zeros(m, m, device=device)
            J_fx = torch.zeros(m, 2*n, device=device)
            J_fz = torch.zeros(m, 3*n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            # 预先计算系数（使用detach()提高效率）
            coefficient_pose = P_6d[i].detach().clone().view(1, 6).requires_grad_(True)
            coefs = get_coefs(coefficient_pose, pts3d, K, create_graph=False).detach()

            # 准备可求导的张量
            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().view(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().view(-1).detach().requires_grad_()
            K_flat = K.clone().view(-1).detach().requires_grad_()

            # 计算雅可比矩阵的每一行
            for j in range(m):
                if j > 0:
                    # 清零之前的梯度
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                # 将角轴旋转向量转换为旋转矩阵
                R = kn.geometry.angle_axis_to_rotation_matrix(P_6d_flat[0:m-3].view(1, 3))

                # 构建投影矩阵
                P = torch.cat((R[0, 0:3, 0:3].view(3, 3), P_6d_flat[m-3:m].view(3, 1)), dim=-1)
                KP = torch.mm(K_flat.view(3, 3), P)
                
                # 准备点数据
                pts2d_i = pts2d_flat.view(n, 2).transpose(0, 1)
                pts3d_i = torch.cat((pts3d_flat.view(n, 3), torch.ones(n, 1, device=device)), dim=-1).t()
                
                # 投影3D点到2D平面
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2, :].view(1, n)

                # 计算残差
                r = pts2d_i * Si - proj_i[0:2, :]
                
                # 获取系数
                coef = coefs[:, :, j].transpose(0, 1)  # 大小: [2, n]
                
                # 计算标量函数
                fj = (coef * r).sum()
                
                # 反向传播计算梯度
                fj.backward()
                
                # 保存雅可比矩阵的元素
                J_fy[j, :] = P_6d_flat.grad.clone()
                J_fx[j, :] = pts2d_flat.grad.clone()
                J_fz[j, :] = pts3d_flat.grad.clone()
                J_fK[j, :] = K_flat.grad.clone()

            # A locally singular implicit system has no reliable geometric
            # derivative; leave that sample at zero rather than returning NaNs.
            if not all(torch.isfinite(item).all() for item in (J_fy, J_fx, J_fz, J_fK)):
                continue
            singular_values = torch.linalg.svdvals(J_fy)
            if singular_values[-1] <= singular_values[0] * 1e-8:
                continue
            # 使用线性最小二乘求解（更稳定）
            J_yx = (-1) * torch.linalg.lstsq(J_fy, J_fx).solution
            J_yz = (-1) * torch.linalg.lstsq(J_fy, J_fz).solution
            J_yK = (-1) * torch.linalg.lstsq(J_fy, J_fK).solution

            if not all(torch.isfinite(item).all() for item in (J_yx, J_yz, J_yK)):
                continue
            # 计算梯度
            grad_x[i] = grad_output[i].view(1, m).mm(J_yx).view(n, 2)
            grad_z += grad_output[i].view(1, m).mm(J_yz).view(n, 3)
            grad_K += grad_output[i].view(1, m).mm(J_yK).view(3, 3)

        # 返回各输入的梯度，None表示ini_pose没有梯度
        return grad_x, grad_z, grad_K, None


@torch.enable_grad()
def get_coefs(P_6d, pts3d, K, create_graph=True):
    """
    计算系数
    
    Args:
        P_6d: 6自由度位姿，大小为 [batch_size, 6]
        pts3d: 3D 关键点坐标，大小为 [num_keypoints, 3]
        K: 相机内参矩阵，大小为 [3, 3]
        create_graph: 是否创建计算图
    
    Returns:
        coefs: 系数，大小为 [num_keypoints, 2, 6]
    """
    device = P_6d.device
    n = pts3d.size(0)
    m = P_6d.size(-1)
    
    # 初始化系数张量
    coefs = torch.zeros(n, 2, m, device=device)
    
    # 启用梯度计算
    
    # 复制位姿数据以匹配关键点数量
    y = P_6d.repeat(n, 1)
    
    # 批量投影3D点到2D平面
    proj = batch_project(y, pts3d, K).squeeze()
    
    # 创建单位矩阵
    vec = torch.diag(torch.ones(n, device=device).float())
    
    # 计算每个维度的梯度
    for k in range(2):
        # 计算投影对P_6d的梯度
        y_grad = torch.autograd.grad(
            proj[:, :, k], 
            y, 
            vec, 
            retain_graph=True, 
            create_graph=create_graph
        )
        # 保存系数
        coefs[:, k, :] = -2 * y_grad[0].clone()
    
    return coefs

def batch_project(P, pts3d, K, angle_axis=True):
    """
    批量投影 3D 点到 2D 平面
    
    Args:
        P: 位姿，大小为 [batch_size, 6]（角轴+平移）或 [batch_size, 3, 4]（旋转矩阵+平移）
        pts3d: 3D 关键点坐标，大小为 [num_keypoints, 3]
        K: 相机内参矩阵，大小为 [3, 3]
        angle_axis: P 是否为角轴表示
    
    Returns:
        pts2d_pro: 投影后的 2D 点，大小为 [batch_size, num_keypoints, 2]
    """
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    
    # 为3D点添加齐次坐标
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)
    
    if angle_axis:
        # 将角轴旋转向量转换为旋转矩阵
        R_out = kn.geometry.conversions.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        # 构建投影矩阵 [R | t]
        PM = torch.cat((R_out[:, 0:3, 0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    else:
        # 直接使用提供的投影矩阵
        PM = P
    
    # 转换3D点到相机坐标系
    pts3d_cam = pts3d_h.matmul(PM.transpose(-2, -1))
    
    # 投影到图像平面
    pts2d_proj = pts3d_cam.matmul(K.t())
    
    # 归一化
    S = pts2d_proj[:, :, 2].view(bs, n, 1)
    pts2d_pro = pts2d_proj[:, :, 0:2].div(S)

    return pts2d_pro





