from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

"""
姿态评估工具。

这个模块用于比较预测位姿和真值位姿的差异，主要输出三类指标：
1. 平移误差 `speed_t`
2. 旋转误差 `speed_r`
3. 综合误差 `speed`

`src/pose_utils.py` 和 `src/official_scoring.py` 在验证及目标评估阶段调用
`speed_score()`，因此它属于最终 pose 评估链路的一部分。
"""


def rpy2quat(rpy=None):
    # 将 roll / pitch / yaw 表达转换为四元数，返回顺序为 [qx, qy, qz, qw]。
    # 这里输入顺序不是常见的 [r, p, y] 直接映射，而是按源码中的索引约定取值。
    r = rpy[1] 
    p = rpy[2]
    y = rpy[0]
    cRh = np.cos(r/2)
    sRh = np.sin(r/2)
    cPh = np.cos(p/2)
    sPh = np.sin(p/2)
    cYh = np.cos(y/2)
    sYh = np.sin(y/2)
    qs_cmpl = np.array([ -(np.multiply(np.multiply(sRh,cPh),cYh) - np.multiply(np.multiply(cRh,sPh),sYh)),
                         -(np.multiply(np.multiply(cRh,sPh),cYh) + np.multiply(np.multiply(sRh,cPh),sYh)),
                         -(np.multiply(np.multiply(cRh,cPh),sYh) - np.multiply(np.multiply(sRh,sPh),cYh)),
                         np.multiply(np.multiply(cRh,cPh),cYh) + np.multiply(np.multiply(sRh,sPh),sYh)])
    qs = np.real(qs_cmpl)
    return qs


def eul2quat(ax, ay, az, atol=1e-8):
    '''
    将 ZYX 欧拉角转换为四元数的向量部分。

    Args:
        ax: 绕 X 轴旋转角，单位为弧度。
        ay: 绕 Y 轴旋转角，单位为弧度。
        az: 绕 Z 轴旋转角，单位为弧度。
        atol: 当四元数标量部接近 0 时，数值稳定分支使用的容差。
    Return:
        长度为 3 的 numpy 数组，对应四元数的向量部 [qx, qy, qz]。

    '''
    # 先按 ZYX 欧拉角构造旋转矩阵，再从旋转矩阵反推四元数。
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    # 先根据旋转矩阵迹计算四元数标量部 qs。
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # 当 qs 非常接近 0 时，直接用常规公式会不稳定，因此切换到稳定求法。
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom
        qv[1] = (r[0,2] - r[2,0])/denom
        qv[2] = (r[1,0] - r[0,1])/denom
    return qv

def error_translation(t_pr, t_gt):
    # 计算预测平移和真值平移之间的欧氏距离误差。
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def error_orientation(q_pr, q_gt):
    # 计算预测四元数和真值四元数之间的夹角误差，输出单位为度。
    # q 的排列约定是 [qx, qy, qz, qw]。
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    # 四元数 q 与 -q 表示同一旋转，因此这里取绝对值消除符号二义性。
    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2*np.arccos(qdot)) # [deg]

def speed_score(t_pr, q_pr, t_gt, q_gt, applyThresh=True, rotThresh=0.5, posThresh=0.005):
    """
    计算 SPEED 风格的姿态评分。

    参数:
        t_pr: 预测平移向量。
        q_pr: 预测四元数。
        t_gt: 真值平移向量。
        q_gt: 真值四元数。
        applyThresh: 是否启用挑战赛式阈值截断。
        rotThresh: 旋转误差阈值，单位度。
        posThresh: 归一化平移误差阈值。

    返回:
        speed: 综合误差。若启用阈值且同时满足姿态和平移阈值，则记为 0。
        speed_t: 归一化平移误差。
        speed_r: 旋转误差的弧度形式。
    """
    # rotThresh: rotation threshold [deg]
    # posThresh: normalized translation threshold [m/m]
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(q_pr, q_gt) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    # 平移误差用真值平移的模长归一化，使不同距离尺度下的误差更可比。
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    # 综合评分里旋转项使用弧度而不是角度。
    speed_r = np.deg2rad(err_q)

    # 若启用阈值策略，并且旋转和平移都足够小，则直接记为完美预测。
    if applyThresh and err_q < rotThresh and speed_t < posThresh:
        speed = 0.0
    else:
        # 否则综合误差 = 归一化平移误差 + 旋转弧度误差。
        speed = speed_t + speed_r

    return speed, speed_t, speed_r
