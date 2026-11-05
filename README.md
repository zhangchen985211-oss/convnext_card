# 路线 B：ConvNeXt-Tiny + CAD 关系关键点解码

本目录是独立项目根目录，已包含 ConvNeXt-CARD、共享训练/评测组件、SPEED+ loader、BPnP、伪标签流程和提交导出程序。`train.py` 是启动入口，具体训练实现在 `train_convnext_card.py`。服务器只需上传本目录；数据集、CUDA 环境以及可选 ImageNet 权重需单独提供。完整命令见 [RUNBOOK.md](RUNBOOK.md)，算法与调研建议见 [SERVER_GUIDE.md](SERVER_GUIDE.md)。

```bash
cd "$CODE_ROOT"  # CODE_ROOT 指向上传后的 route_b_convnext_card
python server/configure_server.py --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python server/preflight.py --data-root "$DATA_ROOT"
python prepare_cad_graph.py --kpts "$DATA_ROOT/kpts.mat" --output assets/speed_keypoint_graph.json
python create_maps.py --cfg server_configs/source.json --num-workers 2
python train.py --cfg server_configs/source.json
```

配置脚本生成本地 `server_configs/`，不改模板。首次用 ImageNet 预训练 ConvNeXt 需要网络或 timm 本地缓存；离线时可用配置参数 `--no-pretrained-backbone`，但这会改变 backbone 初始化条件。

## 1. 目标与路线边界

本路线独立完成“更换更强主干并建立结构化位姿预测网络”的实验。目标是在 SPEED/SPEED+ 上得到一个可训练、可评价、可部署的单目航天器位姿模型，并将创新集中在 ConvNeXt-Tiny 主干后的 CAD 条件关键点关系解码器，而不是把多个现成模块简单堆叠。

本路线不依赖其他实验路线的代码、模块或结果。它可以单独训练、单独生成伪标签、单独评价，也可以在论文中作为一套完整方法描述。

建议论文工作名：

> ConvNeXt-CARD: CAD-Conditioned Relational Decoding for Monocular Spacecraft Pose Estimation under Domain Shift

中文可写成：

> 面向域偏移单目航天器位姿估计的 ConvNeXt-CARD：CAD 条件关键点关系解码方法

## 2. 核心创新点

### 2.1 ConvNeXt-Tiny 多尺度主干

采用 ImageNet 预训练 ConvNeXt-Tiny 提取多尺度特征，并通过轻量 FPN 恢复高分辨率关键点信息。ConvNeXt-Tiny 只是主干选择，不能单独作为论文创新；它在低照度、遮挡和尺度变化下是否优于其他主干，仍需相同输入、增强与训练预算下的实验验证。

### 2.2 CAD 条件关键点关系解码器

SPEED 的 11 个关键点不是互相独立的像素点，它们对应固定航天器 CAD 模型上的 3D 点。为此，在解码器中加入一个轻量的关键点关系模块：

1. 用 kpts.mat 中的 3D 关键点建立固定关系矩阵；
2. 从粗关键点 heatmap 提取 11 个关键点 token；
3. 用 CAD 关系矩阵作为 token attention 的相对偏置；
4. 将更新后的 token 重新注入高分辨率特征图；
5. 输出精细 heatmap 和 depth；
6. 用 CAD 边长和 2D 投影一致性约束训练。

该模块不直接复制 StructGCN 的 GCN、KSN 或 DECA-GCN，也不替换 BPnP。它的实现重点是“CAD 关系先验如何进入热图解码过程”，而不是重新设计一个独立的姿态回归器。

### 2.3 粗到细的几何耦合预测

网络先在较低分辨率上得到粗定位，再利用关键点 token 和 CAD 关系完成高分辨率细化：

~~~text
ConvNeXt-Tiny
    ↓
多尺度 FPN
    ↓
粗 heatmap + 粗 depth
    ↓
11 个关键点 token
    ↓
CAD 条件关系注意力
    ↓
细 heatmap + 细 depth
    ↓
BPnP / PnP
    ↓
姿态和重投影损失
~~~

这使网络输出具有明确的 2D—3D 关系，而不是只依赖每个关键点的局部峰值。

### 2.4 论文中的贡献表述

在结果成立后，建议将贡献写成：

1. 构建一个适用于航天器 11 点布局的 CAD 条件关键点关系解码器；
2. 将 CAD 关系先验、粗到细 heatmap/depth 预测和可微 BPnP 统一到一个训练目标中；
3. 在 SPEED+ 的 Lightbox 和 Sunlamp 域上同时报告精度、长尾误差、伪标签覆盖率和计算成本。

不要写成：

- ConvNeXt-Tiny 本身是原创网络；
- 首次使用关键点图结构；
- 首次使用伪标签筛选或重投影损失；
- 只用一个主干替换就解决了 domain gap。

## 3. 相关文献带来的设计约束

| 文献 | 可借鉴内容 | 本路线的处理 |
|---|---|---|
| ConvMamSPN | ConvNeXt-Tiny、几何伪标签和域迁移在航天器任务上可行 | 只借鉴主干选择和任务适配思路，不复制其 Mamba 与几何伪标签流程 |
| SPNv3 | 预训练、输入分辨率和风格增强对 SPEED+ 泛化影响明显 | 用作强基线和增强参考，不把“换 ViT”写成创新 |
| PVSPE | 多尺度特征和姿态不确定性建模有效 | 第一版不加入 Matrix-Fisher 或 Gaussian pose head，避免模型过重 |
| StructGCN | CAD 结构关系可改善外点和关键点耦合 | 采用轻量关系注意力与边一致性损失，不照搬完整 GCN/DECA-GCN |
| DINOv3 自监督 ViT | 大模型和自训练能提高 HIL 泛化 | 暂不采用 3 亿级参数模型，避免显存和训练周期失控 |
| 2025 单 CNN 方法 | 低置信度样本可触发二次推理 | 作为后续工程实验，不放入第一版主模型 |

## 4. 代码策略：修改现有工程，不从零重写

### 4.1 直接复用

以下模块保持原有接口和评价口径：

- BPnP/：可微 PnP；
- loaders/：SPEED/SPEED+ 数据读取；
- src/pose_utils.py 中的去畸变、heatmap 解码、PnP 和投影工具；
- src/pose_solver.py 中与网络架构无关的推理位姿求解；
- src/official_scoring.py 中的官方评分逻辑；
- 现有 kpts.mat 和相机内参；
- 现有 JSON 伪标签格式；
- 现有训练中的 heatmap、PnP 重投影和 3D 结构损失定义。

### 4.2 新增文件

~~~text
models/poseconvnext.py
models/cad_relation_decoder.py
src/cad_graph.py
src/losses_pose_geometry.py
generate_pseudo_labels.py
prepare_cad_graph.py
train.py
train_convnext_card.py
main_loop.py
configs/experiment_convnext_card.json
configs_loop_lightbox/loop_convnext_card_niter_0000.json
~~~

### 4.3 根目录文件的使用方式

`BPnP/`、`loaders/`、`src/pose_utils.py` 和 `src/training_common.py` 提供公共训练、几何和数据工具；
`src/pose_solver.py` 与 `src/official_scoring.py` 提供架构无关的位姿求解和评分逻辑。
`evaluate_target.py` 评测单个 ConvNeXt-CARD 权重，`score_checkpoints_official.py` 批量推理并输出官方评分排名、失败样本和曲线；具体命令见 [RUNBOOK.md](RUNBOOK.md)。
不同路线的 checkpoint 由各自配置的 `path_results` 隔离。

## 5. 网络结构设计

### 5.1 ConvNeXt-Tiny 编码器

使用 timm 的 ImageNet 预训练模型：

~~~python
backbone = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    features_only=True,
    out_indices=(0, 1, 2, 3),
)
~~~

运行时读取实际通道数，不要硬编码：

~~~text
C2: 1/4
C3: 1/8
C4: 1/16
C5: 1/32
~~~

启动时把 feature_channels、输入尺寸、参数量和 FLOPs 写入 run_manifest.json。

### 5.2 多尺度 FPN

每一层经过 1×1 Conv、GroupNorm、GELU，统一到 128 个通道，再采用自顶向下融合：

~~~text
P5 → 上采样并融合 C4 → P4
P4 → 上采样并融合 C3 → P3
P3 → 上采样并融合 C2 → P2
~~~

最终 P2 为输入的 1/4 分辨率。输入 512×512 时，P2 为 128×128；输入 640×640 时，P2 为 160×160。

### 5.3 粗预测头

在 P3 上输出：

~~~python
hm_coarse:     [B, 11, H/8, W/8]
depth_coarse:  [B, 11, H/8, W/8]
~~~

heatmap 使用当前项目已有的 target 编码和数值范围。depth 头也必须保留当前项目的深度参数化，不能在模型替换时改变物理尺度。

### 5.4 关键点 token 提取

对每个 coarse heatmap 做 soft-argmax 得到粗坐标 u_k，并按 heatmap 权重从 P3/P2 特征中池化得到 token：

~~~text
t_k = Σ_xy softmax(hm_k(x,y)) · F(x,y)
~~~

得到的 token 维度建议为 128，数量固定为 11。

### 5.5 CAD 关系矩阵

从 kpts.mat 读取 11 个 CAD 关键点三维坐标 X_k，计算归一化距离：

~~~text
d_ij = ||X_i - X_j|| / median pairwise distance
~~~

关系矩阵建议由两类边组成：

1. 每个点连接距离最近的 2 个点；
2. 对明显属于同一结构部件的点加入人工确认边。

当前 `build_speedplus_graph()` 使用固定的 11 点拓扑边，并从 `kpts.mat` 计算边距离；图会保存到
`assets/speed_keypoint_graph.json`。边集合目前是代码中的显式先验，正式论文应列出关键点编号和边定义，
并做“固定拓扑图 vs. 最近邻图”的敏感性对照。

### 5.6 CAD 条件关系注意力

对 11 个 token 使用轻量关系注意力：

~~~text
q_i = Wq t_i
k_j = Wk t_j
v_j = Wv t_j
a_ij = softmax_j(q_i · k_j / sqrt(d) + b(d_ij))
t'_i = t_i + Σ_j a_ij v_j
~~~

其中 b(d_ij) 是由 CAD 距离经过小型 MLP 得到的相对偏置。只在 CAD 图中的邻接点和自身上计算，避免 11 点全连接带来的无意义自由度。

关系注意力输出再通过 scatter 模块注入 P2：

~~~text
F_refined(x,y) = P2(x,y) + Σ_k G(x,y; u_k, σ) · Wt'_k
~~~

G 是以粗坐标为中心的二维 Gaussian。网络会在粗定位附近重点利用结构关系，不会把背景区域的 token 关系扩散到整张图。

### 5.7 精细预测头

在 F_refined 上输出：

~~~python
hm_fine:       [B, 11, H/4, W/4]
depth_fine:    [B, 11, H/4, W/4]
~~~

当前模型最终返回：

~~~python
{
    "coarse": {"hm_c": hm_coarse, "depth": depth_coarse},
    "fine": {"hm_c": hm_fine, "depth": depth_fine},
    "relation_tokens": relation_tokens,
    "coarse_coords": coarse_coords
}
~~~

兼容旧 PnP 工具时设置 `legacy_output=True`，模型返回 `[coarse, fine]` 两个 stage 字典。
当前没有单独的 `relation_aux` 边预测头；CAD 关系通过 attention 和 edge loss 进入训练。

## 6. 损失函数

### 6.1 有标签 synthetic 数据

~~~text
L_source =
  L_hm
+ λ_pnp L_reproj
+ λ_cad L_cad_edge
+ λ_cons L_stage_cons
~~~

建议初始权重：

~~~text
λ_pnp   = 0.5
λ_cad   = 0.1
λ_cons  = 0.1
~~~

先固定这些值完成基线，不要一开始做大规模权重搜索。

### 6.2 Heatmap 和 depth 损失

coarse 和 fine 分别与原项目 target 计算 MSE 或当前已使用的 heatmap loss。fine 分支权重略高：

~~~text
L_hm = 0.4 · MSE(hm_coarse, target)
     + 0.6 · MSE(hm_fine, target)
~~~

当前实现对 synthetic 训练样本使用数据读取器提供的 camera-space 关键点深度，计算对数域 Smooth L1 深度损失；目标域伪标签若包含可用位姿，也会沿用同一接口。伪标签阶段不会把目标测试集真实位姿写入训练文件。若后续发现深度监督放大伪标签噪声，可单独关闭 `lambda_depth`。

~~~text
L_depth = SmoothL1(log(depth_fine), log(depth_target))
~~~

### 6.3 CAD 边一致性损失

由 fine heatmap 和 depth 解码得到相机坐标点 P_i，对 CAD 图边计算归一化边长：

~~~text
L_cad_edge =
mean_(i,j)∈E | ||P_i - P_j|| / s_pred
              - ||X_i - X_j|| / s_cad |
~~~

其中 s_pred 和 s_cad 分别为当前样本的预测尺度和 CAD 参考尺度。该损失只约束相对结构，不要求网络直接预测绝对尺度。

### 6.4 重投影损失

用 11 个 3D CAD 点和预测 2D 关键点通过 BPnP 得到 R,t，再投影回图像：

~~~text
L_reproj = MSE(π(RX_i+t), u_i^GT) + MSE(π(RX_i+t), u_i^pred)
~~~

当前代码使用两个 MSE 项，分别约束投影点接近 GT 投影和网络预测点；Charbonnier/Huber
可以作为后续稳健损失消融，但尚未接入当前训练器。

### 6.5 阶段一致性损失

将 coarse 坐标上采样到 fine 尺度：

~~~text
L_stage_cons = mean_i ||u_i^fine - upsample(u_i^coarse)||_1
~~~

该项只作为稳定训练的辅助项，不单独用于伪标签筛选。

### 6.6 目标域伪标签训练（当前实现与扩展项）

当前循环只把通过 `staged_quality_fallback` 的有效位姿、深度范围和重投影检查的预测写入伪标签文件；该质量门当前不计算 CAD edge 误差或显式可见率。训练器对这些样本使用与源域相同的 heatmap、depth、BPnP、CAD edge 和阶段一致性损失；它尚未读取 JSON 中的
`quality_weight`，也没有按 gate 分数做 sample weighting：

质量加权目标损失可以作为后续扩展，不应当作当前代码已实现的部分：

~~~text
L_target = w_sample · (L_hm_pseudo + λ_cad L_cad_edge + λ_reproj L_reproj)
~~~

sample weight 由重投影误差、CAD 边误差和可见点比例共同决定：

~~~text
w_sample = clip(
    exp(-e_reproj / τ_reproj)
  · exp(-e_cad / τ_cad)
  · visibility_ratio,
  w_min, 1
)
~~~

当前 loop 配置默认不冻结 backbone；若目标域伪标签噪声导致漂移，可在 loop 配置加入
`freeze_backbone_epochs`，先冻结主干数轮。sample weight 和逐层解冻属于需要独立验证的后续策略。

## 7. 训练流程

### 7.1 B0：环境和接口检查

只做静态和单 batch 检查：

~~~text
输入：[1, 3, 512, 512]（当前 A6000 起始配置）
coarse heatmap：由 timm ConvNeXt stride 和输入尺寸决定，启动时以实际 shape 检查为准
fine heatmap：由 FPN P2 决定，启动时以实际 shape 检查为准
depth channel：11
~~~

必须通过：

- 前向无 NaN/Inf；
- soft-argmax 坐标落在图像范围；
- BPnP 能返回有效姿态；
- 反向传播后每个新模块都有非零梯度；
- 只加载 ImageNet encoder 时分类头被正确跳过；
- 保存和重新加载 checkpoint 后输出一致。

### 7.2 B1：source-only 主干替换

第一阶段只训练 synthetic 数据，关闭目标域伪标签：

~~~text
输入尺寸：512×512
batch size：1
AMP：开启
gradient accumulation：2
optimizer：AdamW
backbone lr：1e-5
decoder lr：1e-4
head lr：1e-4
weight decay：1e-4
warmup：1–2 epoch
~~~

训练策略：

1. 当前 source 配置的前 3 个 epoch 会冻结整个 ConvNeXt encoder；
2. 只训练 FPN、关系解码器和输出头；
3. 第 3 个 epoch 后解冻 encoder，backbone 使用较小学习率；
4. backbone 使用较小学习率，decoder/head 使用 10 倍学习率；
5. 当前 source 配置同时打开 L_hm、L_depth、L_pnp、L_cad_edge 和 L_stage_cons；
6. 若做消融，可从 heatmap/depth/PnP 开始逐项打开结构损失；`L_relation_aux` 尚未实现。

第一版不得同时加入风格增强、伪标签和复杂损失，否则无法判断失败来源。

### 7.3 B2：结构模块验证

固定 ConvNeXt 和训练协议，依次打开：

~~~text
B2-0：普通 FPN + heatmap/depth
B2-1：加入 CAD 条件关系注意力
B2-2：加入 CAD 边一致性损失
B2-3：加入粗到细阶段一致性
~~~

每一步都保存 Lightbox 和 Sunlamp 的验证结果。若某一步 source 指标下降超过 10%，先检查坐标尺度和 CAD 点编号，不要继续添加模块。

### 7.4 B3：风格和传感器增强

按以下顺序增加增强：

1. 高斯噪声、读出噪声、轻微模糊；
2. gamma、对比度、局部过曝；
3. flare 和暗部裁剪；
4. style_transfer_fda.py 生成目标域风格 synthetic 图；
5. RandConv 或 StyleAug。

所有增强必须保持关键点坐标和 CAD 标签不变。每个增强单独记录开关和随机种子。

### 7.5 B4：目标域伪标签自训练

伪标签生成流程（当前门控项不包含 CAD edge 或显式可见率）：

~~~text
目标图像
  ↓
ConvNeXt-CARD 推理
  ↓
fine heatmap/depth 解码
  ↓
BPnP 求姿态
  ↓
重投影误差 + 深度范围 + PnP 内点数
  ↓
接受、降权或拒绝
  ↓
生成 JSON 伪标签
  ↓
全网络更新；可通过 `freeze_backbone_epochs` 冻结 encoder
~~~

推荐 3 轮：

~~~text
第 0 轮：source-only checkpoint
第 1 轮：高阈值，保留约 20%–30% 目标样本
第 2 轮：阈值略放宽，保留约 40%–60%
第 3 轮：只在验证集不恶化时继续
~~~

每轮至少保存伪标签总数、接受/拒绝数量、重投影误差和深度范围；CAD edge 误差、显式可见率和 gate sample weighting 尚未接入当前代码。

## 8. 代码接口细节

### 8.1 模型构造

新增 models/poseconvnext.py：

~~~python
class ConvNeXtCARD(nn.Module):
    def __init__(self, cfg):
        self.backbone = build_convnext_features(cfg)
        self.fpn = MultiScaleFPN(...)
        self.coarse_head = PoseHead(...)
        self.relation_decoder = CADRelationDecoder(...)
        self.fine_head = PoseHead(...)

    def forward(self, x):
        features = self.backbone(x)
        pyramid = self.fpn(features)
        coarse = self.coarse_head(pyramid["P3"])
        tokens = extract_keypoint_tokens(pyramid, coarse["hm_c"])
        refined = self.relation_decoder(pyramid["P2"], tokens)
        fine = self.fine_head(refined)
        return {"coarse": coarse, "fine": fine}
~~~

### 8.2 CAD 图构造

新增 src/cad_graph.py：

~~~python
def build_cad_graph(kpts_3d, knn=2, manual_edges=None):
    ...

def load_cad_graph(path):
    ...

def normalized_edge_lengths(points_3d, edges):
    ...
~~~

训练启动时验证：

- 图中恰好有 11 个点；
- 所有边索引均在 [0, 10]；
- 每个点至少有一条边；
- CAD 距离没有 NaN/Inf；
- kpts.mat 的编号与 heatmap channel 一致。

### 8.3 伪标签接口

`generate_pseudo_labels.py` 输出字段沿用本项目 JSON 协议，并由根目录
`format_and_copy_json()` 清洗；额外的质量诊断写入相邻的 stats/diagnostics 文件。

~~~json
{
  "image_id": "xxx",
  "keypoints_2d": [[u1, v1], "..."],
  "depth": ["..."],
  "rotation": ["..."],
  "translation": ["..."],
  "visibility": ["..."],
  "reprojection_error": 0.0,
  "cad_edge_error": 0.0,
  "quality_weight": 0.0,
  "status": "accepted"
}
~~~

旧数据没有 cad_edge_error 或 quality_weight 时，读取器必须使用默认值，保证旧实验可运行。

### 8.4 checkpoint 和 manifest

每个 checkpoint 同目录保存：

~~~text
model.pt
config.json
run_manifest.json
metrics.json
~~~

run_manifest.json 至少包含 git commit、Python/PyTorch/timm 版本、输入分辨率、batch size、seed、backbone 是否预训练、参数量和 FLOPs、训练数据目录、伪标签轮次和评分脚本版本。

## 9. 配置样例

~~~json
{
  "model_name": "convnext_card",
  "backbone": "convnext_tiny",
  "pretrained": true,
  "rows": 512,
  "cols": 512,
  "num_keypoints": 11,
  "decoder_channels": 128,
  "relation_heads": 4,
  "cad_graph": "assets/speed_keypoint_graph.json",
  "freeze_backbone_epochs": 3,
  "backbone_lr": 0.00001,
  "decoder_lr": 0.0001,
  "head_lr": 0.0001,
  "lambda_hm": 1.0,
  "lambda_depth": 0.2,
  "lambda_pnp": 0.5,
  "lambda_cad": 0.1,
  "lambda_stage_cons": 0.1,
  "amp": true,
  "batch_size": 1,
  "gradient_accumulation_steps": 2
}
~~~

这是当前代码实际读取的关键配置片段；完整模板还包含数据路径、验证、伪标签筛选和图像增强参数。输入尺寸由 `rows` 与 `cols` 控制。

## 10. 实验矩阵

### 10.1 必做消融

| 编号 | 主干 | 关系解码器 | CAD 边损失 | UDA | 目的 |
|---|---|---|---|---|---|
| B0 | ConvNeXt-Tiny | 否 | 否 | 否 | 主干替换基线 |
| B1 | ConvNeXt-Tiny | 是 | 否 | 否 | 验证解码器 |
| B2 | ConvNeXt-Tiny | 是 | 是 | 否 | 验证结构损失 |
| B3 | ConvNeXt-Tiny | 是 | 是 | 是 | 完整方法 |

### 10.2 域泛化实验

分别训练和测试：

~~~text
synthetic → Lightbox
synthetic → Sunlamp
synthetic + style → Lightbox
synthetic + style → Sunlamp
~~~

如果条件允许，再加入只用于测试的 HIL 子集，不能用其姿态标签调阈值。

### 10.3 必须报告的指标

每个域都报告官方 E_pose 或连续评分、rotation mean/median/P95、translation mean/median/P95、normalized translation error、PnP 有效解比例、伪标签覆盖率、平均和 P95 推理延迟、峰值显存、参数量和 FLOPs。

报告结果时同时给出随机种子和三次运行的均值、标准差；若时间不足，至少保留一个固定 seed 的完整日志，并明确说明。

## 11. 失败判断与回退

### 11.1 source-only 不收敛

按顺序排查：

1. 将输入从 512 降到 384；
2. 冻结 ConvNeXt 前三个 stage；
3. 暂时关闭 L_cad、L_relation 和 L_stage_cons；
4. 检查 fine heatmap target 是否与 1/4 分辨率对应；
5. 检查 depth 的单位和归一化；
6. 检查 kpts.mat 的点顺序。

### 11.2 结构模块使结果变差

先保留关系注意力，关闭 L_cad；如果仍然变差，使用距离最近邻图替换人工边，并检查是否存在错误的关键点编号。

### 11.3 source 正常但 Lightbox/Sunlamp 变差

按顺序加入传感器噪声、gamma/flare、style transfer，再启动目标域伪标签。不要同时改变输入分辨率、相机参数和评分脚本。

### 11.4 显存超限

按顺序：

~~~text
AMP
→ batch size 1
→ gradient accumulation
→ 输入 384/448
→ 冻结前两个 ConvNeXt stage
→ decoder channels 128 改为 96
~~~

不得为了省显存删除 11 个关键点或改变官方评价协议。

### 11.5 伪标签污染

降低第一轮接受比例，增大重投影和 CAD 边误差阈值，或者只更新 decoder/head。若连续两轮官方评分下降，停止自训练并回退到 source-only checkpoint。

## 12. 最小可交付版本

第一版不要同时实现所有增强。最小闭环为：

~~~text
ConvNeXt-Tiny
→ FPN
→ 普通 coarse/fine heatmap + depth
→ BPnP
→ source-only 训练
→ 官方评分
~~~

第二版加入：

~~~text
CAD 关系注意力
→ CAD 边一致性损失
→ 结构化伪标签
→ 三轮目标域自训练
~~~

只要 B2 或 B3 在 Lightbox、Sunlamp 至少一个域稳定优于 B0，并且 P95 长尾误差下降，就具备继续写论文的价值。若精度提升有限但延迟和显存明显增加，应将其定位为强主干研究结果，而不是部署模型。

## 13. 实施顺序

~~~text
第 1 步：检查 timm、CUDA、显存和输入尺寸
第 2 步：完成 ConvNeXt-Tiny + FPN 单 batch 测试
第 3 步：完成 source-only 训练和官方评分
第 4 步：加入 CAD 图和关系解码器
第 5 步：加入 CAD 边一致性损失
第 6 步：加入传感器/风格增强
第 7 步：生成目标域伪标签并完成三轮自训练
第 8 步：做 B0–B3 消融和部署成本统计
~~~

最终原则是：只改主干、解码器和本路线定义的结构损失，数据、相机模型、关键点编号、BPnP 接口和官方评分保持固定。这样每次结果都能追溯到一个明确的改动。

## 14. 创新点、证据要求与论文边界

### 14.1 本路线的明确创新点

1. **CAD 条件关系解码**：将 11 个关键点的 CAD 三维关系编码为 attention relative bias，并用于 coarse token 到 fine feature 的关系传播。
2. **粗到细的 2D—3D 耦合**：粗 heatmap 先提供关键点位置和 token，关系解码器再生成 fine heatmap/depth，使精细定位受到 CAD 几何关系约束。
3. **结构损失与可微 BPnP 联合优化**：同时约束关键点热图、depth、CAD 相对边长和姿态重投影，避免只优化局部 heatmap 峰值。
4. **轻量结构先验**：使用固定 CAD 图和关系注意力，不复制完整 GCN、DECA-GCN、Matrix-Fisher 或超大 ViT 姿态头，保持可训练和可部署。

### 14.2 与已有工作的区别

| 已有方向 | 本路线的区别 |
|---|---|
| 单纯替换 ConvNeXt/ViT 主干 | 主干只是基础，主要增量在 CAD 条件解码和结构损失 |
| StructGCN 的 GCN/KSN/DECA-GCN | 不建立完整图卷积姿态回归器，关系只服务于 heatmap/depth 解码 |
| PVSPE 的 PVT、Gaussian/Matrix-Fisher pose | 保留确定性的 heatmap-PnP 接口，不引入完整概率姿态表示 |
| ConvMamSPN 的 Mamba 与几何伪标签 | 不复制 Mamba 混合器和其伪标签流程 |
| 低置信度二次推理 | 二次 crop 只作为后续工程实验，不作为第一版创新 |

论文中应把方法描述为**CAD 条件的关键点关系解码方法**，不能把 ConvNeXt-Tiny 本身写成原创，也不能声称首次使用图结构、PnP 或伪标签。

### 14.3 必须提供的证据

- B0 普通 FPN 与 B1 关系解码器的对比；
- B1 与 B2 的 CAD 边损失消融；
- coarse-only、fine-only 和 coarse-to-fine 的对比；
- Lightbox、Sunlamp 的平均误差与 P95/P99；
- 关系图边数量和构造方式的敏感性；
- 参数量、FLOPs、峰值显存、平均延迟和 P95 延迟；
- 结构模块在伪标签覆盖率固定时是否仍然有效。

### 14.4 数据增强带来的候选扩展（2026-11-04，部分实现）

代码已实现**几何支撑约束的局部退化采样**：仅在有真值的 synthetic 源域使用真实位姿、相机内参和 11 个 CAD 点计算尺度归一化投影敏感度，再从有效点命中且仍保留最低几何支撑的局部 patch 候选中抽样。跨视图 CAD 关系一致性仍属后续研究。增强掩膜只描述人为退化影响，不覆盖现有边界有效性标签，也不自动删除遮挡点的完整监督。

研究差异应落在**最终位姿解算的几何支撑如何控制增强**，并验证关系解码器能否从中受益。部件级风格、几何保持增强、困难增强和普通一致性已有先例；投影雅可比的局部信息指标也不能保证全局 PnP 唯一或伪姿态正确。当前采样器尚无本项目实验收益，不应写成已验证增益或首创结论。完整指标定义、实现边界与最小对照见本路线服务器指南第 9 节。

## 15. 本地开发与单卡 A6000 训练部署

### 15.1 硬件假设

服务器为单张 RTX A6000，名义显存 48GB；由于当前可能只有约一半显存可用，第一轮按 24GB 可用显存设计。训练开始前执行：

~~~bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())"
~~~

不要目据显卡型号直接固定 batch size，必须记录真实的 free、reserved 和 peak allocated memory。

### 15.2 本地开发与服务器训练的分工

本地 Windows 只负责：

1. 编写 ConvNeXt、FPN、CAD relation decoder 和 loss；
2. 用 dummy input 验证 shape、梯度和 checkpoint；
3. 用少量真实文件验证数据读取、去畸变、BPnP 和评分；
4. 检查伪标签 JSON、CAD 图和配置；
5. 运行 lint、import 和单 batch smoke test；
6. 提交 Git commit。

A6000 服务器负责：

1. 完整 source-only 训练；
2. 风格/传感器增强训练；
3. 目标域伪标签和三轮自训练；
4. 消融、多 seed 和部署成本统计。

本地不需要为了跑完整训练而强行压缩模型结构。

### 15.3 当前环境文件的处理

路线目录已提供 `server/env_a6000.yml`，配置 Python 3.10、PyTorch 2.1、CUDA 11.8 和兼容版本的 timm/Kornia。上传服务器时：

- 使用路线目录的 A6000 环境文件；
- 将配置中的占位符替换为服务器绝对路径；
- 安装后记录 Python、PyTorch、CUDA、timm、Kornia 和 OpenCV 版本；
- 确认 timm 能加载 ConvNeXt-Tiny 权重，并重新做 B0 shape test；
- 只有 B0 通过后才启动训练。

后续可以再加入 runtime 检查脚本和小批次 smoke test 工具；它们不是当前训练闭环的前置依赖。

### 15.4 代码上传方式

优先使用 Git：

~~~text
本地：
  git checkout -b codex/route-b-a6000
  git add models src configs diagnostics
  git commit -m "add ConvNeXt CARD pose route"
  git push origin codex/route-b-a6000

A6000：
  git clone 或 git pull
  conda env create -f server/env_a6000.yml
  设置 DATA_ROOT 和 OUTPUT_ROOT
~~~

没有远程仓库时，使用 git bundle 或压缩包。数据集、checkpoint、临时 PDF 和训练结果使用服务器独立目录，不随代码反复上传。

### 15.5 约 24GB 可用显存的起始配置

~~~text
输入：512×512
batch_size：1
AMP：开启
gradient_accumulation_steps：2–4
decoder_channels：96 或 128
backbone：source-only 前 3 epoch 冻结整个 encoder
num_workers：2
~~~

若单 batch 峰值和训练稳定性允许，再逐步扩大：

~~~text
输入 640×640
batch_size 1
decoder_channels 128
按 source validation 判断是否解冻 encoder
~~~

若实际可用接近完整 48GB，可进一步尝试 640/768 输入、batch 4、更多增强和多 seed，但要保持官方评分协议不变。

### 15.6 A6000 上可恢复的较大模型实验

之前因本地硬件保守而没有放入首轮的实验，现在可以作为后续分支恢复：

- ViTPose-S/16，输入 448 或 512；
- PVTb1/PVTb2 风格多尺度 Transformer；
- ConvNeXt-Tiny 更宽的 decoder；
- 640/768 输入和 multi-scale test；
- DINOv3-ViT-L 的冻结 backbone 小规模 pilot。

这些实验仍不应排在 ConvNeXt-CARD 主结果之前。DINOv3-ViT-L 即使能在 A6000 上运行，也需要 gradient checkpoint、冻结大部分 block 和更长训练时间，适合做扩展对照，不适合作为快速论文主线。

### 15.7 服务器验收顺序

~~~text
1. CUDA、timm、BPnP 和环境检查
2. 10 张图单 batch 前向
3. 单 batch 反向和 checkpoint 重载
4. 100 张图 CAD graph 与 BPnP 诊断
5. 1 个 epoch source-only
6. 完整 source-only checkpoint
7. 结构模块 B0–B2 消融
8. 目标域伪标签第 1 轮
9. 三轮自训练
10. 官方评分、P95 和部署成本统计
~~~

每一步保存配置、日志、Git commit、显存峰值、checkpoint 和输出目录。单卡服务器不需要引入 DDP 或多机训练代码。
