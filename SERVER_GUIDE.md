# 路线 B：ConvNeXt-CARD 服务器与算法指南

ConvNeXt-CARD、SPEED+ loader、BPnP、共用训练/评测模块、伪标签流程和 CSV 导出代码均已包含在本目录。上传本目录即可作为代码根目录运行。SPEED+ 数据、PyTorch/CUDA 环境和可选 timm 预训练权重需另外提供。完整执行顺序见 [RUNBOOK.md](RUNBOOK.md)。

## 1. 上传目录和预检

上传后将本目录设为 `CODE_ROOT` 并从该目录执行脚本。数据根目录通过配置脚本填写；目标域、CUDA 和文件完整性用 `server/preflight.py` 检查：

```bash
python server/configure_server.py --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python server/preflight.py --data-root "$DATA_ROOT"
```

请以 RUNBOOK 为运行命令的唯一操作指南；本文件后续章节记录方法审查、姿态解算和数据增强建议。

## 7. 姿态解算和后端优化

### 7.1 当前代码实际采用的位姿算法

路线 B 的“网络训练几何优化”和“推理时位姿解算”是两条不同链路。

**网络训练。** ConvNeXt-Tiny/FPN 产生 coarse/fine 热图和深度图；fine 热图经 soft-argmax 得到 2D 关键点。训练器以 BPnP 求出相机位姿，再将 CAD 3D 点投影回图像，优化 `projected-vs-GT` 与 `projected-vs-prediction` 两项 MSE。附加损失包括关键点热图、camera-space 深度、coarse/fine 坐标一致性和 CAD 图边长一致性；AdamW 更新网络参数。`BPnP_fast` 的前向使用 OpenCV `solvePnPRansac`（ITERATIVE）生成初值，再用 ITERATIVE `solvePnP` 求解；自定义反向通过隐式雅可比把位姿梯度传回 2D 关键点、CAD 点和相机内参。它不是可学习的优化器，也不是对网络参数直接运行 PnP；它让 PnP 几何约束能够参与网络训练。

**伪标签。** `generate_pseudo_labels.py` 调用共享 `pose_utils.generate_json_loop()` 的 `staged_quality_fallback`，先检查两个 top-7 分支是否都通过严格质量门，再尝试 top-9 扩展；从第 1 轮起允许合格的单分支 top-7 结果。扩展和单分支仍需通过深度、内点、选中点及全点重投影门限；平均点融合默认禁用，直接 PnP 不进入伪标签链路。两级来自同一个网络，双分支通过不代表独立模型共识，也没有额外的相对旋转/平移一致性硬门限。

**目标评估。** `evaluate_target.py` 使用另一条共享求解链：在初始 RANSAC 像素门限下比较各 top-k 与分支，全部失败才逐级放宽 RANSAC 门限，之后尝试融合和直接 PnP。这里的有限数值与正深度检查用于求解有效性，不等于伪标签严格质量门。`pose_refine_lm=true` 时，两条链均可对 RANSAC 内点做带回滚的 LM，默认关闭；直接 PnP 没有 RANSAC 内点，不能冒充内点证据，也不会执行内点 LM。训练中的 CAD edge loss 没有作为推理时的在线优化目标。

所以路线 B 已把 PnP 几何用于训练，默认推理采用“RANSAC/直接 PnP + 候选筛选”，另外提供可选的内点 LM。热图 coarse-to-fine 和 CAD 关系解码本身不等于后端姿态精修。

### 7.2 最适合路线 B 的首轮后端

**先在 fine 点上做 RANSAC 初始化和 LM 精修。** 以当前共享 EPNP-RANSAC 候选作为初值，在其内点上调用 `solvePnPRefineLM`，并把精修后的 pose 交给现有全点/正深度/官方指标检查。LM 精修要对 coarse 和 fine 两级分别记录结果；主结果优先 fine，因为它是最终高分辨率 heatmap。仅当 validation 显示 coarse 初值对小目标更稳，再考虑 coarse 初值到 fine 精修的两阶段组合。OpenCV 还提供 VVS；其旋转增量通过 SO(3) 指数映射更新，适合作为一个独立后端对照，但不是首轮必需项。

这比先更换主干或另写复杂 BA 更适合快速形成结果：不改变 ConvNeXt-CARD 网络、不动现有训练损失和 BPnP 接口，只增加推理端后处理，可用同一 checkpoint 做前后配对比较。OpenCV 另有 VVS 精修，可作为单项 solver ablation；第一版建议只新增 LM，保持变量可控。

### 7.3 结构化的不确定性加权

路线 B 已经输出 fine heatmap，所以不必立刻添加一套 MC-dropout 或多模型 ensemble。可从 fine heatmap 的局部 softmax 分布估计每个点的二维均值和协方差，先校准 logits 温度，再使用协方差迹/熵作为伪标签置信度；稳定后再把协方差逆用于加权 robust LM。目标函数可写成：

~~~text
min_(R,t) Σ_i ρ( e_i(R,t)^T Σ_i^(-1) e_i(R,t) )
~~~

其中 `e_i` 是第 i 个关键点的 2D 重投影残差，`Σ_i` 是其像素定位协方差，`ρ` 用 Huber 限制异常点影响。权重需设置下限/上限并在 synthetic validation 校准；热图过平、双峰或数值退化时应走当前无权重 RANSAC fallback。2024 年 IEEE TAES 的 SPEED 工作已研究以各关键点空间不确定性加权 PnP，并用语义熵做单帧自评，证明这条增量与本路线任务形态相符。

若要更强且更可发表的版本，可在 fine head 旁增加 `log_sigma_x, log_sigma_y` 和相关角的预测头，以 Gaussian NLL 监督 aleatoric 不确定性；这会改模型和 loss，建议排在热图矩估计之后。对于项目当前追求的快出结果，先做“热图协方差诊断 + LM”即可。

### 7.4 可保留但不作为第一优先级的方向

- **SA-LMPE 式外点退火：** 2019 KPEC 获胜方法在 RANSAC PnP 初值后使用 Huber + LM，并逐轮收紧残差阈值、剔除离群关键点。可以作为 LM 后的扩展，但阈值必须用 SPEED+ synthetic validation 重新标定。
- **替代 PnP 初始化器：** 可固定验证集比较 EPNP、ITERATIVE、SQPnP。不要把求解器堆叠成一串按测试真值挑选的回退；SQPnP 不会替代 RANSAC 外点过滤或不确定性加权。
- **置信度触发的第二次 crop/inference：** 2025 年 Acta Astronautica 的 SPEED 研究探索了按关键点置信度拒绝外点，并判断是否值得进行第二次推理。路线 B 可在第一版 LM 与不确定性评估通过后再实现；它涉及额外延迟，且不是纯姿态优化。
- **严重截断场景的不确定性感知 PnP：** Liu 等人 2026 年的全文采用 Swin-Transformer U-Net 预测掩码、逐像素物体坐标和 NIG 证据参数；将 3D 坐标不确定性经投影雅可比传播为 2D 残差协方差，以 Mahalanobis 重投影误差运行 LM，并以局部 Hessian 近似位姿协方差。该方法面向超近距离严重截断，并在独立 STRD 数据集验证；它不是现有 11 点 heatmap 网络可直接替换的 PnP 插件。论文还报告对称几何/极近距离仍会出现单帧歧义，位姿协方差也建立在局部近似假设上。
- **深度辅助姿态优化：** 目前 depth head 只参加训练监督。先把预测深度用于 pose quality consistency 诊断，不建议直接和 2D PnP 残差混成联合优化，以免未标定的深度偏差污染姿态。

### 7.5 将 2026 年方法迁移到当前关键点代码的建议

这篇全文**强化了不确定性感知 PnP 的研究依据，但不改变路线 B 首轮先加 RANSAC 内点 LM 的优先级**。全文方法依赖逐像素 3D 坐标图和对应的不确定性图来应对固定关键点被截出画面的问题；路线 B 当前输出的是稀疏 CAD 关键点热图，因此不应把 NIG 坐标回归或完整稠密管线写成现有实现，也不必为快速得到 SPEED+ 主结果先重做网络。

建议后端按三步实现：先以现有 RANSAC 候选和内点运行普通 `solvePnPRefineLM`；再从 fine heatmap 局部分布估计每个关键点的二维均值/协方差，在合成验证集校准尺度后，用其逆协方差白化重投影残差并做 Huber/IRLS LM；最后可把加权雅可比的 `H⁻¹` 作为局部位姿不确定性诊断量，辅助伪标签接受/拒绝。低置信或双峰热图、协方差病态、优化恶化时回退现有无权重 RANSAC 结果。这里的热图协方差不是 NIG 分解出的 epistemic 与 aleatoric 不确定性；若之后增加显式不确定性头，需用证据回归损失训练并单独检查校准，不能只因输出了方差就称为可靠。

校准和验收只能用合成训练/验证拆分，报告覆盖率、误差分箱、伪标签误收/误拒和官方姿态分数。近似 `H⁻¹` 可能因外点、模型偏差、对称结构及几何退化而失准。若目标扩展到论文中的严重截断场景，再另建裁剪压力测试并评估稠密对应；这会改变模型输出、监督标签和训练流程，属于后续研究分支。论文所用 STRD 的独立实验结果不能直接等同于 SPEED+ Lightbox/Sunlamp 上的收益。

### 7.6 实验验收和参考依据

用完全相同的 checkpoint 和输入，依次比较现有 fallback、fallback+LM、fallback+LM+热图协方差。至少报告 SPEED+ 官方分数、旋转/平移误差、PnP 成功率、全点重投影、内点数、正深度比例、伪标签覆盖率、平均/P95 延迟。重投影变小不是采用新后端的充分条件；若官方位姿指标下降，应回退。

- Chen et al., “Satellite Pose Estimation with Deep Landmark Regression and Nonlinear Pose Refinement,” ICCVW 2019. [CVF 全文 PDF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Chen_Satellite_Pose_Estimation_with_Deep_Landmark_Regression_and_Nonlinear_Pose_ICCVW_2019_paper.pdf)
- Wang et al., “Monocular Satellite Pose Estimation Based on Uncertainty Estimation and Self-Assessment,” IEEE Transactions on Aerospace and Electronic Systems, 2024. DOI: `10.1109/TAES.2024.3441569`.
- Bechini and Lavagna, “Robust and efficient single-CNN-based spacecraft relative pose estimation from monocular images,” Acta Astronautica, 2025. DOI: `10.1016/j.actaastro.2025.04.016`.
- Liu, Yu, Liu, Han, and Liao, “Truncation-robust pose estimation for non-cooperative spacecraft using dense correspondence and uncertainty estimation,” Acta Astronautica, 246 (2026), 570–583. DOI: `10.1016/j.actaastro.2026.04.033`.
- Chen et al., “End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization,” CVPR 2020（BPnP 的相关原始工作）。[CVF 论文页](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_End-to-End_Learnable_Geometric_Vision_by_Backpropagating_PnP_Optimization_CVPR_2020_paper.html)
- OpenCV calib3d 文档：[`solvePnPRefineLM`、`solvePnPRefineVVS` 与 PnP API](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。

普通内点 LM 已接入，默认 `pose_refine_lm=false`；退火外点剔除和显式不确定性加权尚未实现。路线 B 仍未在此 Windows 主机或 A6000 上完成 GPU 训练验证；首次服务器验收顺序见第 6 节。

## 8. 2026-11-04 结构、回退与质量门审查后的运行约定

### 8.1 保留严格质量门，缩短到实际使用的日程

ConvNeXt-CARD 的 coarse/fine 关键点进入 EPNP-RANSAC 后，会由预测位姿重新投影生成目标域监督。因此应保留严格质量门：成功求出一个位姿并不能说明伪标签可以用于下一轮训练。两个输出共享网络，双分支成功也不构成独立模型共识。目前没有额外的分支间旋转/平移一致性硬门限，不能把 `strict` 解释成“两级姿态已严格相同”。

原代码存在两个重要问题：双分支 PnP 成功时，即使后续 `quality_passed=false` 也会放行；expanded/single/fused 原定第 9/12/14 轮启用，但默认 `--niters 4` 仅生成 0/1/2 轮伪标签，实际一次都触发不了。前者已修正为两个分支都必须通过严格检查，后者已改成以下显式起点：

| 伪标签轮次 | 允许的策略，按顺序执行 | 最低内点支持 |
| --- | --- | --- |
| niter_0000 | 两分支 top-7 严格检查；失败后尝试各分支 top-9 | strict 每分支至少 5/7；expanded 候选至少 7/9 |
| niter_0001、0002 | strict → expanded → 单分支 top-7 | single 至少 6/7，其余同上 |
| 默认全部轮次 | 不启用平均关键点融合，不使用直接 PnP 生成标签 | `pseudo_fused_start_round=null` |

对应配置是 `pseudo_expanded_start_round=0`、`pseudo_single_start_round=1`、`pseudo_fused_start_round=null`。这是与三轮预算相符的工程日程，尚未被实验证明优于所有轮次 strict-only。扩展候选和单分支均不能跳过同一组残差与深度质量门；对通过门限的候选按几何残差和支持数排序，原始热图峰值只作为最后的并列项。旧共享调用不带新字段时仍保留 9/12/14 的兼容默认值，因此自建配置必须显式带上新字段。

### 8.2 参数是否合理

| 参数 | 512² 网络输入下的起点 | 审查判断 |
| --- | --- | --- |
| `pseudo_reprojection_error` | 2 px | RANSAC 内点半径，已显式配置，避免依赖路径名称的历史启发式 |
| `pseudo_max_selected_mean_reprojection` | 1.5 px | 所有选中点的均值，包含 RANSAC 判为外点的点；比 2 px 内点半径严格是有意的高纯度要求，可能降低覆盖率 |
| `pseudo_max_selected_reprojection` | 3 px | 任一选中点的残差上限，防止均值掩盖单点偏差 |
| `pseudo_max_full_mean_reprojection` | 8 px | 全 11 点均值上限，用来约束未选点的几何一致性 |
| `pseudo_max_full_reprojection` | 20 px | 全点最大值，容许少量弱点偏差，但不能让其无限大 |
| `pseudo_min_depth / pseudo_max_depth` | 0.1 / 20 m | 全 CAD 点相机深度范围，属于物理检查，不能直接判断真实姿态误差 |
| strict / expanded / single | 5/7、7/9、6/7 | 扩展与单分支都提高支持比例，作为起点有逻辑依据；并非论文给出的通用最优值 |

以上数值可用于首轮，不宜为增加样本数统一放松。先在固定 synthetic validation 及预先定义的曝光、噪声、模糊压力集上，查看保留率、真实旋转/平移误差和错误放行率，再一次只调一个门限。目标域真实标签留作最终评估；无标签目标诊断只能揭示拒绝原因和数值分布，不能直接给出伪标签真实准确率。

新增 `pose_threshold_reference_size=[512,512]`。由 512² 改为 384² 时，上述像素门限统一乘 0.75，深度及内点数不变。原图 1920×1200 到 512² 的两轴相机缩放仍由现有输入管线完成；这里的缩放指相对参考网络输入的后续变化。若新输入改变参考纵横比，当前函数报错，必须先恢复参考坐标再使用单个像素半径。B 的 `rows`、`cols` 和预生成热图尺寸也应同步更新。

RANSAC 的 `confidence=0.99` 是算法期望采样成功概率，既不是热图可信度，也不是当前伪标签有 99% 概率正确。当前默认迭代预算和内点比例仍会影响实际成功率。OpenCV 对恰好 4/5 点可能直接求解并将全部点返回为内点；当前默认 7/9 点避开了这一精确最小集条件，自定义小子集时不能只看内点率。

### 8.3 最终评估的多级回退应如何理解

评估要尽量覆盖所有比赛图片，仍保留短链：在初始像素门限下比较 top-7/8/9/10/11 与 coarse/fine → 全失败才逐级增加 RANSAC 半径 → 融合 → 直接 PnP。模板使用 `pose_eval_search="all_topks"`，排序依据是全点均值、选中点均值、内点数，再看峰值；可切回 `first_valid` 做固定 checkpoint 的延迟与精度对照。

这条链只检查有限数值、正深度和对应点支持，没有复用上述严格伪标签门限。最终救援得到的 pose 可以用于提交，但不能因此回流训练。直接 PnP 的 `inlier_count=0`、`ransac_verified=false`，`support_count` 表示使用点数。`*_solver_diagnostics.jsonl` 保存求解策略、残差、支持和精修状态，metrics 同时统计策略频次；有任何缺失样本时 `official_valid=false`，solved-only 分数只是诊断。

多候选搜索只能缓解当前某个子集求解失败；相同网络的错误对应或对称歧义可能同时骗过全部候选。融合默认仅留在最终评估救援中；平均两个不同错误模式的坐标没有理论质量保证。新增搜索是否值得采用，要看固定验证集的官方位姿误差和耗时，不能只看求解成功率。

### 8.4 已经落实的指南建议与尚未实现的内容

- **已实现可选内点 LM，默认关闭。** 位于共享 `src/pose_refinement.py`，在各分支 RANSAC 后执行。正确映射局部内点到 CAD 编号，复制 OpenCV 初值；异常、深度越界、支持减少、内点 RMSE、选中点均值/最大值、全点均值/最大值或留出点均值恶化时保留原解。直接 PnP 无内点不精修。该保守守卫可能舍弃真实更好的 pose，因此仍要用真值验证是否有实际收益。
- **已实现预处理参数固定、像素门限缩放、几何候选排序和诊断补充。** 三个模板显式记录 ImageNet RGB 均值/标准差，未改变原有 B 的归一化行为。原始热图峰值没有校准为概率，不再作为候选几何质量的主要排序依据。
- **已实现完整 UDA 调度的目标缓存互斥。** 同 DATA_ROOT、同域的两个受保护 `main_loop.py` 不能同时占用 `lightbox_train` 或 `sunlamp_train`。锁位于数据根目录，持有到该实验循环结束；独立 `create_maps.py`、准备命令及旧根目录调度器不在保护范围内。中断遗留锁只有确认无训练进程使用缓存后才能手动删除。
- **尚未实现** source replay、EMA teacher、FIFO、在线 TTA、协方差加权 LM、外点退火、新的像素域噪声/曝光调度和 letterbox。当前 `quality_weight` 不进入 loss，`use_style_augmentation` 也没有接入；不应仅设置这些字段就声称启用了相关方法。FDA 仍是已有离线数据脚本，可独立设计对照。

普通 LM、候选搜索和质量门修复属于实现完善。路线 B 的模型研究点仍应围绕 CAD 条件关系解码、跨尺度关键点与几何约束，并靠对比/消融验证；这些基础工程不单独构成新的论文方法。暂不加入 EMA/原型/FIFO，是为了保持当前可检验的训练接口，不是因为本地主机不能训练。

### 8.5 验证范围与服务器使用顺序

本地五个测试文件报告 41 项通过，Python AST 与静态检查通过；`test_cad_data.py` 进程退出时仍有本机原生库段错误，虽然测试报告为通过，须在正式训练环境复跑确认。现有测试覆盖路径/恢复、姿态求解、CAD 数据和热图输入保护；不包含真实 ConvNeXt 前向、完整 BPnP 反向、GPU 显存或目标域精度测试。

服务器上传更新后的路线 B 文件夹后，从 source checkpoint 引导 niter_0000，以新版本重新生成伪标签和热图，按 `--niters 4` 完成三次训练。先保存 LM 关闭的固定基线；然后只打开 `pose_refine_lm=true` 用同 checkpoint 在 synthetic validation 配对比较。训练验证和目标域评估共享 `src.pose_estimation.estimate_pose` 的求解与回退流程；checkpoint 选择采用未阈值化的 SPEED 验证代理分数，最终报告则计算官方阈值分数，因此需同时记录验证代理值与最终官方结果。单张 A6000、约 24GB 可用显存下仍按第 6 节验收，不因本机缺 GPU 而删减模型主线。

审查证据：Bechini and Lavagna, DOI `10.1016/j.actaastro.2025.04.016`，PDF pp. 202–203、206，支持置信度子集、逐步扩点和验证集参数搜索；其 `NkP=4`、`tau_conf=0.88` 来自不同网络及有标签 SPEED 参数搜索，不能移植成当前通用阈值。Feng et al., DOI `10.1109/TAES.2025.3623610`，PDF pp. 453、456，讨论错误伪标签积累，支持严格筛选与一致性，不能直接证明本项目 LM 有效。OpenCV 4.12.0 的 [RANSAC 实现](https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/src/solvepnp.cpp#L255-L294) 与 [API 定义](https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/include/opencv2/calib3d.hpp) 用于核对内点、像素半径和 LM 行为；服务器仍需记录其实际 OpenCV 版本。

## 9. 2026-11-04 数据增强调研与代码实施裁决

### 9.1 证据取舍与已有工作的约束

本轮以 JAIS、Acta Astronautica、Journal of Spacecraft and Rockets 及 CVPR、ICLR、NeurIPS 等相关全文作为主要方案依据，不据出版商名称推断单篇工作质量，也不添加未经查询的 JCR 分区。MDPI 论文保留为方法重合核对，不能用期刊取舍排除已经存在的先例。本节独立定义 ConvNeXt-CARD 的增强设计、资源约束和实施顺序。

| 文献 | 已核实的内容 | 对 ConvNeXt-CARD 的意义 |
|---|---|---|
| Legrand et al., **Domain Generalization for In-Orbit 6D Pose Estimation**, JAIS, 2025；DOI `10.2514/1.I011498`；作者版 `arXiv:2406.11743` | 已读全文；SPEED+ 中随机组合局部曝光、纹理/Fourier、Hide&Seek、噪声等，提供分项消融 | 主要领域依据；先建立较强增强基线，再评估结构解码器的增量 |
| Bechini and Lavagna, **Robust and efficient single-CNN-based spacecraft relative pose estimation from monocular images**, Acta Astronautica, 2025；DOI `10.1016/j.actaastro.2025.04.016` | 已读用户提供全文；传感器噪声、亮区 flare/blooming 与风格随机化 | 支持受控成像退化；这些算子本身不是本路线贡献 |
| Park and D'Amico, **Bridging the Domain Gap for Flight-Ready Spaceborne Vision**, Journal of Spacecraft and Rockets, 63(1), 2026；DOI `10.2514/1.A36130` | 已读用户提供全文；增强、预训练和分辨率对跨域性能有重要影响 | 固定这些因素，防止将基础训练设置的收益误归于 CAD 关系模块 |
| Hendrycks et al., **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty**, ICLR 2020；`arXiv:1912.02781` | 已读全文；随机增强链混合和多视图一致性 | 可做只含光度操作的变体；不能直接混合不同几何视图而沿用一个热图标签 |
| Xu et al., **A Fourier-based Framework for Domain Generalization**, CVPR 2021；`arXiv:2105.11120` | 已读全文；源图像幅度混合、保留相位与 co-teacher | 可单独借鉴幅度混合做 DG 对照；它与使用目标风格的 FDA 数据协议不同 |
| Zhou et al., **Domain Generalization with MixStyle**, ICLR 2021；`arXiv:2104.02008`；Li et al., **Uncertainty Modeling for Out-of-Distribution Generalization**, ICLR 2022；`arXiv:2202.03958` | 已读全文；批内伙伴和批间统计决定风格混合/扰动 | batch=1 时不能直接照搬；也不能靠梯度累积修复 |
| **Adversarial Semantic Data Augmentation for Human Pose Estimation**, ECCV 2020；`arXiv:2008.00697`；**AugMax: Adversarial Composition of Random Augmentations for Robust Training**, NeurIPS 2021；`arXiv:2110.13771` | 已读全文；语义区域、困难增强与对抗搜索已有方法 | 候选创新必须超出“CAD 区域加扰动”或“按任务 loss 选最难增强” |

最接近的补充工作必须保留其访问状态与边界：

- **PVSAR**，Aerospace 2024，DOI `10.3390/aerospace11110948`：摘要和作者代码已核实可见面局部亮度变换、原关键点标签保持；正文受 HTTP 403 阻挡，未读全文。
- **A Unified Framework for Cross-Domain Space Drone Pose Estimation Integrating Offline Domain Generalization with Online Domain Adaptation**，Drones 2025，DOI `10.3390/drones9110774`：摘要确认 offline DG、DSU 与 online adaptation；正文受 HTTP 403 阻挡，具体增强细节未核实。
- **Topological-Projection-Consistent Keypoint Extraction for PnP-Based Space-Target Pose Measurement Under Extreme Illumination**，Sensors 2026，DOI `10.3390/s26196079`：已读公开全文；代理几何光照、清晰/退化结构蒸馏和加权 PnP 已有先例，但实验使用自建实验室目标，不能作为 SPEED+ 同协议分数。
- **Component-Aware Structure-Preserving Style Transfer for Satellite Visual Sim2Real Data Construction**，`arXiv:2605.19624v3`，2026：已读最新版全文；部件掩膜与结构保持风格迁移已有先例；尚不能标成已核实的正式期刊论文。

暂不以补下载两篇 MDPI 正文作为代码实施前提；如要在论文中主张新的表面级增强，再补全文逐项比较。没有看到相同表述不足以证明首创，局部几何信息矩阵也不是本项目发明的数学工具。

### 9.2 当前实现与适配入口

| 位置 | 当前状态 | 需要明确的边界 |
|---|---|---|
| `src/input_transforms.py` + `train.py` | BGR 转 RGB、resize；启用新增强后去畸变→像素域曝光/gamma/对比度/blur/noise→ImageNet Normalize | 验证、伪标签不加随机增强；`enabled=false` 回退旧增强顺序 |
| 根目录共享 loader | 图像和 11 通道热图分别经过同一 `transform_input` | 新像素扰动必须移到图像专用接口，不能污染热图 |
| `train.py` | 原有损失外，synthetic 样本可用 source-only CAD 支撑约束采局部 patch | 已有局部采样，不代表跨外观关系损失；coarse/fine 同图一致性不是双视图训练 |
| `models/poseconvnext.py` | encoder → FPN → relation decoder，返回 `relation_tokens` | 特征扰动有插入点，但原版 MixStyle/DSU 仍受真实 batch 大小限制 |
| 配置 | `batch_size=1`，支持 AMP/梯度累积；`image_augmentation` 为实际使用项 | `use_style_augmentation` 仍未消费；梯度累积不产生批间统计 |
| `quality_weight` 与 `visible_kpts` | 前者不进入训练 loss；后者为边界有效性启发式 | 不能将其当作现成的样本加权训练与真实遮挡标注 |

当前新增图像路径为：BGR→RGB/resize → 与标签一致的去畸变 → `[0,1]` RGB 外观扰动 → ImageNet Normalize → ConvNeXt-CARD。关闭新增强时保留旧行为；验证和伪标签推理使用确定性预处理。基于投影点生成局部 patch 候选使用去畸变后的模型坐标，而非原图坐标。

### 9.3 首轮基础增强

当前实现独立开关的曝光/gain、gamma、对比度、随机强度噪声与高斯模糊；局部 patch 为矩形、内部 darken/lighten 插值，概率由配置控制。模板概率为 1.0；可降低该值保留未经全局外观扰动的样本。模糊/噪声/局部退化各有独立概率，不应将其解释为已经实现的多算子随机策略或软边缘阴影。

纹理/频域增强另做单项。FACT 幅度混合可从源数据额外读取伙伴图像或使用缓存，适配 batch=1；FDA 使用目标风格池，必须记录目标图像使用协议。保留相位不能保证每个细小边缘的可定位性不变，不能以 FID 改善代替位姿验证。FDA 的元数据默认不复制，生成子集必须同步图像与标签清单；现有 `beta=0` 仍修改低频，不是严格原图对照。

原版 MixStyle、DSU 与 ALOFT 不作为第一批即插即用模块。ALOFT（**ALOFT: A Lightweight MLP-like Architecture with Dynamic Low-frequency Transform for Domain Generalization**, CVPR 2023；`arXiv:2303.11674`，已读全文）结合 GFNet 频域模块，扰动统计同样涉及批间方差，不能直接当成输入 FDA。若后续做特征扰动，需先验证真实 batch≥2，或明确实现跨步伙伴/统计队列；队列本身不是新的论文核心。

### 9.4 路线 B 的候选创新：几何支撑约束的局部退化采样

研究问题：同样面积与强度的光照/遮挡扰动，破坏不同关键点组合时，是否对最终位姿造成不同风险？能否借助 CAD 和相机几何，选择有挑战且仍保留足够几何支撑的训练图像？**首版源域采样器已实现，但其阈值和下游增益尚未实验验证。**

第一版只对有真实位姿的 synthetic 图像启用；当前代码已实现全图光度增强和有限的矩形局部退化采样，不需要 GAN、CAD 表面重渲染或目标域真值：

1. 用真实位姿、`kpts.mat` 和缩放后的相机矩阵，将 11 个 CAD 点投影到模型坐标系。
2. 设目标特征长度为 `L`，对局部位姿增量使用 `eta = [delta_rotation, delta_translation/L]`；投影坐标分别除以图像宽高。计算 `J_i = d normalized_projection_i / d eta`，避免单位和分辨率主导条件指标。
3. 当前实现随机生成 4 个矩形候选，以投影关键点邻域与 patch 的相交作为二值 `w_i` 代理。矩形内部对图像做局部 darken/lighten；它不是连续软阴影，也不是真实可见性、预测协方差或已校准的 Fisher 信息。
4. 当前形成 `H_aug = sum_i(w_i * J_i.T @ J_i)` 并以 `H_clean` 白化后的最小广义特征值比评估保留支撑；清晰矩阵秩不足时退回全局光度增强，不用阻尼伪造秩。阈值尚需在 synthetic validation 校准。
5. 光度增强不改变关键点坐标，所以 `J_i` 本身不因调亮变更，变化来自 `w_i`。若清晰视图已经几何退化，回到弱增强，不用接近零的特征值比值作决策，也不把该样本从全部训练中删除。
6. 当前先剔除不影响任何有效图像内关键点邻域的背景候选，再在剩余候选中均匀随机抽取；几何筛选启用时还需满足支撑下限。若没有候选同时命中有效点并满足支撑，就不加局部 patch，仅保留全图光度变化。日志记录受影响有效点、背景-only 候选数和无命中样本数。

当前实现对每张图的少量候选做无梯度 6×6 矩阵计算，尚未缓存雅可比；不需对每个候选额外运行网络或多步对抗优化。稀疏点邻域只作影响区域代理；没有表面网格/掩膜时，不能称其为精确的材质或表面光照模拟。

**理论边界。** 这是给定线性化位姿和可靠性假设下的局部几何代理，不保证 PnP 全局唯一、外点消失或图像仍可识别。目标域若以后使用伪姿态计算同类量，必须来自冻结的弱视图预测并停止梯度；它仍可能因共同错误而误判，不能借用目标真实位姿。

与已有工作的候选差异，是用最终位姿解算的几何支撑选择增强，而不是仅保持像素投影、按 CAD 面随机调亮或按任务损失寻找最困难样本。是否构成足够方法增量，仍需全文比较与实验，不能宣称首次使用可观测性或信息矩阵。

### 9.5 与 CAD 关系解码器结合的最小一致性设计

在上述采样有收益后，再比较同几何弱/强视图的 fine 关键点坐标或投影 CAD 边向量。一致性项首先定义在可解释的几何输出上，不同时加入热图、token、边长和全局原型等多种新损失。`relation_tokens` 是可用接口，不代表强制 token 完全相同就有收益。

弱视图输出停止梯度，强视图用于训练；这不是 EMA teacher。若只有一部分样本使用配对视图，可从约四分之一训练样本开始校准，逐视图执行并记录真实显存/时间。原有 coarse/fine 同图一致性仍单独保留，不能与跨外观项混名。

图像只改变光度或遮挡时，原位姿、热图坐标和深度真值不变。新增 `observation_mask` 用于描述退化区域，独立于原 `visible_kpts`；如筛选 teacher 一致性，边的两个端点都需符合所定义的可靠条件。源域完整监督可以继续学习遮挡点补全；不得自动用观察掩膜删除所有 BPnP、CAD 与深度监督。全无有效点时必须有明确零损失/跳过分支，不能产生空均值或 NaN。

目标域仍先走既有严格伪标签与热图循环。新增强不能给同一图像生成多套互相覆盖的缓存标签；共享热图保持原标签坐标，视图及掩膜在训练内产生。`quality_weight` 尚未进入 loss，若需要软加权必须连同数据读取、损失归一化与日志一起实现，不能只增加 JSON 字段。

### 9.6 最小对照与服务器安排

| 实验 | 保持固定 | 新增因素 |
|---|---|---|
| B-aug0 | ConvNeXt-CARD、输入大小、损失、训练预算与后端固定 | `image_augmentation.enabled=false`，保留旧输入路径 |
| B-aug1 | 同一模型与预算 | 新像素增强开启、局部 patch 概率设 0；使用独立输出目录 |
| B-aug2 | 相同增强类别和候选预算 | 局部 patch 开启，`local.geometry.enabled=false`；使用独立输出目录 |
| B-aug3 | 相同类别、强度分布与候选生成预算 | `local.geometry.enabled=true`；这是当前配置，报告实际点命中与背景-only 比例 |
| B-aug4 | 固定 B-aug3 的采样策略 | 增加一种跨外观几何关系一致性 |

先完成基线与基础增强，做小规模随机/几何筛选对照，再决定完整训练 B-aug3/4；不用一次跑满所有组合。每个消融使用独立配置及 `path_results`，避免轮次 checkpoint 覆盖。若采样后的区域或强度分布改变，必须报告，必要时做匹配分布对照，避免把收益全归于信息指标。若声称关系解码器与增强存在互补，至少补普通 FPN 在相同增强下的对照。

强度和信息阈值仅用 synthetic validation 及受控退化版本设定；目标域真值不进入增强、伪标签或参数选择。固定设置后报告 Lightbox/Sunlamp 分域官方分数、旋转/平移误差、完整覆盖失败情况、P95/P99、UDA 覆盖率、平均/P95 耗时及显存峰值。使用目标风格池或全目标域自训练时明确 transductive 协议；inductive holdout 图像不得进入这些池。

单卡 A6000 约 24GB 可用显存下，优先采用无训练参数的像素增强和 6×6 几何矩阵计算；配对训练增加额外前向，不能预先保证原 batch 配置仍适用。可用现有 AMP、顺序弱视图推理及降低配对比例控制开销。梯度累积只影响优化，不使 MixStyle/DSU 获得真实批间统计。暂缓 GAN/扩散、NeRF、多步对抗搜索及在线 TTA，是为了控制实验归因和实现成本，与本机是否训练无关。

### 9.7 调研时的实施建议（已落实像素增强与局部采样）

**调研时的建议是不重写 ConvNeXt-CARD；当前已完成以下数据侧改动。** 第一批补图像专用像素增强、可关闭的强度/概率设置、掩膜和可视化记录；第二批在源域实现几何支撑诊断和可关闭采样；有证据后再加入单一跨视图关系损失。不要通过打开占位布尔字段宣称这些模块已可运行，也不因调研而同时替换主干、后端和 UDA 协议。

验收覆盖关闭兼容、随机种子、标签/内参不变、掩膜对齐、几何尺度/退化处理和缓存隔离。像素增强与源域几何支撑采样已接入并通过 CPU 检查；跨视图关系损失未实现。当前环境没有 PyTorch，尚未运行 ConvNeXt、AMP、BPnP 反向或 A6000 训练。

## 10. 代码链路复审与尚待服务器验证的边界

路线 B 的完整链路是：synthetic train/validation 热图与 CAD 图准备 → ConvNeXt-CARD source-only 训练/validation 选模 → 将所选权重放入目标域 `niter_0000` → 按轮生成伪标签、训练 heatmap、执行下一轮 → Lightbox 和 Sunlamp 分别评估 → 生成带域标识的组合 CSV。目标域真实位姿只供冻结后的评估使用；source-only 局部几何支撑采样不读取目标域标签。

本轮修正/确认的边界：Windows spawn 下热图生成模块导入不再读取文件或清空缓存；绝对配置路径稳定规范；UDA 调度器默认拒绝覆盖已存在轮次，并校验前轮检查点和完成标记；已完成训练标记不能被普通训练命令静默删除；B checkpoint 采用完整严格加载，避免未匹配参数仍被随机初始化却看似成功；评估阈值独立于文件名；PnP 失败逐样本跳过，有限位姿还要通过正深度检查。

仍需关注的逻辑边界：

- 当前几何筛选是已知 synthetic 位姿下的线性化支撑代理，不是测得的图像可见性或校准不确定性。`visible_kpts` 只是有效图像区域代理；检查背景-only 候选和无合格几何区域的频率。
- min support ratio 0.35、至少 6 个点、4 个候选和局部退化概率 0.25 都是待校准起点；没有目标域真值参与的理由不能声称已提升泛化。
- `quality_weight` 未参与 loss，`use_style_augmentation` 未接入，跨视图关系损失也没有实现；不能在论文中描述为已运行方法。
- source validation 不是目标端官方 staged fallback；选源 checkpoint 后仍要按相同协议分别报告两个目标域。
- Restart 只处理 UDA 轮次边界，不保存 optimizer 状态；目标缓存共享时须串行。进程意外结束留下的锁只能在确认无相关进程后清理。
- 当前开发环境无 PyTorch，未验证 timm checkpoint、ConvNeXt/AMP 前后向、BPnP 真实梯度、GPU 显存和 A6000 长训练。本轮测试是 CPU 几何/路径/逻辑测试，不是训练验收。

完整命令、模型权重准备、缓存、恢复、指标和提交流程见 [RUNBOOK.md](RUNBOOK.md)。
