# 路线 B：ConvNeXt-CARD 服务器运行手册

本目录本身就是项目根目录。它包含 ConvNeXt-CARD 模型、SPEED+ 公共训练与评测组件、数据加载器、BPnP、热图生成器、目标域伪标签流程和 CSV 导出程序。服务器只需上传本目录；不需要上传仓库根目录或从其它目录复制代码。SPEED+ 数据、训练输出和可选 timm 预训练权重仍需在服务器上单独准备。

## 1. 上传后的目录和数据

例如上传到 `/workspace/route_b_convnext_card`：

```bash
export CODE_ROOT=/workspace/route_b_convnext_card
export DATA_ROOT=/data/SPEED+
export OUTPUT_ROOT=/data/spacecraft-runs
cd "$CODE_ROOT"
```

代码根目录应直接看到 `train.py`、`train_convnext_card.py`、`create_maps.py`、`models/poseconvnext.py`、`BPnP/`、`loaders/`、`src/`、`configs/` 和 `server/`。`src/training_common.py` 提供数据加载、验证指标等共用组件；`train.py` 是命令入口，训练实现位于 `train_convnext_card.py`。

`DATA_ROOT` 至少应有：

```text
camera.json
kpts.mat
synthetic/{images/,train.json,validation.json}
lightbox/{images/,test.json}
sunlamp/{images/,test.json}
```

目标域测试标注用于最终公开评测，伪标签学习只读目标图像，不把测试姿态写入训练标签。训练视图由 `prepare_target_data.py` 创建。

## 2. 创建环境并检查服务器

配置按一张 A6000、512×512、batch size 1、AMP 和梯度累积起步。若作业只分到约 24 GB 显存，先维持配置，不要提高 batch size。

```bash
conda env create -f server/env_a6000.yml
conda activate speedplus-route-b
python server/preflight.py --data-root "$DATA_ROOT"
nvidia-smi
```

预检会导入本路线模型和共享运行模块、检查 CUDA、显存、`camera.json`、11 个 CAD 点以及三个数据 split 的标注和图像文件。暂时没有目标域时可先用 `--source-only`；正式开训前需不带它完整检查。

## 3. 生成服务器配置和 CAD 图

以下命令把具体数据/输出路径写入本路线的 `server_configs/`，不修改模板：

```bash
python server/configure_server.py \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT"
```

首次使用 `pretrained=true` 会让 `timm` 获取 ConvNeXt-Tiny 的 ImageNet 权重，服务器需能访问模型权重源或已有本地缓存。离线时生成配置需加 `--no-pretrained-backbone`，此时 encoder 从随机权重开始，实验条件与预训练版本不同。覆盖之前生成的配置时加 `--force`。

生成固定 11 点 CAD 关系图：

```bash
mkdir -p assets
python prepare_cad_graph.py \
  --kpts "$DATA_ROOT/kpts.mat" \
  --output assets/speed_keypoint_graph.json
```

## 4. 生成合成集热图并训练源模型

```bash
python create_maps.py --cfg server_configs/source.json --num-workers 2
python train.py --cfg server_configs/source.json
```

热图写入 `$DATA_ROOT/synthetic/kptsmap/`。checkpoint 与日志位于：

```text
$OUTPUT_ROOT/route_b_convnext_card/server_configs/source.json/ckpt/
$OUTPUT_ROOT/route_b_convnext_card/server_configs/source.json/logs/
```

选 synthetic validation 最优的 `best_epoch_*.pth` 作为 Lightbox/Sunlamp 两个目标适配过程的第 0 轮 checkpoint。不要使用目标测试标签筛选源模型。

## 5. Lightbox/Sunlamp 目标域适配

先创建各自的目标图像视图：

```bash
python prepare_target_data.py --data-root "$DATA_ROOT" --split lightbox
python prepare_target_data.py --data-root "$DATA_ROOT" --split sunlamp
```

如果文件系统不允许符号链接，加 `--copy-images`。每个域都需要独立初始化并依次运行。下面 `--niters 4` 会训练 0001、0002、0003 三轮：

```bash
SOURCE_CKPT=/absolute/path/to/best_epoch_XX_SCORE.pth

python bootstrap_loop.py \
  --cfg server_configs/lightbox/loop_convnext_card_niter_0000.json \
  --checkpoint "$SOURCE_CKPT"
python main_loop.py \
  --base-config server_configs/lightbox \
  --base-file loop_convnext_card_niter_0000.json \
  --niters 4
```

Sunlamp 使用相应的独立配置：

```bash
python bootstrap_loop.py \
  --cfg server_configs/sunlamp/loop_convnext_card_sunlamp_niter_0000.json \
  --checkpoint "$SOURCE_CKPT"
python main_loop.py \
  --base-config server_configs/sunlamp \
  --base-file loop_convnext_card_sunlamp_niter_0000.json \
  --niters 4
```

调度器每轮运行伪标签推理与质量筛选、格式化标签、重建目标热图，再加载上一轮 checkpoint 训练。出错会停止并保留现场。中断后可通过 `--start-iteration 2 --niters 4` 从指定轮次继续；要重训已有的部分轮次需在对应命令显式加 `--restart-iteration`。先核对该轮配置、缓存和 checkpoint 的目标域一致。

## 6. 目标域评测和 CSV

用最终一轮权重分别评测原始 target split；以下输出路径可以改到任意可写磁盘：

```bash
python evaluate_target.py \
  --cfg server_configs/lightbox/loop_convnext_card_niter_0003.json \
  --checkpoint "$OUTPUT_ROOT/route_b_convnext_lightbox/server_configs/lightbox/loop_convnext_card_niter_0003.json/ckpt/init.pth" \
  --dataset-dir "$DATA_ROOT/lightbox" --dataset-root "$DATA_ROOT" \
  --output-json "$OUTPUT_ROOT/route_b_lightbox_predictions.json" \
  --metrics-json "$OUTPUT_ROOT/route_b_lightbox_metrics.json"

python evaluate_target.py \
  --cfg server_configs/sunlamp/loop_convnext_card_sunlamp_niter_0003.json \
  --checkpoint "$OUTPUT_ROOT/route_b_convnext_sunlamp/server_configs/sunlamp/loop_convnext_card_sunlamp_niter_0003.json/ckpt/init.pth" \
  --dataset-dir "$DATA_ROOT/sunlamp" --dataset-root "$DATA_ROOT" \
  --output-json "$OUTPUT_ROOT/route_b_sunlamp_predictions.json" \
  --metrics-json "$OUTPUT_ROOT/route_b_sunlamp_metrics.json"
```

需要一次比较同一目标域下的多轮 checkpoint 时，批量评分入口会按配置中的 `path_results` 查找 checkpoint，并生成官方有效性排名、失败样本清单和分数曲线。它调用与单模型评测相同的 ConvNeXt 推理入口；`--max-samples` 只适用于调试，部分样本不会计入官方有效成绩。

```bash
python score_checkpoints_official.py \
  --results-dir "$OUTPUT_ROOT/route_b_convnext_lightbox" \
  --config-dir server_configs/lightbox \
  --split lightbox \
  --dataset-dir "$DATA_ROOT/lightbox" --dataset-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/route_b_convnext_lightbox/official_score"
```

Sunlamp 评测时，将上述命令中的 `lightbox` 分别改为 `sunlamp`，并将结果目录改为 `$OUTPUT_ROOT/route_b_convnext_sunlamp`。可通过 `--checkpoint-pattern 'best_epoch_*.pth'` 批量评测各轮中的匹配权重；默认只评测 `init.pth`。

确认两份 metrics 都具有完整覆盖且 `official_valid=true` 后生成比赛 CSV：

```bash
python export_submission.py --domain both \
  --lightbox-predictions "$OUTPUT_ROOT/route_b_lightbox_predictions.json" \
  --lightbox-test-json "$DATA_ROOT/lightbox/test.json" \
  --sunlamp-predictions "$OUTPUT_ROOT/route_b_sunlamp_predictions.json" \
  --sunlamp-test-json "$DATA_ROOT/sunlamp/test.json" \
  --output "$OUTPUT_ROOT/route_b_submission.csv"
```

## 7. 显存与依赖边界

OOM 时先将 `num_workers` 调低至 0 或 1；再确认 AMP 开启并将 `decoder_channels` 从 128 降到 96。改变图像分辨率时需同步 heatmap 和相机参数尺度。按服务器给当前进程分配的显存判断，不以显卡型号标称值代替。

代码和 BPnP 依赖已包含在本目录；仍需安装环境文件里的软件、提供 SPEED+ 数据，并为 ImageNet 初始化提供联网或已有 timm 缓存。代码目录不包含数据、checkpoint，也不承诺当前本机可验证 CUDA 训练。
