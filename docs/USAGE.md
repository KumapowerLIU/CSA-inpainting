# CSA Inpainting 使用指南

[项目首页](../README.md) · [迁移依据](MIGRATION.md) · [验证记录](VALIDATION.md)

本项目将 ICCV 2019 的 CSA 图像修复方法重写为独立 Python 包。训练、评估、推理和自检统一通过 `csa` 命令完成。默认保留两阶段 U-Net、CSA、VGG 一致性监督和两个判别器。

本次完成的是代码与软件验证。尚未开展数据集训练，也没有宣称达到论文指标。算法数值处理、梯度和旧代码之间的差异均记录在[迁移说明](MIGRATION.md)中。

## 安装环境

项目默认使用 Python 3.13，支持 Python 3.11+。PyTorch 固定为 **2.14.1**，torchvision 固定为 **0.29.1**。版本于 2026-10-20 核对官方发布信息与包索引后选定。

推荐使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和仓库中的依赖锁文件。在项目根目录执行：

```bash
uv sync --locked --extra dev
uv run --locked csa --help
```

`--locked` 要求解析结果与 `uv.lock` 一致，避免运行命令时悄悄更新依赖。`--extra dev` 安装 pytest 和 Ruff 等开发工具。

也可以使用标准 venv + pip：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
csa --help
```

CUDA 服务器请先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与 GPU、驱动兼容的 **torch 2.14.1 / torchvision 0.29.1** wheel，再安装本项目。若使用特定 CUDA 索引，在该 pip 环境中直接运行 `csa`，避免另一个环境管理器重新同步并替换已选择的 wheel。

CPU 专用的 pip 安装可以先运行：

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

自动设备选择顺序为 CUDA → Apple MPS → CPU，可用 `--device cpu`、`--device mps` 或 `--device cuda:0` 显式指定。混合精度目前仅用于 CUDA，CPU/MPS 使用 fp32。

下文命令使用 uv；在已激活的 pip 环境中，直接运行对应的 `csa`、`pytest` 或 `python` 命令即可。

## 无数据自检

```bash
# 小网络、不同掩码、两次 D/G 更新与断点恢复
uv run --locked csa smoke --device cpu --batch-size 2 --steps 2

# Apple Silicon
uv run --locked csa smoke --device mps --batch-size 2 --steps 2 \
  --output runs/smoke-mps

# 自动测试
uv run --locked --extra dev pytest -q

# 完整默认架构的离线单步检查
uv run --locked python scripts/check_full_step.py --device cpu --batch-size 2
```

Smoke 使用小型网络和固定特征提取器，不下载数据集或 VGG 权重。完整架构检查使用真实 VGG 层结构和随机权重。两者用于验证程序链路，输出不能作为修复效果展示。

正式训练命令使用 ImageNet 预训练 VGG16；首次启动会通过 torchvision 下载权重。离线机器需要提前准备 torch hub 缓存。下载失败会报错，不会自动改用随机特征。恢复新版完整断点无需重新下载 VGG。

## 配置一次实验

[`configs/default.yaml`](../configs/default.yaml) 包含模型、数据和训练三个配置段。可以复制它保存独立实验配置，也可以使用命令行覆盖少量参数：

```bash
uv run --locked csa train --config configs/default.yaml \
  --set data.batch_size=4 train.output=runs/csa-b4 \
  --dry-run
```

`--dry-run` 校验并打印最终 YAML，不加载图片、模型或 VGG 权重。`--set` 使用已定义的 `段名.字段名=值`，拼错字段或非法值会报错。

准备好数据后，在配置中填写路径：

```yaml
data:
  train_images: /path/to/train
  val_images: /path/to/validation
  train_masks: /path/to/train_masks
  val_masks: /path/to/validation_masks
  mask_mode: files
  mask_white_is_hole: true
  batch_size: 4
train:
  output: runs/csa-b4
```

上面是需要修改的字段示例，其他设置可保留默认值。所有相对路径按命令执行时的工作目录解释。每个新实验应使用单独输出目录；非空目录只能从该目录自己的 `latest.pt` 续训，防止覆盖或混合已有实验。

### 图片与掩码

图片递归读取，不要求类别目录。默认转换为 RGB，双三次缩放到 256×256，训练时随机水平翻转。数据增强由 seed、epoch 和样本索引决定，不依赖 worker 数量或预取顺序。

每张图片都有自己的二值掩码，整个 batch 的掩码形状为 `B × 1 × H × W`，**1 / 白色表示缺失区域**。掩码使用最近邻缩放。如果原始掩码的白色表示有效区域，设置 `mask_white_is_hole: false`。

| 模式 | 行为 |
| :--- | :--- |
| `center` | 中央方形洞；默认 `center_overlap: 4`，256 图像中洞为 120×120；设为 0 得到 128×128 |
| `files` | 训练时随机选择 mask；验证时按排序后的图片和 mask 索引固定配对，mask 数量不足时循环 |
| `brush` | 内置可复现笔刷掩码，适用于开发和新实验；不等同于论文使用的不规则掩码数据集 |

验证集的固定配对规则是索引配对，不是同名文件匹配。需要特定图像/掩码对应关系时，应确保二者的排序一致。

### 默认优化设置

| 参数 | 默认值 |
| :--- | :--- |
| 图像尺寸 | 256×256 |
| 模型宽度 / 深度 | 64 / 8 |
| Adam 学习率 | `2e-4` |
| Adam betas | `(0.5, 0.999)` |
| 重建 / 一致性 / 对抗损失权重 | `1 / 0.01 / 0.002` |
| 学习率日程 | 前 20 个 epoch 固定，之后线性衰减，总计 120 个 epoch |
| 混合精度 | `off` |

损失比例来自论文，epoch 日程沿用旧训练配置。这些值尚未在新版上经过收敛验证。增大 batch 不自动缩放学习率；RaLSGAN 的均值跨 batch 与 patch 计算，因此不同 batch size 的优化轨迹不会完全相同。

CUDA 可将 `train.amp` 设为 `fp16` 或 `bf16`。相似度搜索和 CSA 递推保持 fp32。bf16 需要硬件支持；这两种模式尚未在当前机器上实测。

## 训练与续训

```bash
uv run --locked csa train --config configs/default.yaml

uv run --locked csa train --config configs/default.yaml \
  --resume runs/csa/latest.pt

uv run --locked csa train --config configs/default.yaml --device cuda:0
```

每次实验记录最终配置、训练损失、checkpoint，以及可选的验证结果：

```text
runs/csa/
├── config.yaml
├── invocations.jsonl
├── train.jsonl
├── latest.pt
└── validation-0001/
    ├── metrics.json
    └── 00000.png
```

验证预览从左到右为：输入、rough 预测、completed 修复图、GT。

`invocations.jsonl` 为每次正式启动追加一条记录，包括运行版本、解析后的设备、源码 SHA256、Git revision/dirty 状态、配置路径与命令行覆盖项。`uv.lock` 固定依赖解析结果，启动记录补充本次实际运行的代码与环境信息。

Checkpoint 保存模型、Adam、学习率调度器、AMP scaler、epoch、step 和随机状态。当前在 **epoch 边界**保存与恢复，中断时最多损失当前 epoch 的进度。最后不足 batch size 的一批样本仍参与训练。

续训要求网络和优化设置与已保存配置一致；可以迁移数据路径、运行设备和 worker 数。相同软件/设备环境的下一步一致性已测试，跨设备或跨库版本不保证逐比特一致。数据内容变化也会改变后续训练。

## 评估

```bash
uv run --locked csa evaluate --checkpoint runs/csa/latest.pt \
  --images /path/to/validation --masks /path/to/validation_masks \
  --batch-size 4 --output runs/evaluation
```

`--black-is-hole` 可覆盖掩码极性。若不传 `--images` / `--masks`，使用 checkpoint 配置中的验证路径；也可通过 `--mask-mode center` 或 `--mask-mode brush` 选择其他掩码方式。

评估计算 completed composite 的完整图像 L1、L2、PSNR、SSIM，同时报告 hole L1/L2/PSNR。保存逐图结果与逐图平均值，最后不足 batch 的样本不会被赋予错误权重。

| 指标 | 约定 |
| :--- | :--- |
| L1 / L2 | 基于 `[0,1]` 的 MAE / MSE，结果是小数；与论文百分比对照时乘 100 |
| PSNR | 单位 dB；零误差上限为 120 dB |
| SSIM | RGB，各通道平均，Gaussian window 11×11，sigma=1.5，population covariance |
| Hole metrics | 仅对缺失区域统计，按缺失像素数 × 通道数归一化 |

对照论文表格前，需要统一数据划分、预处理、掩码、权重与评估协议。软件检查通过不代表论文指标已经复现。

## 单图推理

```bash
uv run --locked csa infer --checkpoint runs/csa/latest.pt \
  --image /path/to/input.png --mask /path/to/mask.png \
  --output runs/result.png
```

输入图像和掩码尺寸必须一致。推理时自动在右侧与底部补齐到网络需要的尺寸倍数，输出时裁回原尺寸；已知区域在 composite 中保持不变。

默认深度为 8，网络输入尺寸必须是 256 的倍数。高分辨率输入会增大 CSA 特征搜索成本，不能据此推断任意尺寸都能在当前显存中运行。

## 代码设计与边界

新版使用原生 `nn.Module`、自动微分、显式张量接口、dataclass 配置、`torch.amp` 和 `torch.inference_mode`。CSA 将搜索索引与可导的分数计算分开，并按块搜索，以减少需要保留的计算图。批量中每张图片独立处理自己的 mask。

“现代化”在这里指可读、可验证、便于维护和实验。CSA 的生成算法仍然是顺序递推；本版尚未实现 DDP、梯度累积、专用 CUDA kernel 或 `torch.compile` 加速，不能宣称 GPU 性能最优。进一步优化需要先 profile，再证明输出与梯度正确，并测量实际收益。

旧版 checkpoint 不直接兼容。历史 Python 实现仅供考据，不参与新版运行；项目使用命令行完成全部工作流。

更多细节见[迁移说明](MIGRATION.md)和[验证记录](VALIDATION.md)。
