# 本次验证记录

日期：2026-10-20。环境：macOS arm64、Python 3.13、PyTorch 2.14.1、
torchvision 0.29.1，Apple MPS 可用，CUDA 不可用。
依赖由 [`uv.lock`](../uv.lock) 锁定，机器可读结果在
[`validation-results.json`](validation-results.json)。

## 已通过

| 检查 | 结果 |
|---|---|
| `pytest -q` | 42 passed，2 skipped（CUDA fp16/bf16 无设备） |
| `ruff check src tests scripts` | 通过 |
| `ruff format --check src tests scripts` | 20 个 Python 文件格式检查通过 |
| `uv sync --locked --extra dev` / `uv lock --check` | 锁文件可解析，当前环境同步成功 |
| `uv pip check` | 无依赖冲突 |
| Python 安装包构建 | 前轮成功构建 wheel；本轮 uv 重新构建 editable 包并安装成功 |
| CPU smoke：batch=2，64×64，小网络，2 步 D/G 更新 | 所有损失有限，断点重载输出逐值一致 |
| MPS smoke：相同设置 | 所有损失有限，断点重载输出逐值一致 |
| 完整默认网络：CPU、batch=2、256×256，1 步 D/G 更新 | 完成，loss/gradient 有限 |
| README 本地渲染与链接检查 | Chromium 渲染检查通过，本地文档/图片链接有效 |
| 仓库清理 | 无 notebook，旧字节码与旧 Jupyter 环境文件已删除 |

完整网络检查使用默认宽度 64、深度 8、两个默认判别器、真实 VGG16 层结构，
生成器参数总数为 132,111,750。**VGG 权重随机，仅用于离线软件集成检查**。
它不验证 ImageNet 特征质量、收敛或修复效果。
可用 `python scripts/check_full_step.py --device cpu --batch-size 2` 重跑这一检查。
该检查显式选择 `diagnostic="random_vgg"`，报告与 checkpoint 会标明特征来源。

自动测试覆盖：

- NumPy 独立公式实现与 CSA 输出对照，包括正/负特征、矩形特征图。
- float64 数值梯度检查，避免“forward 看起来正常、backward 错误”。
- batch 内不同掩码与逐张调用的输出/梯度一致；不同样本没有交叉梯度。
- 分块搜索、全零特征、空洞掩码、全缺失与非法掩码。
- 特征 mask 与原始三层 4×4 常量卷积逐值对照。
- rough←refinement、consistency→encoder/decoder、feature GAN→generator 的梯度连通。
- VGG pool3 / relu4_3 的层索引、输入标准化、冻结权重和输入梯度。
- 默认深度的 256×512 矩形网络前向（测试使用较窄通道）。
- G/P/D/F 四个网络确实更新参数。
- 保存/恢复后，下一步损失与生成器参数在同一 CPU 环境逐值一致。
- 递归图片加载、mask 极性、固定验证配对、跨 worker 的增强可复现。
- L1/L2/PSNR 已知数值、SSIM 恒等值、验证最后一个不足 batch 的正确加权。
- 训练 → 保存 → 恢复 → 评估 → 原尺寸推理命令入口全链路。
  入口测试用小网络 fixture 替换训练器特征，不下载预训练 VGG。
- 67×71 推理图片自动补齐/裁剪，已知区域保存为 PNG 后像素完全一致。
- 配置覆盖、类型/非有限值校验、不创建数据/模型的 dry-run。
- 新实验不能覆盖非空输出目录，运行记录包含完整有效配置和源码 SHA256。
- 等待恢复或恢复失败的训练器不能继续更新、保存或推进 epoch。
- 随机 VGG 诊断状态不能伪装为正常训练 checkpoint。
- 选定 CUDA 设备的 BF16 检查与不同 GPU 数量的 RNG 恢复拒绝逻辑。
  这两项逻辑测试使用 mock，不代表 CUDA 硬件实测。

## 未做的验证

没有训练 CelebA、Paris StreetView 或 Places2；没有核验论文数值或主观修复质量。
没有 CUDA 硬件实测、AMP 收敛测试、多 GPU 或长时间性能测试。
没有加载作者旧 checkpoint；新版 checkpoint 与旧格式不兼容。
没有把随机网络预览作为模型效果展示。

代码和数学层面的检查已经完成。后续如要验收论文复现，仍需固定数据划分、
掩码协议、学习率日程、预训练 VGG、完整训练过程和独立评估记录。
