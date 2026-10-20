# 迁移依据与有意差异

基线为作者仓库 commit `f0b0712aa950cc559062dfef04c2f3f884a1f3e2`。
原始 Python 源码和论文图在 `legacy/`，不参与新版 import、安装或测试收集。
按项目整理要求，所有 notebook 和历史 Python 字节码已删除；原始版本仍可在上述 Git
基线中查阅。新版全部通过 Python 包和命令行运行。

参考：[ICCV 正式论文入口](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.html)、
[作者论文全文和附录](https://arxiv.org/html/1905.12384)。

## 代码审计

| 原位置 | 发现 | 新版处理 |
|---|---|---|
| `train.ipynb` | `mask[0][0]` 丢弃 batch 中其他样本的 mask | B×1×H×W 的逐样本 mask |
| `util/util.py:cal_feat_mask` | 强制 batch=1 | 批量 avg_pool，等价于原 4×4 常量卷积 |
| `models/CSA_model.py` | mask 缓存与第一张样本耦合，`fixed_mask` 容易使动态 mask 失效 | mask 是 forward 的显式参数，无跨次缓存 |
| `models/CSAFunction.py` | 用 LongTensor 存递推浮点权重 | 自动微分浮点计算，不保存整张 attention 路由矩阵 |
| 同上 | 手写 backward 额外添加 identity 梯度，空间索引使用 `i*h+j` | 删除手写 backward，支持矩形特征图 |
| 同上及 `NonparametricShift.py` | 动态构建卷积、`.data`、`Variable`，检查 `is_available` 未调用函数 | 张量矩阵乘与 gather，跟随输入设备 |
| `models/CSA.py:forward` | 用 `.data` 拼粗预测，切断 refinement→rough 路径；还会原地修改输入 | `torch.where` 非原地组合，端到端可导 |
| `models/CSA.py:backward_G` | 一致性损失包装为 `Variable(gl.loss.data,...)`，与网络断开 | 显式返回 encoder/decoder 特征并直接计算损失 |
| `models/CSA.py:backward_D/backward_G` | fake VGG 由 `fake_B.data` 得到，特征对抗项不能训练 G | D 时 detach，G 时保留 VGG 对输入的梯度 |
| `models/vgg16.py` | 未做 ImageNet 标准化；`relu3_3` 实际是 pool3 | 明确 `pool3` 与 `relu4_3`，使用权重枚举、RGB 标准化 |
| checkpoint | 仅保存网络参数 | 同时保存 Adam、scheduler、scaler、epoch、step 和 RNG |
| `GANLoss` | 名字含 wgan_gp/vanilla，但实际公式始终是 RaLSGAN | 仅提供明确的 relativistic average least squares loss |

## 保留的结构

Rough 使用 8 层 pix2pix U-Net，4×4 stride-2 卷积、InstanceNorm、skip、tanh 输出。
Refinement 保留先 stem，再 dilated 4×4 stride-2 与 3×3 channel convolution 的结构，
在第 3 次下采样（256→32）后、InstanceNorm 前放置 512 通道 CSA。
其对应 decoder 的 512 通道特征也参与一致性监督。
最深 1×1 层不使用 InstanceNorm；refinement 输出沿用源码的线性输出，不擅自加入 tanh。
新版激活均为非原地操作，避免旧递归模块中 `LeakyReLU(inplace=True)` 改写用于 skip
拼接的输入；因此这里的“保留结构”指层数、通道与连接设计，不是复刻所有原地副作用。

图像判别器保留 70×70 PatchGAN；特征判别器沿用源码的
VGG pool3（256×32×32）→ 三层 stride-2 4×4 卷积（512 通道）设计。
论文正文的 pool3 描述与该输入一致，但附录 Table 5 的 VGG 层名和 stride 标注与源码不同。
这里明确以已发布代码拓扑为准，没有声称两者完全一致。

默认填洞颜色保持源码的 `[123,104,117]/127.5-1`。输入是 RGB，未将其重新解释为 BGR。

## CSA 数学与数值处理

1. 每张样本独立地在已知区域搜索 1×1 feature patch。
2. 使用论文 Eq. 1 的双边 L2 标准化 cosine similarity。原代码只标准化 known patch，
   argmax 通常相同，但记录的分值和后续递推混合比例并不相同。
3. 洞内按行优先顺序遍历。第一点取最佳源 patch，随后用当前原始 query 与前一个
   已生成 patch 的 cosine similarity 构成 Eq. 3 的递推。
4. 已知区域保持不变。硬 argmax 的索引不求导，但选中分数、源值与递推权重参与求导；
   在非平局、非 clamp 边界上用 float64 数值差分验证。
5. **数值稳定性改动**：CSA 输入来自线性卷积，可能有负 cosine。
   原始比值在 `Dad + Dmax ≈ 0` 时可能发散。新版把两种相似度裁到非负，
   总和接近零时退回最佳源 patch。两种分值为正时与 Eq. 3 一致；
   对负分值的处理是明确的工程选择，尚需后续训练/消融验证，不是论文原式逐值复现。
6. 空洞掩码是恒等映射；全缺失特征图无可搜索上下文，报出错误。
   极细原图掩码经过阈值下采样可能没有 feature holes，此时 CSA 为恒等，
   图像重建与对抗训练仍有效。

搜索阶段在 no_grad 下分块求索引，然后重新计算选中的相似度以保留必要梯度；
避免保存完整两两相似度的 autograd 图。生成过程仍是顺序递推，并按样本循环，
这保证任意形状 mask 的正确性，但未将其宣称为充分优化的并行 CUDA kernel。
不额外加入原版自定义 backward 的 identity 直通梯度。

## 损失与训练

- reconstruction 是 rough/refined 对 GT 的完整图像 mean L1。
- consistency 对 CSA encoder 与对应 decoder 的误差同时施加 feature mask，
  然后对整张特征图 mean MSE。避免旧实现只 mask prediction、未 mask target
  造成的已知区域常数偏置。论文写作求和、源码为 mean；此处保留 mean 的尺度约定。
- 默认 loss 权重采用论文 `1, 0.01, 0.002`；旧 notebook 为 `100, 1, 0.2`，
  比例相同，总尺度不同。Adam 对整体尺度并非在所有数值条件下严格不变。
- G 的对抗损失将图像与特征两项相加；D 的两项平均，与旧源码组合方式一致。
- VGG 参数冻结但 fake 分支对输入可导；GT 分支 no_grad。
- 两阶段共同接受 refinement 梯度，符合论文端到端训练意图；旧 `.data` 隔离不保留。
- 采用单个 G optimizer 管理两个生成器，单个 D optimizer 管理两个判别器，
  四者仍有独立的逐参数 Adam 状态，学习率和 betas 相同。
- InstanceNorm 使特征归一化不依赖同批其他图片，但 RaLSGAN 的均值故意跨整个
  batch/patch 计算；增大 batch 不意味着优化轨迹与 batch=1 相同。

## 可复现边界

配置、CPU/Python/CUDA/MPS 随机状态、优化器等都写入新版 checkpoint。
数据增强按 seed/epoch/index 派生；shuffle 按 seed+epoch 派生，因此不依赖 worker 数量。
恢复精度的测试针对同一 CPU 软件环境；跨设备/跨库版本不保证逐比特一致。
只在 epoch 末保存；不声称支持 epoch 中间精确恢复。
允许迁移文件路径、worker 数和运行设备，但数据内容变更会改变后续训练，即使配置兼容。
本版不包含 DDP、梯度累积或 torch.compile（动态 CSA 递推需先做编译兼容性与性能验证）。
依赖由 `uv.lock` 锁定；实验支持带类型校验的 `--set` 和不加载数据/模型的 `--dry-run`，
每次运行记录环境、配置来源和源码哈希。设计取舍见 [DESIGN.md](DESIGN.md)。

旧 checkpoint 参数名、历史梯度语义、VGG 预处理都与本版有差异，不能直接 strict=False
装入后称为成功迁移。尚无用户提供的旧 checkpoint，本次未实现或声称验证权重转换。
