<!-- Visual reference: https://github.com/VoltAgent/awesome-design-md/blob/main/design-md/apple/DESIGN.md
     Adapted to GitHub Markdown with local SVG artwork and native, selectable text. -->

<br>

<p align="center"><sub>ICCV 2019 · MODERN PYTORCH REIMPLEMENTATION</sub></p>

<h1 align="center">CSA Inpainting</h1>

<p align="center">
  Coherent Semantic Attention for Image Inpainting<br>
  <sub>Hongyu Liu · Bin Jiang · Yi Xiao · Chao Yang</sub>
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.html">Paper ↗</a>
  &nbsp; · &nbsp;
  <a href="#quick-start">Quick start ↓</a>
  &nbsp; · &nbsp;
  <a href="docs/USAGE.md">中文指南 ↗</a>
</p>

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
  <img src="docs/assets/hero.svg" alt="The whole picture. A conceptual illustration of interrupted contours becoming continuous through inpainting." width="100%">
</picture>

<p align="center"><sub>Context search. Coherent generation. Image refinement. · Conceptual illustration.</sub></p>

<br>

The original two-stage inpainting method, rebuilt as a focused Python package. Independent masks for every sample. Native autograd throughout. One command-line interface for training, evaluation, and inference.

**Research status.** Software and gradient checks pass, including batched CPU and Apple MPS updates. Dataset training and reproduction of the paper's metrics are still pending. [See the validation record →](docs/VALIDATION.md)

<br>

## Quick start

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run these commands from the repository root:

```bash
uv sync --locked --extra dev
uv run --locked csa smoke --device cpu --batch-size 2 --steps 2
```

`uv sync` creates an isolated environment and installs the locked dependencies. The smoke check runs two optimizer steps and verifies checkpoint restoration using synthetic inputs, without downloading datasets or VGG weights.

**Python 3.13** by default, with 3.11+ supported. **PyTorch 2.14.1** and **torchvision 0.29.1** pinned.

[pip & CUDA installation →](docs/USAGE.md#安装环境) &nbsp; · &nbsp; [Default configuration →](configs/default.yaml)

<br>

## Context becomes continuity.

The rough U-Net predicts the missing structure. The refinement U-Net applies CSA at one-eighth resolution: each missing feature retrieves a matching known feature, then incorporates the preceding generated feature. VGG consistency supervision and image/feature discriminators guide the result.

<br>

<p align="center">
  <img src="docs/assets/paper-results.jpg" alt="Historical 2019 results: paired groups of Input, Result, and Ground Truth for buildings, faces, and landscapes." width="880">
</p>

<p align="center">
  <sub>Original 2019 results · Input / Result / Ground Truth<br>
  These examples come from the historical repository. They were not produced by this rewrite.</sub>
</p>

<br>

## Built for experiments.

**Independent samples.** Each sample has its own mask. Cosine search is chunked, and the coherent generation recurrence remains differentiable.

**Every run has a record.** Typed configuration, YAML overrides, dry runs, and source/environment records make experiment settings inspectable.

**Resume with the full state.** Checkpoints include the model, Adam optimizer, scheduler, AMP scaler, epoch, step, and RNG state. Resuming is supported at epoch boundaries.

**Choose your device.** CPU, CUDA, and Apple MPS are supported in code. CUDA uses `torch.amp` with fp32 CSA computation; hardware validation is pending.

<br>

## One interface. Every workflow.

Preview an experiment before allocating a model or loading data:

```bash
uv run --locked csa train --config configs/default.yaml \
  --set data.batch_size=4 train.output=runs/csa-b4 \
  --dry-run
```

Set your data paths in [`configs/default.yaml`](configs/default.yaml), then train:

```bash
uv run --locked csa train --config configs/default.yaml
```

<details>
<summary><strong>Resume, evaluate, and inpaint</strong></summary>

```bash
# Resume at an epoch boundary
uv run --locked csa train --config configs/default.yaml \
  --resume runs/csa/latest.pt

# Evaluate with fixed image/mask pairs
uv run --locked csa evaluate --checkpoint runs/csa/latest.pt \
  --images /path/to/validation --masks /path/to/masks \
  --batch-size 4 --output runs/evaluation

# Inpaint one image
uv run --locked csa infer --checkpoint runs/csa/latest.pt \
  --image /path/to/input.png --mask /path/to/mask.png \
  --output runs/result.png
```

</details>

**Mask convention:** white / `1` means missing. Masks have shape `B × 1 × H × W`, with one independent mask per sample. Inference preserves input dimensions and known pixels. Training uses ImageNet-pretrained VGG16.

[Full usage guide →](docs/USAGE.md) &nbsp; · &nbsp; [Numerical and gradient changes →](docs/MIGRATION.md)

<br>

## Validation

| Check | Evidence |
| :--- | :--- |
| CSA mathematics | Independent NumPy reference and float64 numerical gradient checks |
| Batch isolation | Outputs and gradients compared with individual calls using different masks |
| CPU & Apple MPS | Batch-2 optimizer updates and exact checkpoint output round trips |
| Full architecture | CPU · 256 × 256 · batch 2 · one complete discriminator/generator update |
| CUDA & mixed precision | Tests provided; hardware validation pending |
| Dataset quality & paper metrics | Not yet evaluated |

```bash
uv run --locked --extra dev pytest -q
```

The full-architecture check uses randomly initialized VGG weights to verify execution. It does not establish feature quality or convergence. CSA generation remains sequential; optimal GPU throughput, DDP, and legacy checkpoint compatibility are not claimed.

[Validation details →](docs/VALIDATION.md) &nbsp; · &nbsp; [Engineering decisions →](docs/DESIGN.md)

<br>

## Explore the implementation

<details>
<summary><strong>A small package, with explicit responsibilities</strong></summary>

```text
src/csa_inpainting/
├── attention.py     Cosine search and coherent generation
├── networks.py      U-Nets, discriminators, VGG features
├── masks.py         Binary masks and feature-space projection
├── losses.py        Reconstruction, consistency, RaLSGAN
├── data.py          Images, masks, deterministic augmentation
├── engine.py        Optimization, device handling, checkpoints
├── experiment.py    Output protection and invocation records
├── evaluation.py    Per-image metrics and previews
├── diagnostics.py   Synthetic software-check fixtures
├── config.py        Typed, validated experiment settings
└── cli.py           Train · evaluate · infer · smoke

configs/             Experiment defaults
tests/               Numerical and integration checks
docs/                Guides, design decisions, verification
legacy/              Historical Python reference
```

The active package runs independently of the archived implementation.

</details>

[Source code →](src/csa_inpainting) &nbsp; · &nbsp; [Tests →](tests) &nbsp; · &nbsp; [Historical implementation →](legacy)

<br>

## Citation

```bibtex
@InProceedings{Liu_2019_ICCV,
  author    = {Liu, Hongyu and Jiang, Bin and Xiao, Yi and Yang, Chao},
  title     = {Coherent Semantic Attention for Image Inpainting},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  month     = {October},
  year      = {2019}
}
```

<br>

This project retains the original [CC BY-NC 4.0 notice](LICENSE.md) for educational and academic research use. The historical implementation is based on commit [`f0b0712`](https://github.com/KumapowerLIU/CSA-inpainting/tree/f0b0712aa950cc559062dfef04c2f3f884a1f3e2).
