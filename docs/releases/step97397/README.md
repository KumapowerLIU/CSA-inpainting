---
language: en
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-to-image
tags:
  - image-inpainting
  - coherent-semantic-attention
  - iccv-2019
  - pytorch
  - research
---

# CSA · stage 2 · step 97,397

**A modern PyTorch research checkpoint for Coherent Semantic Attention for Image
Inpainting (ICCV 2019).** This checkpoint combines a separately trained rough
network with a refinement network trained using the original PatchGAN family,
L1, LPIPS, and CSA consistency supervision.

This is the selected 2026 training run, rather than the original 2019 paper
weights. It was selected by visual review, not an automatic best-metric rule.
Model repository: [KumaPower/CSA](https://huggingface.co/KumaPower/CSA) ·
[Download the 528.5 MB inference weight](https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt) ·
[Source repository](https://github.com/KumapowerLIU/CSA-inpainting/tree/master).

![Selected stage-2 completions at step 97,397](../../assets/csa-step97397.jpg)

Columns: masked input, restored image, original. See the
[full eight-case gallery](../../assets/csa-step97397-full.jpg) for additional
examples, including faces and structures that the model does not reconstruct well.

## Model at a glance

| Property | Value |
|---|---|
| Refinement checkpoint | Stage 2, step **97,397** |
| Frozen rough parent | Stage 1, step **156,181** |
| Architecture | Two U-Nets with coherent semantic attention; width 64, depth 8 |
| Refinement output | `tanh`, then composite known pixels from the input |
| Training resolution | 256 × 256 RGB |
| Training masks | Random rectangles, strokes, or their union; 10–40% missing area |
| Stage-2 image discriminator | Original PatchGAN; no Real-ESRGAN initialization |
| Feature discriminator | Original feature PatchGAN |
| Stage-2 batch / precision | 192 / BF16 on one H100 |
| Optimizer | Adam, initial LR 0.0002, betas (0.5, 0.999) |

Stage 2 optimizes full-image raw-refinement **L1 × 0.1**, completed-image
**LPIPS (VGG) × 1**, **CSA consistency × 0.01**, and image/feature relativistic
average least-squares GAN terms, **0.002 each**. Stage 1 is frozen throughout
refinement training. These modern training choices differ from the paper's
original recipe. The sanitized [training configuration](training-config.yaml)
records the selected stage-2 settings.

## Training data

The base index contains 296,145 Unsplash records and 901,730 images sampled from
the 1,803,460-image Places365-Standard training archive. Later stage-2 training
also includes 30,128 distinct HDTF talking-head frames, repeated across 299,469
sampling entries. The final index has **1,497,344 entries**, approximately 20%
face sampling weight. Entry counts are not counts of distinct image contents;
cross-source content deduplication was not performed.

HDTF validation frames were separated by source-video groups. Identity-disjoint
splits have not been established. The evaluation below uses the unchanged
general-image holdout, not a dedicated face benchmark. Training images and
private dataset indices are not distributed with the checkpoint.

The [dataset guide](../../DATASETS.md) lists official acquisition links, the
Places subset procedure, and the limits of reproducing these local splits.
See [Training](../../TRAINING.md) for an executable two-stage recipe with your
own complete RGB images.

## Evaluation

The recorded check uses 64 fixed records from a 2,000-record Unsplash holdout,
fixed random masks, FP32 inference, and batch size 1 on an H100. The holdout is
record-level and is not content-deduplicated. Metrics average per-image values;
L1 is measured on RGB values in [0, 1].

| Output | Hole PSNR ↑ | Hole L1 ↓ | Full-image SSIM ↑ |
|---|---:|---:|---:|
| Frozen stage 1 | 18.4163 | 0.083474 | 0.863315 |
| Stage 2 · 97,397 | 18.1563 | 0.083444 | 0.867322 |

This is a small internal spot check, not the paper's evaluation protocol or a
state-of-the-art comparison. Refinement improves SSIM on this sample while hole
PSNR is lower than the frozen rough model. The machine-readable report is
[`evaluation.json`](evaluation.json).

Natural backgrounds and smooth regions are useful qualitative examples. Large
holes over faces, buildings, and missing semantic objects can still produce
blur, incorrect geometry, or repeated texture. The full gallery includes these
failure cases; inpainting is a plausible completion, not a recovery guarantee.

## Files and inference

The release inference weight is **`csa-stage2-step97397.pt`**. It contains both
the rough and refinement networks and the configuration required for inference.
The file is 528,510,920 bytes (528.5 MB). No separate stage-1 weight,
discriminator, VGG/LPIPS checkpoint, or training
dataset is required at inference time. The file is loaded with PyTorch's
`weights_only=True` and strict model-state matching.

Use **Python 3.10+** with a matching PyTorch/torchvision pair from the
[official PyTorch installer](https://pytorch.org/get-started/locally/).
The package supports PyTorch `>=2.6,<3` and torchvision `>=0.21,<1`;
not every cross-version pairing is compatible.

The repository includes an [installable wheel](https://huggingface.co/KumaPower/CSA/resolve/main/csa_inpainting-0.3.0-py3-none-any.whl),
[complete source archive](https://huggingface.co/KumaPower/CSA/resolve/main/csa-inpainting-source.tar.gz), and
[example images](https://huggingface.co/KumaPower/CSA/tree/main/examples). A source checkout is optional.
After installing the appropriate PyTorch/torchvision pair in your environment:

```bash
python -m pip install \
  "https://huggingface.co/KumaPower/CSA/resolve/main/csa_inpainting-0.3.0-py3-none-any.whl"

curl -fL https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt \
  -o csa-stage2-step97397.pt
mkdir -p examples
curl -fL https://huggingface.co/KumaPower/CSA/resolve/main/examples/masked.png \
  -o examples/masked.png
curl -fL https://huggingface.co/KumaPower/CSA/resolve/main/examples/mask.png \
  -o examples/mask.png

csa infer --checkpoint csa-stage2-step97397.pt \
  --image examples/masked.png --mask examples/mask.png \
  --output output.png --device auto
```

The [reference example output](https://huggingface.co/KumaPower/CSA/resolve/main/examples/expected.png)
is provided for comparison. The CLI selects CUDA, then MPS, then CPU when
`--device auto` is used. No additional pretrained networks are downloaded for
inference.

White mask pixels mean **missing**, black pixels mean **known**. Use `--device
cpu` for CPU execution. Images and masks must have matching dimensions. The CLI
pads to the network divisor and crops back to the original size; this checkpoint
was trained at 256 × 256, so arbitrary-size quality and memory use are not
validated. Prefer PNG output to preserve known pixels exactly.

The separate **training checkpoint** retains optimizers, discriminators,
schedulers, RNG state, and the data cursor for resuming this experiment. It is
preserved privately and is not needed for inference. SHA256 checksums and the
source snapshot accompany the assembled upload package.

## Provenance

| Artifact | SHA256 |
|---|---|
| Portable inference checkpoint | `605b80c18dcb0856367e6276813df0b54ec42aabea782af1631052aedb7abdfe` |
| Full stage-2 training checkpoint | `9613d50e7d5cc7e7bf097b13a3987d29d086f82c84596c4a7b35411dcfa92c52` |
| Frozen stage-1 parent | `0ede01d5945c1bb16fa08553ddff5a0279a9db219601729420e277e78748b041` |

The export preserves the trained network parameters without quantization or
fine-tuning. All generator tensors have identical checksums before and after
export. Eight full-versus-export forward comparisons are bitwise equal under
deterministic GPU algorithms on PyTorch 2.6.0 / H100; see
[`export-verification.json`](export-verification.json). Backend choices can
change numerical results, particularly the discrete nearest-neighbor selection
in CSA; cross-device or default-kernel bitwise identity is not promised.
Packaging and inference checks do not change or resume training.

## Citation and license

Hongyu Liu, Bin Jiang, Yi Xiao, and Chao Yang. **Coherent Semantic Attention for
Image Inpainting.** ICCV 2019.

```bibtex
@InProceedings{Liu_2019_ICCV,
  author = {Liu, Hongyu and Jiang, Bin and Xiao, Yi and Yang, Chao},
  title = {Coherent Semantic Attention for Image Inpainting},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```

The release retains the repository's **CC BY-NC 4.0 / noncommercial academic
research** notice. Dataset and example-image rights remain with their respective
owners; this release does not grant additional dataset rights.
