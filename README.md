<div align="center">

<p><strong>ICCV 2019 · Modern PyTorch implementation</strong></p>

# Coherent Semantic Attention for Image Inpainting

<p>Recover missing image regions with a rough prediction and context-aware refinement.</p>

<p><strong>Hongyu Liu · Bin Jiang · Yi Xiao · Chao Yang</strong></p>

<p>
  <a href="https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.html">Paper</a> ·
  <a href="#try-the-checkpoint">Quickstart</a> ·
  <a href="https://huggingface.co/KumaPower/CSA">Hugging Face</a> ·
  <a href="docs/releases/step97397/README.md">Checkpoint card</a> ·
  <a href="docs/TRAINING.md">Training</a> ·
  <a href="docs/DATASETS.md">Datasets</a> ·
  <a href="docs/MIGRATION.md">Implementation notes</a>
</p>

<img src="docs/assets/csa-step97397.jpg" alt="Actual step 97,397 inference: masked input, restored image, and original image, shown for three selected examples." width="960">

<p><sub>Actual inference from the modern stage-2 checkpoint at step 97,397. Columns: masked input · restored · original.<br>
Selected examples; <a href="docs/assets/csa-step97397-full.jpg">view all eight preview cases, including failures</a>.</sub></p>

</div>

This is the author's CSA research repository, reorganized into a typed Python package with explicit masks, native autograd attention, separate training stages, and portable inference weights. The current checkpoint was trained with random masks, **L1 + LPIPS**, and the original **image PatchGAN plus feature discriminator**. Its image discriminator uses the original PatchGAN architecture and random initialization.

The 2019 paper and code are the starting point. This trained release includes deliberate numerical, loss, and training changes; it is **a modern research experiment, not a paper-equivalent reproduction or a state-of-the-art claim**. See the [migration notes](docs/MIGRATION.md) for what changed.

## Try the checkpoint

Use Python **3.10 or newer**. Install a matching PyTorch/torchvision pair for your CPU or CUDA platform using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install this repository. Supported dependency ranges are PyTorch `>=2.6,<3` and torchvision `>=0.21,<1`; these ranges do not mean every cross-version pairing is compatible.

```bash
git clone --branch codex/modern-pytorch https://github.com/KumapowerLIU/CSA-inpainting.git
cd CSA-inpainting
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching torch + torchvision pair for your platform first.
python -m pip install -e .
```

Download the 528.5 MB [portable checkpoint](https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt) from [KumaPower/CSA](https://huggingface.co/KumaPower/CSA). The [checkpoint card](docs/releases/step97397/README.md) records its checksum and provenance. Run the included example:

```bash
mkdir -p checkpoints
curl -fL https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt \
  -o checkpoints/csa-stage2-step97397.pt

csa infer \
  --checkpoint checkpoints/csa-stage2-step97397.pt \
  --image examples/masked.png \
  --mask examples/mask.png \
  --output outputs/restored.png \
  --device auto
```

**White mask pixels mark the hole.** The output copies known pixels from the input and fills the masked region. The release was trained at 256 × 256; inference pads other sizes to the network's stride and crops the output back, but quality at other resolutions has not been established. Compare the example with [`examples/expected.png`](examples/expected.png).

Inference needs only the portable checkpoint: no discriminator, VGG, LPIPS, or pretrained-weight download. `auto` selects CUDA, then MPS, then CPU. See [usage](docs/USAGE.md) for custom images, folder evaluation, and an offline software smoke test.

## What the release contains

| Component | Current release |
|---|---|
| Rough prediction | Separately trained U-Net, frozen during stage 2 |
| Refinement | CSA refinement network with bounded `tanh` output |
| Masks | Per-image random rectangles, brush strokes, or their union; 10–40% missing area |
| Stage-2 image objective | `0.1 × L1 + 1.0 × LPIPS` |
| Additional supervision | `0.01 × consistency + 0.002 × (image GAN + feature GAN)` |
| Discriminators | Original image PatchGAN and VGG-feature PatchGAN; RaLSGAN objective |
| Inference artifact | `csa-stage2-step97397.pt` · 528.5 MB · both generators and public configuration |

Training and CUDA execution have been exercised in the staged training runs. The release preserves the stage-2 architecture and output semantics of the paused step-97,397 checkpoint. The [checkpoint card](docs/releases/step97397/README.md) records its provenance, checksums, validation, and limitations.

## Train and extend

The trained release uses **sequential stages**: train rough prediction first, freeze it, then train CSA refinement and its discriminators. The [training guide](docs/TRAINING.md) documents the indexed-data runner, losses, batch-size probes, and checkpoint resume.

1. Install the training extra with `python -m pip install -e '.[perceptual]'` on a CUDA machine.
2. Put complete RGB images in separate train/validation folders, then follow the [runtime and index preparation](docs/TRAINING.md#prepare-independent-runtimes).
3. Run [stage 1](docs/TRAINING.md#stage-1-rough-reconstruction), select its checkpoint on held-out images, then run [stage 2](docs/TRAINING.md#stage-2-csa-refinement) with that frozen parent.

Both stages use **L1 = 0.1 and LPIPS = 1** in the documented fresh-training recipe. Stage 2 adds CSA consistency and the original image/feature PatchGAN losses. Masks are generated online, with 10–40% missing area; no mask dataset is needed. The guide starts at batch 8 and includes a memory probe. The recorded runs used batch 256 for the stage-1 LPIPS phase and 192 for stage 2 on one H100.

### Training data and where to find it

| Dataset | Used in this release | Official source |
|---|---|---|
| Unsplash-derived photographs | 296,145 training records; 2,000 held-out records | [Unsplash Dataset](https://unsplash.com/data) · [access and schema](https://github.com/unsplash/datasets) |
| Places365-Standard | 901,730 images: a fixed random half of the 1,803,460-image training archive | [Download page](http://places2.csail.mit.edu/download.html) · [CSAIL repository](https://github.com/CSAILVision/places365) |
| HDTF talking-head frames | Added during stage 2: 30,128 selected frames, repeated into 299,469 sampling entries | [HDTF data and preprocessing](https://github.com/MRzzm/HDTF) |

The base mixture has **1,197,875 entries**; the final stage-2 mixture has **1,497,344 sampling entries**, approximately 20% faces. This is not a count of unique image contents. The Unsplash selection and HDTF frame manifests are private experiment indices, so downloading the public sources does not recreate the exact split. The [dataset guide](docs/DATASETS.md) explains acquisition, subset selection, validation boundaries, and how to use your own images. These are the modern release's data choices, not the original paper's benchmark protocol.

### Other workflows and reference

The package also retains `csa train`, an image-folder **joint-training** entry point. It updates rough and refinement in the same training loop with detached rough inputs to refinement. It is a separate workflow; it does not reproduce the staged release simply by using the default config.

| Guide | Contents |
|---|---|
| [Usage](docs/USAGE.md) | Installation, masks, inference, evaluation, smoke checks |
| [Training](docs/TRAINING.md) | Stage 1 → stage 2, indexed datasets, objectives, resume |
| [Datasets](docs/DATASETS.md) | Data sources, official access links, mixture and split details |
| [Checkpoints](docs/CHECKPOINTS.md) | Portable export, full training states, compatibility |
| [Migration](docs/MIGRATION.md) | Original-code audit, preserved structure, deliberate differences |
| [Design](docs/DESIGN.md) | Package layout and engineering decisions |

```text
src/csa_inpainting/    models, attention, losses, trainers, checkpoint I/O, CLI
scripts/remote/       indexed-data training and evaluation runners
configs/              joint-training configuration
examples/             input, mask, and expected release output
docs/releases/        checkpoint provenance and evaluation records
legacy/               original source, excluded from the installed package
tests/                numerical, gradient, checkpoint, and training checks
```

## Paper, history, and license

CSA was introduced in **Coherent Semantic Attention for Image Inpainting**, ICCV 2019. The historical implementation is preserved in [`legacy/`](legacy/) and in the [original Git revision](https://github.com/KumapowerLIU/CSA-inpainting/tree/f0b0712aa950cc559062dfef04c2f3f884a1f3e2). Historical paper figures are not used as evidence for this checkpoint's performance.

```bibtex
@InProceedings{Liu_2019_ICCV,
  author    = {Liu, Hongyu and Jiang, Bin and Xiao, Yi and Yang, Chao},
  title     = {Coherent Semantic Attention for Image Inpainting},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  month     = {October},
  year      = {2019}
}
```

The original attribution and noncommercial research-use notice are retained in [LICENSE.md](LICENSE.md). Third-party components retain their own notices. Please check the checkpoint card for artifact-specific information before redistribution.
