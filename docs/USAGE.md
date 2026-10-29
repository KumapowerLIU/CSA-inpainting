# Usage

For the trained release, start with the [README quickstart](../README.md#try-the-checkpoint). This guide covers the installed `csa` CLI. Sequential stage training has its own [training guide](TRAINING.md).

## Installation and devices

For a pinned Linux/NVIDIA setup, use the [README installation recipe](../README.md#1-install). The production runtime was checked with Python 3.10.14, PyTorch 2.6.0+cu124, and torchvision 0.21.0+cu124; package imports and torchvision's compiled CPU operators were also verified. These wheels include the CUDA 12.4 runtime and require a compatible NVIDIA driver, without a separate CUDA toolkit installation.

Use Python 3.10+ and a matching PyTorch/torchvision pair within the package's supported ranges: PyTorch `>=2.6,<3`, torchvision `>=0.21,<1`. Choose the appropriate CPU or CUDA wheels from the [official installer](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e .
csa --help
```

For training with LPIPS or for development:

```bash
python -m pip install -e '.[perceptual]'
python -m pip install -e '.[dev]'
```

`--device auto` selects CUDA → MPS → CPU. An explicit device such as `cuda:0`, `mps`, or `cpu` overrides it. FP16/BF16 training is CUDA-only. The indexed staged runner requires CUDA; this is separate from the inference CLI's device support. Supported version ranges describe installation compatibility, not a claim that every version/device combination was tested. The lockfile records the development environment.

## Download the release weight

The model repository is [KumaPower/CSA](https://huggingface.co/KumaPower/CSA). Download the same step-97,397 portable weight used in the release previews:

```bash
mkdir -p checkpoints
curl -fL https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt \
  -o checkpoints/csa-stage2-step97397.pt
```

See [CHECKPOINTS.md](CHECKPOINTS.md) for its SHA-256 and provenance. To use the model without a source checkout, the [Hugging Face model card](https://huggingface.co/KumaPower/CSA#files-and-inference) provides wheel installation and example-image download commands.

## Inpaint one image

```bash
csa infer \
  --checkpoint checkpoints/csa-stage2-step97397.pt \
  --image examples/masked.png \
  --mask examples/mask.png \
  --output outputs/restored.png \
  --device auto
```

- Inputs are RGB images; EXIF orientation is applied before processing.
- The mask must match the oriented image's width and height. White is missing, black is known; use `--black-is-hole` to invert this convention.
- The image values beneath the hole are replaced internally. You may supply either an already masked image or an intact image with a mask.
- A completely missing image is rejected because CSA needs known context. Known pixels are copied into the completed output.
- Save as PNG when exact preservation of known pixels matters; JPEG encoding can change pixels outside the hole too.
- The default network needs spatial dimensions divisible by 256. The CLI uses replicate padding and then crops back to the original dimensions; it does not silently resize your image to 256 × 256.
- The release was trained at 256 × 256. Larger inputs cost more memory and have not been established as a higher-resolution operating point.

`examples/expected.png` is the reference output for the release example. Different devices and library versions can introduce numerical differences; exact cross-device equality is not promised.

The inference loader recognizes portable stage-2 weights, full modern stage-2 checkpoints, and older modern joint-training formats. It selects their correct output semantics. It does not convert original 2019 checkpoints or load stage-1-only checkpoints as a complete inpainting model. See [checkpoint compatibility](CHECKPOINTS.md).

## Evaluate an image directory

```bash
csa evaluate \
  --checkpoint checkpoints/csa-stage2-step97397.pt \
  --images data/validation \
  --mask-mode random \
  --batch-size 1 \
  --workers 0 \
  --save-limit 8 \
  --output outputs/evaluation \
  --device auto
```

Portable weights carry no dataset paths, so `--images` is required. Images are discovered recursively, converted to RGB, and resized to the configured evaluation resolution (256 × 256 for this release). Generated validation masks are deterministic for the saved seed and sample index. Keep the image ordering, seed, mask mode, and preprocessing fixed when comparing checkpoints.

To use external masks, replace `--mask-mode random` with `--masks data/validation-masks`. Images and masks are independently sorted and matched by index; masks cycle if fewer than images. **This is not filename-stem matching.** Masks use nearest-neighbor resizing, and `--black-is-hole` reverses their polarity.

The output contains `metrics.json` with per-image values and their means. Preview strips show **masked input → rough prediction → completed refinement → original**. The rough preview is the raw rough prediction; the other completed output preserves known pixels.

Metrics use RGB values in `[0,1]`: hole L1/L2 average over missing pixels and channels, hole PSNR is in dB, and full-image SSIM uses Gaussian weights with sigma 1.5 and population covariance. Full-image metrics include the known pixels copied from the input. Report mask area and hole metrics alongside them.

## Mask modes

| Mode | Behavior |
|---|---|
| `random` | Random rectangle, brush strokes, or their union, with bounded missing area |
| `brush` | Free-form brush masks |
| `center` | Fixed central square, useful for controlled comparisons |
| `files` | External masks supplied by `--masks` or configuration |

The release uses `random`, with 10–40% missing area. In training, the seed depends on epoch and sample index, so masks change across epochs and differ within a batch. Validation fixes the epoch component. A rectangular hole is one valid outcome of the random mixture, not evidence that training uses only a central square.

The default center mask is 120 × 120 at a 256 × 256 image size (`center_overlap=4`); overlap 0 gives 128 × 128. Internally, masks are Boolean `B × 1 × H × W`, with `True` marking the hole.

## Offline software checks

```bash
csa smoke --device cpu --batch-size 2 --steps 2
```

This runs a tiny synthetic training/checkpoint round trip without downloading VGG or LPIPS weights. Its outputs are diagnostics, not a trained demo or evidence of restoration quality.

For the repository test suite:

```bash
python -m pytest -q
```

Real perceptual training uses pretrained VGG/LPIPS weights. Fresh training can download them if they are absent from the cache; it fails rather than silently substituting random feature weights. Portable inference does not require either network.

## The joint-training CLI

`csa train` remains available for image-folder experiments. It trains rough and refinement within each batch, using separate generator optimizers and a detached rough input to refinement. Its defaults and checkpoint semantics differ from the sequential stage-2 release.

Review the resolved configuration before starting:

```bash
csa train --config configs/default.yaml \
  --set data.train_images=data/train data.val_images=data/validation \
        train.output=runs/joint-example train.device=cuda:0 \
        data.batch_size=8 train.amp=bf16 \
  --dry-run
```

Remove `--dry-run` to train. Paths are relative to the process working directory. Use a new output directory for a new experiment; resume an existing run with its own `latest.pt`. The joint CLI saves at epoch boundaries. For sequential training, step checkpoints, and sample-cursor resume, use [TRAINING.md](TRAINING.md).
