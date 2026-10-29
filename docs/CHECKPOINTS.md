# Checkpoints

The release repository is [KumaPower/CSA](https://huggingface.co/KumaPower/CSA). Its [528.5 MB portable checkpoint](https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt) is named **`csa-stage2-step97397.pt`**. The [checkpoint card](releases/step97397/README.md) records the checksum, training provenance, and evaluation.

```bash
mkdir -p checkpoints
curl -fL https://huggingface.co/KumaPower/CSA/resolve/main/csa-stage2-step97397.pt \
  -o checkpoints/csa-stage2-step97397.pt
```

The expected weight SHA-256 is `605b80c18dcb0856367e6276813df0b54ec42aabea782af1631052aedb7abdfe` (528,510,920 bytes). The same Hugging Face repository also contains the [installable wheel](https://huggingface.co/KumaPower/CSA/resolve/main/csa_inpainting-0.3.0-py3-none-any.whl), [source archive](https://huggingface.co/KumaPower/CSA/resolve/main/csa-inpainting-source.tar.gz), and [example inputs](https://huggingface.co/KumaPower/CSA/tree/main/examples).

## Portable inference versus training state

| Artifact | Contains | Purpose |
|---|---|---|
| `csa_inference_v1` | Rough + refinement weights, fill color, model/evaluation configuration, public provenance and hashes | Inference and evaluation |
| `stage2_v1` | Both generators, discriminators, perceptual networks, optimizer/scheduler/scaler states, RNG, configuration, progress and provenance | Resume sequential stage 2; also accepted for inference |
| `stage1_v1` | Rough model and its complete training state | Resume stage 1 or initialize a fresh stage 2 |
| Modern joint formats 1/2 | Earlier modern joint-training model/state | Inference; format-2 resume under matching joint semantics |
| Original 2019 files | Legacy parameter names and training conventions | Historical reference; no validated converter is provided |

The portable release deliberately omits discriminators, VGG, LPIPS, optimizers, RNG, and private filesystem paths. It is smaller and independent of training-time perceptual downloads. **It cannot resume training.** Keep the full checkpoint privately if continuation matters.

Stage 2 uses a `tanh` refinement head and frozen rough model. Older modern joint checkpoints retain their linear refinement output. The loader keeps those semantics distinct instead of silently changing the network during loading.

## Export a trained stage-2 checkpoint

```bash
csa export \
  --checkpoint runs/stage2/run/latest.pt \
  --output checkpoints/csa-stage2-step97397.pt
```

Export runs on CPU. It accepts production `stage2_v1` checkpoints with matching stage-2 semantics and ImageNet-VGG feature provenance; diagnostic fixtures, stage-1 files, and joint-training formats are rejected. Generator weights must use the expected native FP32 state: export rejects implicit dtype conversion and checks that the loaded generator state matches the source. The output must be a **new file**: the command does not overwrite an existing path.

The exported package includes:

- Both generator components and fill color, with a generator-state SHA-256.
- Architecture parameters and explicit stage-2 output semantics.
- Evaluation resolution, seed, mask convention, and generated-mask settings.
- Source checkpoint hash, step/epoch, training discriminator type, and sanitized parent-stage provenance.
- The input contract: RGB in `[-1,1]`, `True` means missing, and known pixels are copied to the completed image.

The CLI prints export metadata as JSON. Keep the exported file's external SHA-256 with the model card as well; the internal generator hash serves a different purpose from a whole-file checksum.

## Use a portable package

```bash
csa infer \
  --checkpoint checkpoints/csa-stage2-step97397.pt \
  --image examples/masked.png --mask examples/mask.png \
  --output outputs/restored.png

csa evaluate \
  --checkpoint checkpoints/csa-stage2-step97397.pt \
  --images data/validation --mask-mode random \
  --batch-size 1 --output outputs/evaluation
```

Use the [usage guide](USAGE.md) for mask polarity, image sizes, metric definitions, and evaluation preprocessing. A portable checkpoint has no default dataset path: evaluation requires `--images`.

## Resume a full checkpoint

Use the original training family and a compatible configuration. For sequential stage 2, pass `--stage stage2 --resume FULL_CHECKPOINT` to the indexed runner; do not pass the portable release or reinitialize from stage 1. A full stage-2 checkpoint already contains the frozen rough network and does not require the original stage-1 file to remain present.

Full training states can include private paths and dataset/run metadata. Publish an export and a reviewed model card rather than copying a raw training checkpoint into a public repository. See [TRAINING.md](TRAINING.md) for save cadence, clean stopping, and explicit dataset transitions.

Checkpoint loading uses restricted, weights-only deserialization and validates format, configuration, and training semantics. This is not a promise of bitwise reproducibility across devices or library versions, nor a converter for incompatible model families.
