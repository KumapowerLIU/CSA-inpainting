# Training

The modern release uses two sequential training stages. First learn a rough reconstruction; then freeze that model and train CSA refinement with image and feature discriminators. This guide documents the single-GPU indexed-data runner used for that workflow. The released checkpoint is one experiment in this implementation, not a claim that the recipe reproduces the 2019 paper's benchmark.

## Choose the right entry point

| Entry point | Training behavior |
|---|---|
| `scripts/remote/csa_remote_train.py ... --stage stage1` | Rough network only; L1 and optional completed-image LPIPS |
| `scripts/remote/csa_remote_train.py ... --stage stage2` | Frozen rough network; train refinement and both discriminators |
| `csa train --config ...` | Joint loop: update rough and refinement every batch, with a detached rough input to refinement |

The indexed runner requires **CUDA**. Despite its directory name, it is an ordinary local Python process on the training machine; it does not connect to a remote host. For dataset names, the released model's mixture, and official acquisition links, see [Datasets](DATASETS.md). You can also train on your own complete RGB images. Masks are generated during training; no mask dataset is required.

## Install the training environment

Use Python 3.10+ on the CUDA machine. If you already have this checkout and a suitable environment, start at the install command:

```bash
git clone --branch codex/modern-pytorch https://github.com/KumapowerLIU/CSA-inpainting.git
cd CSA-inpainting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matching CUDA PyTorch/torchvision pair using the [official installer](https://pytorch.org/get-started/locally/), within the supported ranges of PyTorch `>=2.6,<3` and torchvision `>=0.21,<1`. Then install the training dependencies and check the selected device:

```bash
python -m pip install -e '.[perceptual]'
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Fresh perceptual training needs pretrained VGG/LPIPS weights. The first probe can download missing weights; use a populated PyTorch cache when training offline. A download failure is not replaced with random perceptual features. Portable inference does not need these networks.

The examples below use **one GPU and batch size 8** as a starting point. The training batch, resolution, mask coverage, dtype, and perceptual chunk size all affect memory. Probe your actual configuration before increasing the batch. BF16 needs a supported GPU; otherwise use `--amp fp16` or `--amp off` as appropriate. Keep the same precision policy for a continuing experiment.

## Prepare independent runtimes

From the repository root, create a frozen source snapshot for each stage:

```bash
mkdir -p runs/stage1/source runs/stage1/data
mkdir -p runs/stage2/source runs/stage2/data
cp -R src/csa_inpainting runs/stage1/source/
cp -R src/csa_inpainting runs/stage2/source/
cp scripts/remote/*.py runs/stage1/
cp scripts/remote/*.py runs/stage2/
```

The runner records source hashes from `RUNTIME/source`. Use that snapshot in `PYTHONPATH` when executing it. Do not repeatedly copy new source over a runtime that you intend to resume as the same experiment.

Each runtime needs:

```text
data/train.jsonl
data/validation.jsonl
data/dataset.json
```

A JSONL row may reference an image file:

```json
{"format": "image", "path": "/absolute/path/to/image.jpg"}
```

Other supported rows reference the contents of a member in an **uncompressed** TAR archive, or an indexed legacy TFRecord. See [record_images.py](../scripts/remote/record_images.py) for the exact schemas. The runner's `index --source ...` command is specific to the original TFRecord schema and accompanying `.tfindex` files; it is not a general TFRecord or image-folder importer.

Put complete images in two separate folders, allowing arbitrary subdirectories:

```text
data/train/          # training images
data/validation/     # held-out images
```

The following command produces both runtimes' indices. These example folders are independent of the frozen source snapshots under `runs/`:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
splits = {}
for split, folder in (("train", "data/train"), ("validation", "data/validation")):
    paths = sorted(
        path.resolve()
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"No images found in {folder}")
    splits[split] = paths

if set(splits["train"]) & set(splits["validation"]):
    raise ValueError("Train and validation paths overlap")

encoded = {
    split: "".join(
        json.dumps({"format": "image", "path": str(path)}) + "\n"
        for path in paths
    ).encode("utf-8")
    for split, paths in splits.items()
}
metadata = {
    "total": sum(map(len, splits.values())),
    "train": len(splits["train"]),
    "validation": len(splits["validation"]),
    "seed": 2019,
    "index_sha256": hashlib.sha256(encoded["train"]).hexdigest(),
    "index_sha256_scope": "exact bytes of train.jsonl",
    "validation_sha256": hashlib.sha256(encoded["validation"]).hexdigest(),
    "split": "user-supplied folders; content deduplication is the user's responsibility",
}
for runtime in ("runs/stage1", "runs/stage2"):
    destination = Path(runtime) / "data"
    destination.mkdir(parents=True, exist_ok=True)
    for split, payload in encoded.items():
        (destination / f"{split}.jsonl").write_bytes(payload)
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
```

Keep the indexed image contents and order stable when resuming. This manifest hashes the index, not every image's bytes, and does not discover visually duplicated images across splits. Use datasets you are entitled to train on, with a validation split appropriate to the question you want to measure. The local release's private training manifest is not distributed.

Images are EXIF-corrected, converted to RGB, resized to 256 × 256, and randomly flipped in training. Random masks combine rectangles, brush strokes, or both, with 10–40% missing area. They vary with epoch and sample index; validation masks are fixed. A rectangular mask is therefore a valid sample, but the recipe does not use a fixed central hole.

## Stage 1: rough reconstruction

Probe real training memory first. This performs optimizer updates in a disposable trainer and discards them:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=runs/stage1/source \
python runs/stage1/csa_remote_train.py probe \
  --runtime runs/stage1 --stage stage1 \
  --batch-size 8 --workers 4 --amp bf16 --steps 2 \
  --mask-mode random \
  --reconstruction-weight 0.1 --lpips-weight 1 \
  --lpips-net vgg --lpips-batch-size 8
```

Start a fresh stage-1 experiment:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=runs/stage1/source \
python runs/stage1/csa_remote_train.py train \
  --runtime runs/stage1 --stage stage1 \
  --batch-size 8 --workers 4 --amp bf16 \
  --mask-mode random \
  --reconstruction-weight 0.1 --lpips-weight 1 \
  --lpips-net vgg --lpips-batch-size 8 \
  --save-every 100 --preview-every 500
```

The objective is `0.1 × L1(rough, target) + LPIPS(completed_rough, target)`. L1 averages over the full image; LPIPS sees the completed image with known pixels copied from the target. Stage 1 has no discriminator. These explicit coefficients describe the chosen LPIPS recipe; runner defaults are L1 = 1 and LPIPS = 0.

Choose the stage-1 checkpoint using held-out results rather than a fixed step count alone. Finish or cleanly stop stage 1 before reading `runs/stage1/run/latest.pt` as the parent of stage 2.

## Stage 2: CSA refinement

Use a **full stage-1 checkpoint**, not the portable release, to initialize a fresh stage 2:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=runs/stage2/source \
python runs/stage2/csa_remote_train.py train \
  --runtime runs/stage2 --stage stage2 \
  --init-stage1 runs/stage1/run/latest.pt \
  --batch-size 8 --workers 4 --amp bf16 \
  --mask-mode random --image-discriminator patch \
  --reconstruction-weight 0.1 --lpips-weight 1 \
  --lpips-net vgg --lpips-batch-size 8 \
  --save-every 100 --preview-every 500
```

For a stage-2 memory probe, use the same arguments with positional mode `probe` instead of `train` and add `--steps 2`. Each probe's updates are discarded. Only begin training after the probe succeeds with suitable memory headroom.

Initialization copies rough weights and fill color, freezes rough in evaluation mode, and initializes refinement and both discriminators afresh. Stage-2 step/epoch counters and optimizers start from zero; parent checkpoint provenance is retained. The current release uses the original PatchGAN architecture with random initialization. An experimental Real-ESRGAN discriminator path remains in the code but is not the released model's training recipe.

The stage-2 generator objective is:

```text
0.1 × full-image L1(refined, target)
+ 1.0 × LPIPS(completed_refined, target)
+ 0.01 × masked-feature consistency
+ 0.002 × (image RaLSGAN + feature RaLSGAN)
```

The refinement head uses `tanh`. Consistency applies the hole mask to encoder/decoder feature errors, then averages over the full feature grid. Image and feature GAN terms are added for G and averaged for D. VGG/LPIPS parameters are frozen while gradients through generated images remain enabled.

LPIPS/VGG activation chunks bound memory; `--lpips-batch-size` is not the optimizer batch size. RaLSGAN means use the complete batch/patch outputs. Chunking does not replace the configured batch with gradient accumulation.

### Released run versus a fresh experiment

| Setting | Stage 1 LPIPS phase | Released stage 2 |
|---|---|---|
| Optimizer batch | 256 | 192 |
| GPU | 1 × NVIDIA H100 80GB HBM3 | 1 × NVIDIA H100 80GB HBM3 |
| GPU memory capacity | 80 GB | 80 GB |
| Precision | BF16 mixed precision | BF16 mixed precision |
| Image resolution | 256 × 256 | 256 × 256 |
| DataLoader workers | 32 | 32 |
| LPIPS activation chunk | 8 | 8 |
| Selected checkpoint step | 156,181 | 97,397 |
| Observed PyTorch peak allocated | 9.15 GiB | 31.18 GiB |
| Observed PyTorch peak reserved | 12.08 GiB | 34.42 GiB |
| Memory measurement scope | Production-training snapshot, 2026-10-23 | Two-update PatchGAN + face-mixture probe, 2026-10-27 |

The 80 GB figure is physical card capacity. The observed peaks are historical PyTorch allocator measurements from the stated scopes, not a full-run hardware minimum. Reserved memory includes the allocator cache; CUDA context and driver allocations can add to the process memory shown by `nvidia-smi`.

Use these batch sizes only after a successful probe on your available memory; they are recorded settings, not minimum requirements. Keep `--lpips-batch-size 8` when increasing `--batch-size` if you want the same activation chunk bound. No distributed launch or gradient accumulation is needed for the documented single-GPU workflow.

The commands above give a consistent fresh-training recipe. The released experiment includes earlier stage-1 loss/data/batch transitions and a later stage-2 face-data addition; simply running these commands on newly obtained public data will not recreate its exact trajectory. See [dataset provenance](DATASETS.md) and the [selected stage-2 configuration](releases/step97397/training-config.yaml). That YAML is a documentary snapshot; do not pass it to the joint `csa train` command expecting staged behavior.

Both stages use the configured Adam/schedule defaults: learning rate `2e-4`, betas `(0.5, 0.999)`, 120 epochs with decay beginning at epoch 20. Pass `--epochs` and `--decay-start` for a deliberate new schedule. A step-97,397 snapshot does not imply this schedule reached completion.

## Checkpointing, stopping, and resuming

The runner maintains `run/latest.pt` at the requested save cadence, at epoch completion, and on a handled SIGINT/SIGTERM. It also saves at step 10 for an initial recovery point. It does not retain a new full checkpoint for every step or automatically select a “best” checkpoint. Copy an accepted `latest.pt` to a separately named file before later saves overwrite it.

Preview/evaluation directories are written at step 10, every `--preview-every` steps, and at epoch completion. The periodic previews use up to the first 64 validation records; epoch-end evaluation uses the full validation index. Use matching images, masks, and inference settings when comparing selected checkpoints.

Use Ctrl-C or a normal SIGTERM and let the current batch finish and the atomic save complete. A forced process kill cannot perform that save. The runtime's `training-state.json`, `run/train.jsonl`, and checkpoint step provide progress evidence; a launcher PID alone does not establish that training is advancing.

To resume, repeat the same stage command and configuration, replacing `--init-stage1 ...` with:

```text
--resume runs/stage2/run/latest.pt
```

For stage 1, add `--resume runs/stage1/run/latest.pt` to its command. Resume restores model, optimizer, scheduler, scaler, RNG, and sample position. A stage-2 full checkpoint includes its perceptual weights and frozen rough model; resume does not need the original stage-1 file or a fresh pretrained download.

Changing data, loss, or batch is a recorded experiment transition, not an exact continuation:

- `--allow-dataset-change`: stage-1/2 resume only; keeps learned/optimizer state and starts the changed dataset from its new cursor.
- `--allow-batch-size-change`: stage-1 resume only; retains progress by sample position.
- `--allow-loss-change`: stage-1 resume only; validates and records an explicit objective transition.

A nonempty output directory cannot be silently reused for a fresh run. Keep independent runtimes for independent experiments.

The optional Linux launchers [launch_stage1.sh](../scripts/remote/launch_stage1.sh) and [launch_stage2.sh](../scripts/remote/launch_stage2.sh) add GPU-UUID locks, free-memory checks, and clean signal handling. Their environment-variable defaults differ from the direct runner; inspect their headers and arguments before using them. They are not a GPU scheduler.

## Evaluate and publish

After cleanly stopping stage 2, evaluate its full checkpoint on a held-out image directory:

```bash
csa evaluate \
  --checkpoint runs/stage2/run/latest.pt \
  --images data/validation --mask-mode random \
  --batch-size 1 --workers 0 --save-limit 8 \
  --output outputs/stage2-evaluation --device cuda:0
```

See [USAGE.md](USAGE.md) for metrics and preprocessing. The package CLI does not load stage-1-only checkpoints as complete two-stage models; use the stage-1 runner's saved validation previews to inspect rough reconstruction. For an indexed stage-2 runtime, [evaluate_stage2_latest.py](../scripts/remote/evaluate_stage2_latest.py) implements the fixed local 64-image protocol used in the release record.

Export a selected **full stage-2 checkpoint** with `csa export` before sharing it; see [CHECKPOINTS.md](CHECKPOINTS.md) for the command and compatibility contract. The portable artifact contains the rough and refinement networks, inference configuration, and public provenance. It has no optimizer or discriminator state and cannot resume training. Keep the full checkpoint privately for continuation, and publish real previews and the evaluation protocol together, including failure cases.
