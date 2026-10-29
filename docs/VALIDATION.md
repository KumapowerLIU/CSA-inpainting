# Release validation · 2026-10-29

This record concerns the modern stage-2 checkpoint at **97,397** and the portable
inference release. The [2026-10-20 diagnostic record](VALIDATION_2026-10-20.md)
is retained as history; its earlier absence of CUDA/dataset validation does not
describe the current release.

## Checkpoint and production-runtime checks

| Check | Result |
|---|---|
| Source/portable generator tensors | Identical SHA256 for every tensor, dtype, name, and shape |
| Eight real images, full versus portable model | Bitwise-equal completed outputs with deterministic GPU algorithms |
| Known-pixel composition | Exact on all eight cases |
| Real CLI example | Passed; at most one 8-bit level from the earlier preview |
| Production Python 3.10.14 / Torch 2.6.0+cu124 | 21 release tests passed |
| H100 FP32 export verification | Passed; no VGG/LPIPS downloads |
| Source semantics | Training-snapshot and release model/attention definitions match |
| Weight artifact | 528,510,920 bytes; whole-file SHA256 recorded in the model card |

See [export-verification.json](releases/step97397/export-verification.json),
[source-compatibility.json](releases/step97397/source-compatibility.json), and the
[model card](releases/step97397/README.md). Full training state is preserved
separately. The export changes neither weights nor output activation.

Default backend kernels can round differently even on the same device. CSA uses
a discrete nearest-neighbor choice, so small numerical changes can affect local
outputs. Exact forward comparisons above fix deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. They are not a claim of bitwise equality across
hardware, library versions, layouts, or backend settings. The earlier gallery
used the training runtime's default FP32 backend; its PNG differences are
recorded separately from the source-versus-export equality check.

## Package and software checks

The local development checks use Python 3.12 and PyTorch 2.14.1 on macOS arm64.
Ruff lint/format, lockfile consistency, wheel build, editable install, and the
CPU smoke checkpoint round trip pass. The full suite passes **313 tests and 12 subtests**, with 16 CUDA-dependent
tests skipped locally. Results are recorded in
[`validation-release.json`](validation-release.json).

The smoke and unit tests use synthetic fixtures. They establish software
behavior, not restoration quality. The resume regression requires bitwise-identical restored model, optimizer,
scheduler, scaler, and RNG state. It compares subsequent losses and functional
outputs numerically, since CPU backward roundoff in near-zero gradients can be
amplified by Adam without changing the restored state.

The GitHub CI matrix is configured for Python 3.10 / Torch 2.6 and Python 3.12 /
Torch 2.14.1. Configuration is not evidence that a hosted CI run has executed.
README rendering and local links were checked; no repository or weight upload
was performed during the validation itself. The portable release has since been
published at [KumaPower/CSA](https://huggingface.co/KumaPower/CSA).

## Qualitative and metric evidence

The [64-image report](releases/step97397/evaluation.json) and
[eight-case gallery](assets/csa-step97397-full.jpg) use the selected 97,397-step
checkpoint. This is a fixed internal spot check, not an official paper benchmark.
Faces and repeated scene geometry remain difficult. No original-2019 checkpoint
converter, matched paper reproduction, arbitrary-resolution quality claim, or
multi-GPU equivalence is asserted.
