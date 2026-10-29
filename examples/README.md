# One-image example

`masked.png`, `mask.png`, and `expected.png` are a fixed validation example from
the stage-2 checkpoint at step **97,397**. White mask pixels mark the region to
inpaint; black pixels mark the known image. The input already hides the target
pixels. `expected.png` is the model output, not the original photograph.

```bash
csa infer --checkpoint checkpoints/csa-stage2-step97397.pt \
  --image examples/masked.png --mask examples/mask.png \
  --output output.png --device cpu
```

The reference was generated on an H100 with FP32 inference. Floating-point
differences across devices or PyTorch versions can affect final quantization.
Known pixels are composited from the input. Evaluation details and all eight
displayed cases are in [the release notes](../docs/releases/step97397/README.md).
