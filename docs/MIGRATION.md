# Migration notes

This modern implementation is based on the author's repository at commit [`f0b0712aa950cc559062dfef04c2f3f884a1f3e2`](https://github.com/KumapowerLIU/CSA-inpainting/tree/f0b0712aa950cc559062dfef04c2f3f884a1f3e2), with the [ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.html) as the method reference. Original Python source and historical figures remain in `legacy/`, outside active imports, packaging, and tests. Historical notebooks and bytecode are not part of the modern working tree; the original Git revision remains available.

The modern trained release is not a numerically identical port or a paper-equivalent reproduction. This document records the important boundaries.

## Original-code audit

| Original behavior | Modern implementation |
|---|---|
| `mask[0][0]` and feature-mask utilities effectively force one sample's mask | Explicit `B × 1 × H × W` masks; feature masks are computed per sample |
| Cached masks can outlive the input that created them | Mask passed explicitly on every forward |
| Floating CSA recurrence weights stored in integer tensors | Floating-point native autograd |
| Custom backward adds an identity gradient; rectangular indexing uses `i*h+j` | Remove custom backward; correct rectangular spatial indexing |
| Dynamic convolution construction, `.data`, and `Variable` | Tensor search/gather and input-following device placement |
| Coarse output inserted using `.data`, mutating inputs | Explicit `rough.detach()` and non-mutating `torch.where` composition |
| Consistency loss wrapped from `gl.loss.data`, disconnecting the graph | Explicit encoder/decoder features and differentiable consistency loss |
| Fake VGG features created from detached images during G training | Detach for D; preserve fake-image gradients through frozen VGG for G |
| VGG omits ImageNet normalization and calls pool3 `relu3_3` | Correct RGB normalization and explicit `pool3`/`relu4_3` outputs |
| Checkpoints primarily contain network weights | Versioned model, optimizer, scheduler, scaler, RNG, and progress states |
| GAN mode names suggest several objectives while the formula is RaLSGAN | Explicit relativistic-average least-squares losses |

These fixes preserve deliberate gradient boundaries while removing accidental disconnections. They also change optimization behavior, so software correctness checks alone cannot establish equivalent final quality.

## Architecture and intentional differences

The rough model keeps the eight-level pix2pix-style U-Net: 4 × 4 stride-2 convolutions, InstanceNorm, skip connections, and a `tanh` output. Refinement retains the stem, dilated 4 × 4 downsampling, and 3 × 3 channel-convolution design. CSA operates on 512-channel features at the third downsampling level (256 → 32), before InstanceNorm. The corresponding 512-channel decoder features participate in consistency supervision. InstanceNorm is omitted at the deepest 1 × 1 level.

The output activation is versioned: **modern joint training has a linear refinement head; sequential stage 2 has a `tanh` refinement head**. Loading a checkpoint must not silently switch between them.

A further known difference is the deepest refinement downsampling: the modern implementation uses ordinary 4 × 4, stride 2, padding 1, while the original innermost layer uses dilation 2 and padding 3. Parameters and output shapes match, but sampling locations differ. On the default 2 × 2 input, the original samples only the bottom-right location while the modern layer sees all four. This difference is retained for the existing modern checkpoint semantics; it has not been proven to cause a particular quality failure. A future change requires a separately versioned experiment.

Activations avoid in-place mutation, including the historical recursive skip-input side effects. “Preserved structure” therefore refers to the broad layer/channel/connection design, not every incidental behavior of the old implementation.

The image discriminator retains the original 70 × 70 PatchGAN. The feature discriminator follows the released source: VGG pool3 features (256 × 32 × 32 at the training resolution), then three stride-2 4 × 4 convolutions with 512 channels. The paper's appendix Table 5 differs from the released source in VGG layer/stride labels; this implementation explicitly follows the source topology.

The fill color remains `[123, 104, 117] / 127.5 - 1`, interpreted as RGB.

## CSA math and numerical behavior

1. Each sample independently searches known 1 × 1 feature patches.
2. Similarity uses the bilateral L2-normalized cosine in paper Eq. 1. The original source normalized only known patches; the selected argmax is usually unchanged, but its scores and recurrence weights need not match.
3. Holes are processed in row-major order. The first hole takes the best known patch. Later holes combine their best patch with the preceding generated patch using the current original query's similarity.
4. Hard argmax indices are discrete. Gradients flow through selected scores, source values, and recurrence weights, without an extra identity straight-through term.
5. **Numerical stabilization:** because linear features may have negative cosine scores, the original ratio can diverge when its denominator approaches zero. The modern recurrence clamps both similarities to nonnegative values and falls back to the best source patch when their sum is near zero. Positive-score cases follow Eq. 3; the negative-score policy is an explicit engineering choice, not the paper's unchanged formula.
6. Empty feature holes produce an identity mapping. Completely missing feature maps are rejected because no source context exists. Very thin image masks can disappear after feature-mask downsampling; image-level objectives still apply in that case.

Search indices are computed in chunks under `no_grad`, then selected similarities are recomputed with gradients. This avoids retaining a full pairwise-search autograd graph. Batch recurrence pads only the ends of per-sample hole sequences; padded positions cannot affect real holes or other samples. Known positions are preserved.

Float64 gradient checks cover differentiable, non-tie/non-clamp-boundary cases. Batched CUDA reductions can change FP32 rounding, and BF16 may cross adjacent quantization levels. Checks distinguish underlying FP32 error from dtype tolerance. Kernel-level improvements are not evidence of an end-to-end training speedup.

## Loss conventions and the joint trainer

- Reconstruction is full-image mean L1.
- Consistency masks the encoder/decoder error on both sides, then takes mean MSE over the full feature grid. This removes the old known-region constant offset from masking only predictions. It retains the code's mean convention rather than converting the paper's written sums into an unqualified scale change.
- Joint-training defaults are `1 × reconstruction + 0.01 × consistency + 0.002 × GAN`. The old notebook used `100, 1, 0.2`: the ratios match but the total scale differs, and Adam is not strictly invariant to that change under all numerical conditions.
- G sums image and feature adversarial terms; D averages them.
- VGG weights are frozen; generated images retain gradients through VGG, and target features use `no_grad`.
- The original master detaches rough predictions before refinement. The joint trainer preserves this boundary: rough receives its own reconstruction gradient, while refinement losses do not update rough.
- Rough and refinement have separate Adam optimizers and are both updated within each joint batch. This is distinct from training rough to a selected checkpoint before freezing it for stage 2.
- InstanceNorm is sample-independent, but RaLSGAN means intentionally span the batch/patch predictions. Increasing batch size changes the optimization trajectory.

## Sequential stage 2 and the released checkpoint

`Stage2Trainer` and `Stage2InpaintingModel` use the separate `stage2_v1` checkpoint family:

- `--init-stage1` strictly loads rough weights and fill color from a full stage-1 checkpoint. Rough remains frozen, in evaluation mode, under `no_grad`. Refinement and discriminators start fresh.
- The refinement head uses `tanh`, with LPIPS receiving `[-1,1]` RGB directly; no extra clamp or double ImageNet normalization is added before LPIPS.
- The release objective is `0.1 × L1(refined, target) + LPIPS(completed, target) + 0.01 × consistency + 0.002 × (image GAN + feature GAN)`.
- VGG/LPIPS activations are processed in FP32 chunks with recomputation. Full-batch outputs are combined before RaLSGAN means, preserving the configured optimizer batch.
- Stage-2 counters, optimizers, and sample cursor start from zero. Parent checkpoint hash, step, epoch, and semantics are recorded; hashing and loading refer to the same opened checkpoint.
- A full checkpoint contains rough, refinement, discriminators, perceptual weights, optimizer/scheduler/scaler states, RNG, and sample progress. Resume is self-contained and validates perceptual provenance instead of downloading replacement weights.
- Temporary D parameter-freezing flags are restored even on exceptions. A failed partial update prevents continued training or saving an inconsistent state.

The release's image D is the **original PatchGAN**, not a pretrained Real-ESRGAN discriminator. LPIPS is a modern addition, not a loss from the original 2019 paper. See [TRAINING.md](TRAINING.md) for the actual staged commands.

## Retained Real-ESRGAN experiment

A separately versioned experimental path supports `--image-discriminator realesrgan_unet_sn`. It initializes the image discriminator from the official Real-ESRGAN U-Net/SN weights, restarts stage 2 from stage 1, and keeps CSA's RaLSGAN objective. It does not reproduce Real-ESRGAN training.

Only that image D maps RGB from `[-1,1]` to `[0,1]`. Spectral-normalization state is updated once in FP32 before each D/G phase and held fixed across chunks and activation recomputation. Checkpoints record separate semantics, initialization hashes, and normalization policy. Resume does not require the original pretrained file.

The official architecture's BSD-3-Clause notice remains in [discriminators.py](../src/csa_inpainting/discriminators.py). The verified initialization source is [RealESRGAN_x4plus_netD.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth), SHA-256 `afc291719bf0a8bd25ba2b4afbbc5fc0634382dac111d3e1d3a7240d32dfd33a`. This historical experiment is separate from the step-97,397 PatchGAN release.

## Reproducibility and compatibility

Joint format 2 records `rough_detached_separate_adam_v1`; older modern format 1 remains available for inference but cannot claim exact resume under the newer joint semantics. Sequential stage-1 and stage-2 files have their own format and validation rules. Portable `csa_inference_v1` packages retain inference semantics and sanitized provenance but cannot resume training; see [CHECKPOINTS.md](CHECKPOINTS.md).

Training augmentation derives from seed/epoch/index, and shuffle from seed plus epoch. The joint CLI saves at epoch boundaries. The indexed staged runner additionally saves between batches and on clean SIGINT/SIGTERM, with sample-cursor resume. Data identity and configuration must remain compatible; explicitly allowed dataset/loss/batch changes are recorded transitions, not exact replay.

Same-environment resume checks do not guarantee cross-device or cross-version bitwise equality. DDP, gradient accumulation, and `torch.compile` are not implemented as supported training modes. The lockfile records a development baseline, while package dependency ranges permit other compatible installations.

There is no validated converter for original 2019 weights. Loading mismatched parameter names with `strict=False` is not a successful migration. The current trained checkpoint, numerical tests, and visible predictions establish a working modern implementation; they do not remove the deliberate method differences documented here.
