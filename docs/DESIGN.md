# Implementation principles

This rewrite aims to make the method readable, testable and practical to experiment with.
Using a recent PyTorch release is useful; claiming the implementation is optimal would
require measurements and reproduction experiments that have not been run.

## Native PyTorch, explicit behavior

| Choice | Why it is here |
|---|---|
| Ordinary `nn.Module`, `ModuleList` and tensor operations | Network structure and gradient paths remain inspectable without framework hooks. |
| Native autograd for CSA | Removes the old manual backward, integer attention weights and detached loss paths. |
| Typed dataclass configuration and outputs | Named fields make network boundaries and experiment settings easy to read. |
| Explicit `forward(image, mask)` | Every sample owns its mask; there is no stale global mask cache. |
| `torch.amp` with fp32 CSA calculations | Uses the current AMP API while protecting similarity and recurrent calculations. CUDA behavior still needs hardware validation. |
| `torch.inference_mode()` | Evaluation and inference do not construct training graphs. |
| `zero_grad(set_to_none=True)` and explicit detach boundaries | Makes generator/discriminator updates and gradient ownership clear. |
| Frozen VGG parameters, differentiable fake-image path | Feature supervision can update the generator without optimizing VGG. |
| `torch.load(..., weights_only=True)` and atomic checkpoint replacement | Loads supported state types and prevents a partial write replacing the last complete checkpoint. |
| `uv.lock` and a small `pyproject.toml` | Versions are resolved once and can be installed consistently with `uv sync --locked --extra dev`. |

The AMP API choice follows the [PyTorch AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html);
the dependency workflow follows the [uv locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

## Experiments are configuration, not code edits

```bash
uv run --locked csa train --config configs/default.yaml \
  --set data.batch_size=4 train.lr=2e-4 train.output=runs/batch4 \
  --dry-run
```

`--dry-run` resolves and checks configuration without opening a dataset, downloading VGG,
creating an output directory or taking an optimizer step. Unknown fields, incorrect types,
nonfinite values and invalid dimensions fail before training.

Removing `--dry-run` starts the experiment after data paths have been supplied.
The saved config records the effective values. Each invocation also records runtime versions,
the selected device, source file hashes and the available Git revision. Hashes matter because
research work often runs before local changes are committed.

A nonempty output directory can only be continued from its own `latest.pt`; starting a new
experiment requires a different directory. Checkpoints include optimization and RNG state,
and resume validation rejects changes to the training setup that would silently change the run.
Checkpoint metadata distinguishes ImageNet VGG from explicitly requested diagnostic features.
A trainer created for restore cannot update or save until the restore succeeds; partially
restored state is not allowed to continue. Restoring saved CUDA RNG states across hosts with
different GPU counts is rejected explicitly.

## Deliberately not claimed

- **No claim of fastest CSA.** Search is chunked, but generation still follows the paper's
  sequential dependency and runs a Python recurrence for each sample.
- **No blanket `torch.compile` switch.** Dynamic hole counts, indexing and recurrence need
  profiling and compiler-specific validation before such a switch is useful.
- **No substitute attention kernel.** Standard scaled dot-product attention has different
  semantics from hard nearest-neighbor search followed by recurrent patch generation.
- **No DDP, FSDP or fused-optimizer performance claim.** These need appropriate hardware,
  gradient-equivalence checks and measurements, especially for relativistic batch means.
- **No claim of recovered paper quality.** Stable gradients and passing tests establish
  software behavior, not dataset convergence or the published quantitative results.

The original method, numerical changes and topology decisions are described in
[MIGRATION.md](MIGRATION.md). Actual checks and their limits are in
[VALIDATION.md](VALIDATION.md).
