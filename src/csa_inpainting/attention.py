"""CSA search and recurrent generation (paper equations 1–3).

No learned parameters, CUDA-only allocations, shared mask state or manual backward.
The discrete nearest-neighbor index is piecewise constant; selected similarities,
source values and recurrent mixing weights retain their autograd graph.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .masks import validate_mask


class CoherentSemanticAttention(nn.Module):
    def __init__(self, chunk_size: int = 256, eps: float = 1e-8):
        super().__init__()
        if chunk_size < 1 or eps <= 0:
            raise ValueError("chunk_size and eps must be positive")
        self.chunk_size = chunk_size
        self.eps = eps

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 4 or not features.is_floating_point():
            raise ValueError("CSA expects floating B x C x H x W features")
        mask = validate_mask(mask, features)
        # Similarity and recurrence run in fp32 under AMP; retain fp64 for gradcheck.
        with torch.autocast(device_type=features.device.type, enabled=False):
            working = features if features.dtype == torch.float64 else features.float()
            samples = list(working.unbind(0))
            active = []
            queries, references, scores = [], [], []
            for index, (sample, missing) in enumerate(zip(samples, mask.unbind(0), strict=True)):
                # N x C, row-major order; search only this sample's known region.
                values = sample.flatten(1).T
                holes = missing.flatten().nonzero(as_tuple=True)[0]
                if holes.numel() == 0:
                    continue
                known = (~missing.flatten()).nonzero(as_tuple=True)[0]
                if known.numel() == 0:
                    raise ValueError(
                        "CSA needs at least one known feature location in every sample"
                    )
                query_unit = F.normalize(values[holes], dim=1, eps=self.eps)
                sources = values[known]
                source_unit = F.normalize(sources, dim=1, eps=self.eps)
                # The discrete search needs no graph. Recompute selected scores
                # below so source values and recurrent weights remain differentiable.
                with torch.no_grad():
                    nearest = torch.cat(
                        [
                            (part @ source_unit.T).argmax(dim=1)
                            for part in query_unit.split(self.chunk_size)
                        ]
                    )
                queries.append(query_unit)
                references.append(sources[nearest])
                scores.append((query_unit * source_unit[nearest]).sum(dim=1))
                active.append((index, values, holes))

            if active:
                # Tail padding keeps each sample's recurrence independent. The
                # time-major layout batches each step over samples; unbind once
                # avoids per-step full-tensor SelectBackward allocations.
                query_steps = pad_sequence(queries).unbind(0)
                reference_steps = pad_sequence(references).unbind(0)
                score_steps = pad_sequence(scores).unbind(0)
                generated = [reference_steps[0]]
                for query, reference, score in zip(
                    query_steps[1:], reference_steps[1:], score_steps[1:], strict=True
                ):
                    previous = generated[-1]
                    adjacent = (query * F.normalize(previous, dim=-1, eps=self.eps)).sum(dim=-1)
                    # Preserve the existing signed-feature stability rule: clip
                    # negative similarities and fall back if both scores vanish.
                    a, b = adjacent.clamp_min(0), score.clamp_min(0)
                    total = a + b
                    alpha = torch.where(total > self.eps, a / total.clamp_min(self.eps), 0.0)
                    alpha = alpha.unsqueeze(-1)
                    generated.append(alpha * previous + (1 - alpha) * reference)
                per_sample = torch.stack(generated, dim=0).unbind(1)
                for (index, values, holes), replacement in zip(active, per_sample, strict=True):
                    restored = values.index_copy(0, holes, replacement[: holes.numel()])
                    samples[index] = restored.T.reshape_as(samples[index])
            output = torch.stack(samples)
        return output.to(features.dtype)
