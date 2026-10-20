"""CSA search and recurrent generation (paper equations 1–3).

No learned parameters, CUDA-only allocations, shared mask state or manual backward.
The discrete nearest-neighbor index is piecewise constant; selected similarities,
source values and recurrent mixing weights retain their autograd graph.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .masks import validate_mask


class CoherentSemanticAttention(nn.Module):
    def __init__(self, chunk_size: int = 256, eps: float = 1e-8):
        super().__init__()
        if chunk_size < 1 or eps <= 0:
            raise ValueError("chunk_size and eps must be positive")
        self.chunk_size = chunk_size
        self.eps = eps

    def _sample(self, features: Tensor, missing: Tensor) -> Tensor:
        # N x C, row-major order. Keep the known region exactly unchanged.
        values = features.flatten(1).T
        holes = missing.flatten().nonzero(as_tuple=True)[0]
        known = (~missing.flatten()).nonzero(as_tuple=True)[0]
        if holes.numel() == 0:
            return features
        if known.numel() == 0:
            raise ValueError("CSA needs at least one known feature location in every sample")
        queries, sources = values[holes], values[known]
        query_unit = F.normalize(queries, dim=1, eps=self.eps)
        source_unit = F.normalize(sources, dim=1, eps=self.eps)

        # Argmax selection needs no graph. Recompute only selected cosine scores
        # with gradients; this avoids saving the full H*W x H*W search graph.
        with torch.no_grad():
            nearest = torch.cat(
                [(q @ source_unit.T).argmax(dim=1) for q in query_unit.split(self.chunk_size)]
            )
        references = sources[nearest]
        best = (query_unit * source_unit[nearest]).sum(dim=1)

        generated = [references[0]]
        for i in range(1, holes.numel()):
            previous = generated[-1]
            adjacent = (query_unit[i] * F.normalize(previous, dim=0, eps=self.eps)).sum()
            # Eq. 3 is undefined for cancelling/zero scores and can extrapolate
            # without bound for signed activations. Use nonnegative similarities
            # and fall back to the selected source when both offer no evidence.
            a, b = adjacent.clamp_min(0), best[i].clamp_min(0)
            total = a + b
            alpha = torch.where(total > self.eps, a / total.clamp_min(self.eps), 0.0)
            generated.append(alpha * previous + (1 - alpha) * references[i])
        result = values.index_copy(0, holes, torch.stack(generated))
        return result.T.reshape_as(features)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 4 or not features.is_floating_point():
            raise ValueError("CSA expects floating B x C x H x W features")
        mask = validate_mask(mask, features)
        # Similarity and recurrence run in fp32 under AMP; retain fp64 for gradcheck.
        with torch.autocast(device_type=features.device.type, enabled=False):
            working = features if features.dtype == torch.float64 else features.float()
            output = torch.stack(
                [self._sample(sample, hole) for sample, hole in zip(working, mask, strict=True)]
            )
        return output.to(features.dtype)
