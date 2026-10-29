import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.nn import functional as F

from csa_inpainting.attention import CoherentSemanticAttention
from csa_inpainting.masks import feature_mask, validate_mask


class IndexedCSAReference(CoherentSemanticAttention):
    """Snapshot of the tensor-indexed recurrence before the unbind optimization."""

    def _sample(self, features, missing):
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
            a, b = adjacent.clamp_min(0), best[i].clamp_min(0)
            total = a + b
            alpha = torch.where(total > self.eps, a / total.clamp_min(self.eps), 0.0)
            generated.append(alpha * previous + (1 - alpha) * references[i])
        return values.index_copy(0, holes, torch.stack(generated)).T.reshape_as(features)

    def forward(self, features, mask):
        # Keep the pre-batching execution path independent of the implementation
        # under test; overriding only _sample would no longer be sufficient.
        mask = validate_mask(mask, features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            working = features if features.dtype == torch.float64 else features.float()
            output = torch.stack(
                [self._sample(sample, hole) for sample, hole in zip(working, mask, strict=True)]
            )
        return output.to(features.dtype)


@pytest.mark.parametrize("regime", ["positive", "signed", "zero"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable"),
        ),
    ],
)
def test_batched_recurrence_matches_indexed_forward_and_gradient(regime, dtype, device):
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    generator = torch.Generator().manual_seed(712)
    features = torch.randn(4, 5, 3, 4, dtype=dtype, generator=generator).to(device)
    if regime == "positive":
        features = features.abs() + 0.1
    elif regime == "zero":
        features.zero_()
    features.requires_grad_()
    mask = torch.zeros(4, 1, 3, 4, dtype=torch.bool, device=device)
    mask[0, :, 1:, 1:3] = True  # Contiguous holes.
    mask[1, :, ::2, ::2] = True  # Disconnected holes.
    # Sample 2 is entirely known; sample 3 has just one known source.
    mask[3] = True
    mask[3, :, 0, 0] = False
    actual = CoherentSemanticAttention(chunk_size=2)(features, mask)
    expected = IndexedCSAReference(chunk_size=2)(features, mask)
    # CUDA's batched FP32 reductions need not be bitwise identical. A tiny
    # difference can round to adjacent BF16 values; permit one relative ULP,
    # while checking the underlying FP32 calculation separately below.
    tolerance = {"rtol": torch.finfo(dtype).eps, "atol": 2e-5} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(actual, expected, **tolerance)
    assert torch.equal(actual[~mask.expand_as(features)], features[~mask.expand_as(features)])
    # A linear probe also exercises the backward path when all features are zero.
    probe = torch.randn(actual.shape, dtype=actual.dtype, generator=generator).to(device)
    actual_grad = torch.autograd.grad((actual * probe).sum(), features)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), features)[0]
    torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
    assert torch.isfinite(actual_grad).all()
    if dtype == torch.bfloat16:
        float_features = features.detach().float().requires_grad_()
        float_actual = CoherentSemanticAttention(chunk_size=2)(float_features, mask)
        float_expected = IndexedCSAReference(chunk_size=2)(float_features, mask)
        torch.testing.assert_close(float_actual, float_expected, rtol=2e-5, atol=2e-5)
        float_grad = torch.autograd.grad((float_actual * probe.float()).sum(), float_features)[0]
        reference_grad = torch.autograd.grad(
            (float_expected * probe.float()).sum(), float_features
        )[0]
        torch.testing.assert_close(float_grad, reference_grad, rtol=2e-5, atol=2e-5)


def reference_numpy(features, mask):
    """Independent scalar oracle for paper cosine search and Eq. 3."""
    values = features.reshape(features.shape[0], -1).T.copy()
    hole = np.flatnonzero(mask.ravel())
    known = np.flatnonzero(~mask.ravel())
    result = values.copy()
    previous = None
    for index in hole:
        query = values[index]
        score = [
            np.dot(query, values[j])
            / (max(np.linalg.norm(query), 1e-8) * max(np.linalg.norm(values[j]), 1e-8))
            for j in known
        ]
        best = int(np.argmax(score))
        selected = values[known[best]]
        if previous is None:
            previous = selected
        else:
            adjacent = np.dot(query, previous) / (
                max(np.linalg.norm(query), 1e-8) * max(np.linalg.norm(previous), 1e-8)
            )
            a, b = max(adjacent, 0), max(score[best], 0)
            alpha = a / (a + b) if a + b > 1e-8 else 0
            previous = alpha * previous + (1 - alpha) * selected
        result[index] = previous
    return result.T.reshape(features.shape)


@pytest.mark.parametrize("positive", [False, True])
def test_against_independent_reference(positive):
    features = torch.randn(2, 5, 3, 4, dtype=torch.float64)
    if positive:
        features = features.abs() + 0.1
    mask = torch.zeros(2, 1, 3, 4, dtype=torch.bool)
    mask[0, :, 0:2, 1:3] = True
    mask[1, :, 1:, 2:] = True
    output = CoherentSemanticAttention(chunk_size=2)(features, mask)
    expected = np.stack(
        [reference_numpy(f.numpy(), m.numpy()) for f, m in zip(features, mask, strict=True)]
    )
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        output[~mask.expand_as(features)], features[~mask.expand_as(features)]
    )


def test_gradient_numerically():
    features = (torch.rand(1, 3, 2, 3, dtype=torch.float64) + 0.2).requires_grad_()
    mask = torch.tensor([[[[False, True, True], [False, True, False]]]])
    assert gradcheck(lambda x: CoherentSemanticAttention(2)(x, mask), (features,))


def test_ragged_batch_gradient_numerically():
    features = (torch.rand(3, 3, 2, 3, dtype=torch.float64) + 0.2).requires_grad_()
    mask = torch.tensor(
        [
            [[[False, True, True], [False, True, False]]],
            [[[False, False, False], [False, False, False]]],
            [[[True, False, False], [False, True, False]]],
        ]
    )
    assert gradcheck(lambda x: CoherentSemanticAttention(2)(x, mask), (features,))


def test_batch_output_and_gradient_equal_single_samples():
    x = torch.randn(3, 4, 3, 5, requires_grad=True)
    mask = torch.zeros(3, 1, 3, 5, dtype=torch.bool)
    mask[0, :, :2, :2] = True
    mask[1, :, 1:, 2:] = True
    # Third sample deliberately has no holes.
    attention = CoherentSemanticAttention(3)
    actual = attention(x, mask)
    expected = torch.cat([attention(x[i : i + 1], mask[i : i + 1]) for i in range(3)])
    torch.testing.assert_close(actual, expected)
    batch_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    single_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(batch_grad, single_grad)
    first_grad = torch.autograd.grad(attention(x, mask)[0].sum(), x)[0]
    assert torch.count_nonzero(first_grad[1:]) == 0


def test_chunking_does_not_change_results():
    x = torch.rand(1, 6, 4, 5)
    mask = torch.rand(1, 1, 4, 5) > 0.5
    torch.testing.assert_close(
        CoherentSemanticAttention(1)(x, mask), CoherentSemanticAttention(100)(x, mask)
    )


def test_zero_and_signed_features_are_finite():
    for x in (torch.zeros(2, 3, 3, 3), torch.randn(2, 3, 3, 3)):
        x.requires_grad_()
        mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
        mask[:, :, 1:, 1:] = True
        y = CoherentSemanticAttention()(x, mask)
        y.sum().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_invalid_mask_and_all_missing():
    x = torch.randn(2, 3, 4, 4)
    attention = CoherentSemanticAttention()
    with pytest.raises(ValueError, match="one mask per sample"):
        attention(x, torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError, match="binary"):
        attention(x, torch.full((2, 1, 4, 4), 0.5))
    with pytest.raises(ValueError, match="at least one known"):
        attention(x, torch.ones(2, 1, 4, 4))


def test_mask_matches_original_convolutions():
    mask = torch.randint(2, (3, 1, 64, 80)).float()
    reference = mask
    for _ in range(3):
        reference = torch.nn.functional.conv2d(
            reference, torch.ones(1, 1, 4, 4) / 16, stride=2, padding=1
        )
    torch.testing.assert_close(feature_mask(mask), reference > 5 / 16)
