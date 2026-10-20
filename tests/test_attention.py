import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from csa_inpainting.attention import CoherentSemanticAttention
from csa_inpainting.masks import feature_mask


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
