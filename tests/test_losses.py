"""Analytic contracts for RaLSGAN and masked consistency supervision."""

import pytest
import torch

from csa_inpainting.losses import (
    consistency,
    relativistic_discriminator,
    relativistic_generator,
)
from csa_inpainting.networks import InpaintingOutput


def test_ralsgan_known_values_and_analytic_gradients():
    # Unequal real/fake counts make accidental per-pair subtraction detectable.
    # These rational values follow from independently differentiating the
    # population means in RaLSGAN, including the fake mean in G's real term.
    real = torch.tensor([1.0, 2.0], requires_grad=True)
    fake = torch.tensor([-1.0, 0.0, 0.5], requires_grad=True)
    d_loss = relativistic_discriminator(real, fake)
    d_real, d_fake = torch.autograd.grad(d_loss, (real, fake))
    assert d_loss.item() == pytest.approx(55 / 72)
    torch.testing.assert_close(d_real, torch.tensor([5 / 12, 11 / 12]))
    torch.testing.assert_close(d_fake, torch.tensor([-13 / 18, -7 / 18, -2 / 9]))

    g_loss = relativistic_generator(real.detach(), fake)
    g_fake = torch.autograd.grad(g_loss, fake)[0]
    assert g_loss.item() == pytest.approx(535 / 72)
    # G's first (real-data) term contributes -8/9 to EVERY fake gradient;
    # the second term contributes [-7/6, -5/6, -2/3]. Detaching the first
    # relative score would silently discard almost half of this signal.
    expected = torch.tensor([-7 / 6, -5 / 6, -2 / 3]) - 8 / 9
    torch.testing.assert_close(g_fake, expected)


def test_ralsgan_opposes_discriminator_and_generator_at_equal_scores():
    real = torch.zeros(4, requires_grad=True)
    fake = torch.zeros(4, requires_grad=True)
    d_real, d_fake = torch.autograd.grad(relativistic_discriminator(real, fake), (real, fake))
    g_fake = torch.autograd.grad(relativistic_generator(real.detach(), fake), fake)[0]
    # Gradient descent raises real scores and lowers fake scores for D;
    # G instead raises fake scores. Neither uses a sigmoid probability.
    torch.testing.assert_close(d_real, torch.full_like(real, -0.5))
    torch.testing.assert_close(d_fake, torch.full_like(fake, 0.5))
    torch.testing.assert_close(g_fake, torch.full_like(fake, -0.5))


def test_ralsgan_fake_sample_gradient_depends_on_other_samples():
    real = torch.zeros(2)
    fake = torch.zeros(2, requires_grad=True)
    before = torch.autograd.grad(relativistic_generator(real, fake), fake)[0]
    changed = torch.tensor([0.0, 2.0], requires_grad=True)
    after = torch.autograd.grad(relativistic_generator(real, changed), changed)[0]
    # Sample zero is unchanged; only sample one's score changed. The other
    # sample must still affect sample zero through the relativistic average.
    assert before[0].item() == pytest.approx(-1.0)
    assert after[0].item() == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("loss_function", "expected"),
    [(relativistic_discriminator, 358801.0), (relativistic_generator, 361201.0)],
)
def test_ralsgan_reduces_in_float32_before_squaring_half_logits(loss_function, expected):
    real = torch.full((2, 1, 2, 2), 300.0, dtype=torch.float16, requires_grad=True)
    fake = torch.full_like(real, -300.0, requires_grad=True)
    loss = loss_function(real, fake)
    # 599**2 and 601**2 overflow fp16, although both logits and their
    # gradients are representable. Casting only after squaring is too late.
    assert loss.dtype == torch.float32
    assert loss.item() == expected
    gradients = torch.autograd.grad(loss, (real, fake))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_consistency_uses_whole_grid_mean_and_hole_only_direct_gradients():
    encoder = torch.ones(1, 2, 2, 2, requires_grad=True)
    decoder = torch.full_like(encoder, 2.0, requires_grad=True)
    target = torch.zeros_like(encoder)
    mask = torch.tensor([[[[True, False], [False, False]]]])
    image = torch.zeros(1, 3, 2, 2)
    output = InpaintingOutput(image, image, image, image, encoder, decoder, mask)
    loss = consistency(output, target)
    # One quarter of the grid is missing: (1**2 + 2**2) / 4.
    # Dividing by only the number of hole pixels would incorrectly yield 5.
    assert loss.item() == 1.25
    loss.backward()
    expected_encoder = torch.zeros_like(encoder)
    expected_decoder = torch.zeros_like(decoder)
    expected_encoder[:, :, 0, 0] = 0.25
    expected_decoder[:, :, 0, 0] = 0.5
    torch.testing.assert_close(encoder.grad, expected_encoder)
    torch.testing.assert_close(decoder.grad, expected_decoder)
