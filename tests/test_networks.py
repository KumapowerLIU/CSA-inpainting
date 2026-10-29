from dataclasses import replace

import pytest
import torch

from csa_inpainting.diagnostics import TinyFeatures
from csa_inpainting.losses import consistency, relativistic_generator
from csa_inpainting.networks import (
    FeaturePatchDiscriminator,
    InpaintingModel,
    VGGFeatures,
)


def test_model_shapes_and_independent_batch(batch):
    image, mask = batch
    model = InpaintingModel(width=4, depth=6).eval()
    output = model(image, mask)
    assert output.refined.shape == image.shape
    assert output.encoder_features.shape == (2, 32, 8, 8)
    assert output.decoder_features.shape == (2, 32, 8, 8)
    assert torch.equal(output.completed[~mask.expand_as(image)], image[~mask.expand_as(image)])
    separate = torch.cat([model(image[i : i + 1], mask[i : i + 1]).refined for i in range(2)])
    torch.testing.assert_close(output.refined, separate, atol=2e-5, rtol=2e-4)


def test_refinement_gradient_is_isolated_from_rough_stage(batch):
    model = InpaintingModel(width=4, depth=6)
    output = model(*batch)
    output.refined.square().mean().backward()
    assert all(p.grad is None for p in model.rough.parameters())
    assert sum(p.grad.abs().sum() for p in model.refine.parameters() if p.grad is not None) > 0


def test_rough_gradient_comes_only_from_its_own_l1(batch):
    model = InpaintingModel(width=4, depth=6)
    output = model(*batch)
    parameters = tuple(model.rough.parameters())
    own_loss = torch.nn.functional.l1_loss(output.rough, batch[0])
    own_gradients = torch.autograd.grad(own_loss, parameters, retain_graph=True)
    target = TinyFeatures(32)(batch[0])["relu4_3"]
    combined_loss = own_loss + output.refined.square().mean() + consistency(output, target)
    combined_gradients = torch.autograd.grad(combined_loss, parameters)
    for own, combined in zip(own_gradients, combined_gradients, strict=True):
        torch.testing.assert_close(own, combined, atol=0, rtol=0)


def test_consistency_updates_encoder_and_decoder(batch):
    model = InpaintingModel(width=4, depth=6)
    output = model(*batch)
    output.encoder_features.retain_grad()
    output.decoder_features.retain_grad()
    target = TinyFeatures(32)(batch[0])["relu4_3"]
    consistency(output, target).backward()
    assert all(p.grad is None for p in model.rough.parameters())
    for features in (output.encoder_features, output.decoder_features):
        assert features.grad.abs().sum() > 0
    # Decoder supervision also propagates through the encoder's known features.
    # Test direct loss masking on independent leaves, not on that full graph.
    detached = replace(
        output,
        encoder_features=output.encoder_features.detach().requires_grad_(),
        decoder_features=output.decoder_features.detach().requires_grad_(),
    )
    consistency(detached, target).backward()
    for features in (detached.encoder_features, detached.decoder_features):
        assert torch.count_nonzero(features.grad[~output.feature_mask.expand_as(features)]) == 0


def test_feature_adversarial_path_reaches_generator(batch):
    model = InpaintingModel(width=4, depth=6)
    output = model(*batch)
    features = TinyFeatures(32)
    discriminator = FeaturePatchDiscriminator(32, 16).requires_grad_(False)
    fake = discriminator(features(output.refined)["pool3"])
    with torch.no_grad():
        real = discriminator(features(batch[0])["pool3"])
    relativistic_generator(real, fake).backward()
    assert model.refine.head[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.rough.parameters())


def test_real_vgg_indices_normalization_and_input_gradients():
    # Random VGG weights only for interface testing; no network download in tests.
    vgg = VGGFeatures(pretrained=False)
    image = torch.rand(1, 3, 64, 64, requires_grad=True)
    output = vgg(image)
    assert output["pool3"].shape == (1, 256, 8, 8)
    assert output["relu4_3"].shape == (1, 512, 8, 8)
    normalized = ((image + 1) / 2 - vgg.mean) / vgg.std
    torch.testing.assert_close(output["pool3"], vgg.layers[:17](normalized))
    torch.testing.assert_close(output["relu4_3"], vgg.layers(normalized))
    output["pool3"].mean().backward()
    assert image.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in vgg.parameters())


def test_vgg_pool3_early_stop_preserves_values_and_input_gradients():
    vgg = VGGFeatures(pretrained=False)
    image = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_()
    reference = image.detach().clone().requires_grad_()
    later_calls = []
    hook = vgg.layers[17].register_forward_hook(lambda *_: later_calls.append(True))
    early = vgg.pool3_features(image)
    assert later_calls == []
    complete = vgg(reference)["pool3"]
    assert later_calls == [True]
    hook.remove()
    torch.testing.assert_close(early, complete, atol=0, rtol=0)
    probe = torch.randn_like(early)
    early_gradient = torch.autograd.grad((early * probe).sum(), image)[0]
    complete_gradient = torch.autograd.grad((complete * probe).sum(), reference)[0]
    torch.testing.assert_close(early_gradient, complete_gradient, atol=0, rtol=0)


def test_256_topology_and_rectangular_inputs():
    model = InpaintingModel(width=4, depth=8).eval()
    with torch.inference_mode():
        x = torch.zeros(1, 3, 256, 512)
        mask = torch.zeros(1, 1, 256, 512, dtype=torch.bool)
        mask[:, :, 80:160, 160:240] = True
        result = model(x, mask)
    assert result.refined.shape == x.shape
    assert result.encoder_features.shape[-2:] == (32, 64)
    with pytest.raises(ValueError, match="multiples"):
        model(torch.zeros(1, 3, 255, 256), mask[..., :255, :256])
