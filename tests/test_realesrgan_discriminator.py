import copy
import hashlib

import pytest
import torch

from csa_inpainting.discriminators import (
    REALESRGAN_D_ARCHITECTURE,
    UNetDiscriminatorSN,
    load_pretrained_discriminator,
    validate_discriminator_provenance,
)
from csa_inpainting.perceptual import _weights_sha256


def checkpoint(tmp_path, module):
    path = tmp_path / "netD.pth"
    torch.save({"params": copy.deepcopy(module.state_dict())}, path)
    return path


def test_rgb_dense_raw_logits_and_input_gradients():
    model = UNetDiscriminatorSN(num_feat=4)
    image = torch.rand(2, 3, 32, 40, requires_grad=True)
    # Exact constant outside [0, 1] demonstrates no final sigmoid or clipping.
    with torch.no_grad():
        model.conv9.weight.zero_()
        model.conv9.bias.fill_(-2)
    output = model(image)
    assert output.shape == (2, 1, 32, 40)
    torch.testing.assert_close(output, torch.full_like(output, -2), rtol=0, atol=0)
    with torch.no_grad():
        model.conv9.weight.fill_(0.02)
    model(image).square().mean().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_upstream_state_key_contract():
    model = UNetDiscriminatorSN(num_feat=4)
    expected = {"conv0.weight", "conv0.bias", "conv9.weight", "conv9.bias"}
    expected |= {
        f"conv{i}.{name}" for i in range(1, 9) for name in ("weight_orig", "weight_u", "weight_v")
    }
    assert set(model.state_dict()) == expected
    assert not any(
        isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d))
        for layer in model.modules()
    )


def test_strict_roundtrip_preserves_spectral_buffers_and_outputs(tmp_path):
    model = UNetDiscriminatorSN(num_feat=4)
    image = torch.rand(2, 3, 32, 32)
    model(image)  # Initialize the spectral estimates before checking eval output.
    model.eval()
    saved = checkpoint(tmp_path, model)
    restored = UNetDiscriminatorSN(num_feat=4).eval()
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    provenance = load_pretrained_discriminator(
        restored, saved, expected_sha256=digest, source="fixture://netD"
    )
    assert provenance["file_sha256"] == digest
    assert provenance["initial_state_sha256"] == _weights_sha256(model.state_dict())
    assert provenance["architecture"] == REALESRGAN_D_ARCHITECTURE
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)
    validate_discriminator_provenance(provenance, restored)


@pytest.mark.parametrize("corruption", ["key", "shape", "dtype", "nan", "wrapper", "sha"])
def test_corrupt_asset_rejected_without_partial_load(tmp_path, corruption):
    source = UNetDiscriminatorSN(num_feat=4)
    target = UNetDiscriminatorSN(num_feat=4)
    before = copy.deepcopy(target.state_dict())
    state = {"params": copy.deepcopy(source.state_dict())}
    if corruption == "key":
        del state["params"]["conv3.weight_u"]
    elif corruption == "shape":
        state["params"]["conv4.weight_orig"] = torch.ones(1)
    elif corruption == "dtype":
        state["params"]["conv3.weight_orig"] = state["params"]["conv3.weight_orig"].double()
    elif corruption == "nan":
        state["params"]["conv9.bias"].fill_(float("nan"))
    elif corruption == "wrapper":
        state = {"params_ema": state["params"]}
    path = tmp_path / "corrupt.pth"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_pretrained_discriminator(
            target, path, expected_sha256="0" * 64 if corruption == "sha" else None
        )
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_feat", 64),
        ("input_range", [-1, 1]),
        ("architecture", "patch"),
        ("param_key", "params_ema"),
        ("file_sha256", "invalid"),
        ("initial_state_sha256", "x" * 64),
        ("source", ""),
    ],
)
def test_tampered_initialization_provenance_rejected(tmp_path, field, value):
    model = UNetDiscriminatorSN(num_feat=4)
    provenance = load_pretrained_discriminator(model, checkpoint(tmp_path, model))
    provenance[field] = value
    with pytest.raises(ValueError, match="provenance"):
        validate_discriminator_provenance(provenance, model)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 3, 31, 32), (1, 3, 4, 8)])
def test_invalid_image_contract(shape):
    with pytest.raises(ValueError):
        UNetDiscriminatorSN(num_feat=4)(torch.rand(shape))
