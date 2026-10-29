import math
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from test_stage2 import RestorableFakeLPIPS, make_parent, snapshot

from csa_inpainting.config import Config
from csa_inpainting.discriminators import UNetDiscriminatorSN
from csa_inpainting.losses import relativistic_discriminator, relativistic_generator
from csa_inpainting.perceptual import _weights_sha256
from csa_inpainting.stage2 import (
    REALESRGAN_TRAINING_SEMANTICS,
    SPECTRAL_NORM_POLICY,
    TRAINING_SEMANTICS,
    Stage2Trainer,
    load_stage2_checkpoint,
    load_stage2_generator,
)


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=RestorableFakeLPIPS))


@pytest.fixture
def pretrained(tmp_path, tiny_config):
    model = UNetDiscriminatorSN(num_feat=tiny_config.train.discriminator_width)
    # A few upstream training forwards give the diagnostic fixture meaningful
    # spectral vectors, analogous to the actual trained discriminator asset.
    with torch.no_grad():
        for _ in range(3):
            model(torch.rand(1, 3, 16, 16))
    path = tmp_path / "pretrained-d.pt"
    torch.save({"params": model.state_dict()}, path)
    return path


@pytest.fixture
def real_config(tiny_config, pretrained):
    config = Config.from_dict(tiny_config.as_dict())
    config.train.image_discriminator = "realesrgan_unet_sn"
    config.train.image_discriminator_pretrained = str(pretrained)
    config.train.image_discriminator_chunk_size = 1
    config.train.reconstruction_weight = 0.1
    config.train.lpips_weight = 1.0
    config.train.lpips_batch_size = 1
    return config


@pytest.fixture
def ready(tmp_path, tiny_config, real_config, batch):
    parent, parent_path = make_parent(tmp_path, tiny_config, batch)
    trainer = Stage2Trainer(real_config, test_fixture=True)
    trainer.init_from_stage1(parent_path)
    return trainer, parent, parent_path


def buffers(module):
    return {name: value.clone() for name, value in module.named_buffers()}


def test_fresh_pretrained_d_still_starts_from_stage1_only(ready, pretrained):
    trainer, parent, _ = ready
    expected = torch.load(pretrained, weights_only=True)["params"]
    torch.testing.assert_close(trainer.image_d.state_dict(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        trainer.generator.rough.state_dict(), parent.generator.rough.state_dict(), rtol=0, atol=0
    )
    assert trainer.global_step == trainer.completed_epochs == 0
    assert trainer.data_progress is None
    assert not trainer.g_optimizer.state and not trainer.d_optimizer.state
    assert trainer.training_semantics == REALESRGAN_TRAINING_SEMANTICS
    assert trainer.image_d_provenance["initial_state_sha256"] == _weights_sha256(expected)
    assert all(not p.requires_grad for p in trainer.generator.rough.parameters())


def test_pretrained_asset_required_only_for_fresh_initialization(real_config):
    real_config.train.image_discriminator_pretrained = ""
    with pytest.raises(ValueError, match="requires pretrained"):
        Stage2Trainer(real_config, test_fixture=True)
    pending = Stage2Trainer(real_config, test_fixture=True, restoring=True)
    assert not pending._ready
    assert pending.image_d_provenance is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_discriminator", "unknown"),
        ("image_discriminator_chunk_size", 0),
    ],
)
def test_discriminator_config_rejects_invalid_values(tiny_config, field, value):
    setattr(tiny_config.train, field, value)
    with pytest.raises(ValueError):
        tiny_config.validate()


def test_rgb_range_conversion_and_generator_gradient(ready):
    trainer, _, _ = ready
    trainer.image_d.requires_grad_(False)
    trainer._prepare_image_discriminator()
    seen = []
    handle = trainer.image_d.conv0.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    image = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16).requires_grad_()
    before = buffers(trainer.image_d)
    try:
        result = trainer._image_logits(image)
        assert result.shape == (1, 1, 16, 16)
        result.square().mean().backward()
    finally:
        handle.remove()
    for actual in seen:
        torch.testing.assert_close(actual, (image.detach() + 1) / 2, rtol=0, atol=0)
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(p.grad is None for p in trainer.image_d.parameters())
    torch.testing.assert_close(buffers(trainer.image_d), before, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["discriminator", "generator"])
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
def test_chunked_global_ralsgan_matches_full_batch_gradients(real_config, phase, device):
    real_config.train.device = device
    direct_config = Config.from_dict(real_config.as_dict())
    direct_config.train.image_discriminator_chunk_size = 5
    direct = Stage2Trainer(direct_config, test_fixture=True)
    chunked = Stage2Trainer(real_config, test_fixture=True)
    real = torch.rand(3, 3, 16, 16, device=device) * 2 - 1
    fake_a = (torch.rand_like(real) * 2 - 1).requires_grad_()
    fake_b = fake_a.detach().clone().requires_grad_()
    for trainer in (direct, chunked):
        trainer._prepare_image_discriminator()
        if phase == "generator":
            trainer.image_d.requires_grad_(False)
    before_direct, before_chunked = buffers(direct.image_d), buffers(chunked.image_d)
    objective = relativistic_discriminator if phase == "discriminator" else relativistic_generator
    expected = objective(
        direct._image_logits(real, checkpoint_activations=False),
        direct._image_logits(fake_a, checkpoint_activations=False),
    )
    actual = objective(chunked._image_logits(real), chunked._image_logits(fake_b))
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(fake_b.grad, fake_a.grad, rtol=3e-4, atol=2e-6)
    for a, b in zip(direct.image_d.parameters(), chunked.image_d.parameters(), strict=True):
        if phase == "discriminator":
            torch.testing.assert_close(b.grad, a.grad, rtol=3e-4, atol=2e-6)
        else:
            assert a.grad is None and b.grad is None
    torch.testing.assert_close(buffers(direct.image_d), before_direct, rtol=0, atol=0)
    torch.testing.assert_close(buffers(chunked.image_d), before_chunked, rtol=0, atol=0)
    torch.testing.assert_close(before_direct, before_chunked, rtol=0, atol=0)


def test_spectral_updates_happen_only_at_phase_boundaries(ready):
    trainer, _, _ = ready
    before = buffers(trainer.image_d)
    trainer._prepare_image_discriminator()
    prepared = buffers(trainer.image_d)
    assert any(not torch.equal(before[name], value) for name, value in prepared.items())
    real = torch.rand(2, 3, 16, 16) * 2 - 1
    loss = relativistic_discriminator(trainer._image_logits(real), trainer._image_logits(-real))
    loss.backward()
    torch.testing.assert_close(buffers(trainer.image_d), prepared, rtol=0, atol=0)
    trainer.image_d.train()
    with pytest.raises(RuntimeError, match="must be frozen"):
        trainer._image_logits(real)


def test_real_discriminator_full_step_updates_only_trainable_models(ready, batch):
    trainer, _, _ = ready
    rough, d_before = snapshot(trainer.generator.rough), snapshot(trainer.image_d)
    refine_before, feature_before = snapshot(trainer.generator.refine), snapshot(trainer.feature_d)
    for _ in range(2):
        result = trainer.step(*batch)
        assert all(math.isfinite(value) for value in result.values())
        assert result["reconstruction_weighted"] == pytest.approx(0.1 * result["reconstruction"])
        assert result["lpips_weighted"] == result["lpips"]
        expected = (
            0.1 * result["reconstruction"]
            + result["lpips"]
            + trainer.config.train.consistency_weight * result["consistency"]
            + trainer.config.train.adversarial_weight
            * (result["adversarial_image"] + result["adversarial_feature"])
        )
        assert result["generator"] == pytest.approx(expected)
    for old, model in (
        (d_before, trainer.image_d),
        (refine_before, trainer.generator.refine),
        (feature_before, trainer.feature_d),
    ):
        assert any(not torch.equal(old[name], value) for name, value in model.state_dict().items())
    torch.testing.assert_close(snapshot(trainer.generator.rough), rough, rtol=0, atol=0)
    assert all(p.grad is None for p in trainer.generator.rough.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_cuda_bf16_full_step_stays_finite(tmp_path, tiny_config, real_config, batch):
    _, path = make_parent(tmp_path, tiny_config, batch)
    real_config.train.device = "cuda"
    real_config.train.amp = "bf16"
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    trainer = Stage2Trainer(real_config, test_fixture=True)
    trainer.init_from_stage1(path)
    for _ in range(2):
        assert all(math.isfinite(value) for value in trainer.step(*batch).values())
    assert trainer.global_step == 2
    assert all(not module.training for module in trainer.image_d.modules())
    assert all(p.grad is None for p in trainer.generator.rough.parameters())


def test_resume_without_pretrained_or_parent_files_is_exact(
    ready, real_config, tmp_path, pretrained, batch
):
    trainer, _, parent_path = ready
    trainer.step(*batch)
    trainer.finish_epoch()
    trainer.data_progress = {"epoch": 1, "samples_in_epoch": 2, "samples_seen": 4}
    path = tmp_path / "new-stage2.pt"
    trainer.save(path)
    state = load_stage2_checkpoint(path)
    assert state["image_d_spectral_norm_policy"] == SPECTRAL_NORM_POLICY
    assert state["image_d_state_sha256"] == _weights_sha256(trainer.image_d.state_dict())
    pretrained.unlink()
    parent_path.unlink()
    config = Config.from_dict(real_config.as_dict())
    config.train.image_discriminator_pretrained = ""
    restored = Stage2Trainer(config, restoring=True, test_fixture=True)
    restored.restore(path)
    assert torch.equal(torch.get_rng_state(), state["rng_cpu"])
    assert random.getstate() == state["rng_python"]
    assert restored.image_d_provenance == trainer.image_d_provenance
    assert restored.data_progress == trainer.data_progress
    model, _ = load_stage2_generator(path, torch.device("cpu"))
    with torch.inference_mode():
        torch.testing.assert_close(
            model(*batch).completed, trainer.generator(*batch).completed, rtol=0, atol=0
        )
    expected, actual = trainer.step(*batch), restored.step(*batch)
    assert actual == expected
    for name in (
        "generator",
        "image_d",
        "feature_d",
        "features",
        "perceptual",
        "g_optimizer",
        "d_optimizer",
    ):
        torch.testing.assert_close(
            getattr(restored, name).state_dict(),
            getattr(trainer, name).state_dict(),
            rtol=0,
            atol=0,
        )
    for name in ("g_scheduler", "d_scheduler", "scaler"):
        assert getattr(restored, name).state_dict() == getattr(trainer, name).state_dict()


@pytest.mark.parametrize(
    "change",
    [
        "semantics",
        "architecture",
        "sn_policy",
        "weight",
        "hash",
        "source",
        "unknown_key",
    ],
)
def test_checkpoint_discriminator_identity_is_strict(ready, tmp_path, change):
    trainer, _, _ = ready
    path = tmp_path / "stage2.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    if change == "semantics":
        state["training_semantics"] = TRAINING_SEMANTICS
    elif change == "architecture":
        state["image_d_provenance"]["num_feat"] += 1
    elif change == "sn_policy":
        state["image_d_spectral_norm_policy"] = "mutates-on-replay"
    elif change == "weight":
        state["image_d"]["conv0.weight"] += 1
    elif change == "source":
        state["image_d_provenance"]["source"] = ""
    elif change == "unknown_key":
        state["image_d_provenance"]["unknown_key"] = 1
    else:
        state["image_d_provenance"]["initial_state_sha256"] = "bad-hash"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_stage2_checkpoint(path)


def test_legacy_patch_checkpoint_without_new_metadata_still_loads(tmp_path, tiny_config, batch):
    _, parent_path = make_parent(tmp_path, tiny_config, batch)
    trainer = Stage2Trainer(tiny_config, test_fixture=True)
    trainer.init_from_stage1(parent_path)
    path = tmp_path / "legacy.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    for key in ("image_d_provenance", "image_d_state_sha256", "image_d_spectral_norm_policy"):
        del state[key]
    for key in (
        "image_discriminator",
        "image_discriminator_pretrained",
        "image_discriminator_chunk_size",
    ):
        del state["config"]["train"][key]
    torch.save(state, path)
    restored = Stage2Trainer(tiny_config, test_fixture=True, restoring=True)
    restored.restore(path)
    assert restored.training_semantics == TRAINING_SEMANTICS
    assert restored.image_d_provenance is None
