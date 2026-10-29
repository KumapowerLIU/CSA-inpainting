import copy
import math
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from test_stage1_lpips import FakeLPIPS
from torch.nn import functional as F

from csa_inpainting.config import Config
from csa_inpainting.losses import consistency, relativistic_generator
from csa_inpainting.stage1 import Stage1Trainer, load_stage1_checkpoint
from csa_inpainting.stage2 import Stage2Trainer, load_stage2_generator


class RestorableFakeLPIPS(FakeLPIPS):
    def __init__(self, **kwargs):
        restoring = kwargs["pnet_rand"]
        assert kwargs["pretrained"] is not restoring
        super().__init__(**{**kwargs, "pretrained": True, "pnet_rand": False})
        if restoring:
            # Stand in for a different random skeleton: exact recovery must
            # actually load saved feature/calibration tensors, not use defaults.
            with torch.no_grad():
                self.net.weight.fill_(0.07)
                self.calibration.fill_(0.9)


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=RestorableFakeLPIPS))


@pytest.fixture
def stage2_config(tiny_config):
    config = Config.from_dict(tiny_config.as_dict())
    config.train.reconstruction_weight = 0.1
    config.train.lpips_weight = 1.0
    config.train.lpips_batch_size = 1
    return config


def make_parent(tmp_path, config, batch):
    parent = Stage1Trainer(Config.from_dict(config.as_dict()))
    parent.step(*batch)
    parent.finish_epoch()
    parent.data_progress = {"epoch": 1, "samples_in_epoch": 0, "samples_seen": len(batch[0])}
    path = tmp_path / "stage1.pt"
    parent.save(path)
    return parent, path


def ready_trainer(tmp_path, tiny_config, stage2_config, batch):
    parent, path = make_parent(tmp_path, tiny_config, batch)
    trainer = Stage2Trainer(stage2_config, test_fixture=True)
    trainer.init_from_stage1(path)
    return trainer, parent, path


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_changed(before, module):
    assert any(not torch.equal(value, module.state_dict()[name]) for name, value in before.items())


def assert_pending(trainer, batch, destination):
    with pytest.raises(RuntimeError):
        trainer.step(*batch)
    with pytest.raises(RuntimeError):
        trainer.finish_epoch()
    with pytest.raises(RuntimeError):
        trainer.save(destination)
    assert not destination.exists()


@pytest.mark.parametrize("parent_has_lpips", [False, True])
def test_parent_rough_is_exact_and_stage2_starts_a_new_clock(
    tmp_path, tiny_config, stage2_config, batch, parent_has_lpips
):
    if parent_has_lpips:
        tiny_config.train.lpips_weight = 1.0
        tiny_config.train.reconstruction_weight = 0.1
    parent, path = make_parent(tmp_path, tiny_config, batch)
    trainer = Stage2Trainer(stage2_config, test_fixture=True)
    assert_pending(trainer, batch, tmp_path / "uninitialized.pt")
    trainer.init_from_stage1(path)
    with torch.no_grad():
        expected = parent.generator(*batch)
        actual = trainer.generator(*batch)
    torch.testing.assert_close(actual.rough, expected.rough, atol=0, rtol=0)
    torch.testing.assert_close(actual.masked, expected.masked, atol=0, rtol=0)
    torch.testing.assert_close(trainer.generator.fill, parent.generator.fill, atol=0, rtol=0)
    torch.testing.assert_close(
        trainer.generator.rough.state_dict(), parent.generator.rough.state_dict(), atol=0, rtol=0
    )
    assert trainer.global_step == 0 and trainer.completed_epochs == 0
    assert trainer.data_progress is None
    assert parent.global_step > trainer.global_step and parent.completed_epochs > 0
    assert not trainer.g_optimizer.state and not trainer.d_optimizer.state


@pytest.mark.parametrize("mismatch", ["width", "depth", "fill"])
def test_parent_rough_contract_is_strict(tmp_path, tiny_config, stage2_config, batch, mismatch):
    _, path = make_parent(tmp_path, tiny_config, batch)
    state = load_stage1_checkpoint(path)
    if mismatch == "fill":
        state["generator"]["fill"] = state["generator"]["fill"] + 0.2
    else:
        state["config"]["model"][mismatch] = {"width": 8, "depth": 5}[mismatch]
    torch.save(state, path)
    trainer = Stage2Trainer(stage2_config, test_fixture=True)
    with pytest.raises((ValueError, RuntimeError)):
        trainer.init_from_stage1(path)
    assert_pending(trainer, batch, tmp_path / "bad-parent.pt")


def test_only_refinement_and_discriminators_update(tmp_path, tiny_config, stage2_config, batch):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    rough = snapshot(trainer.generator.rough)
    fill = trainer.generator.fill.clone()
    trainable = [trainer.generator.refine, trainer.image_d, trainer.feature_d]
    before = [snapshot(module) for module in trainable]
    feature_state, perceptual_state = snapshot(trainer.features), snapshot(trainer.perceptual)
    rough_ids = {id(p) for p in trainer.generator.rough.parameters()}
    optimizer_ids = {
        id(p)
        for optimizer in (trainer.g_optimizer, trainer.d_optimizer)
        for group in optimizer.param_groups
        for p in group["params"]
    }
    assert not rough_ids.intersection(optimizer_ids)
    for _ in range(3):
        trainer.generator.train()  # Calling train() must not unfreeze the rough teacher.
        result = trainer.step(*batch)
        assert all(math.isfinite(value) for value in result.values())
        assert not trainer.generator.rough.training
        assert all(
            not p.requires_grad and p.grad is None for p in trainer.generator.rough.parameters()
        )
    torch.testing.assert_close(trainer.generator.rough.state_dict(), rough, atol=0, rtol=0)
    torch.testing.assert_close(trainer.generator.fill, fill, atol=0, rtol=0)
    torch.testing.assert_close(trainer.features.state_dict(), feature_state, atol=0, rtol=0)
    torch.testing.assert_close(trainer.perceptual.state_dict(), perceptual_state, atol=0, rtol=0)
    for old, module in zip(before, trainable, strict=True):
        assert_changed(old, module)
    assert not trainer.features.training and not trainer.perceptual.training
    assert all(p.grad is None for p in trainer.perceptual.parameters())


def test_refinement_is_bounded_and_completed_preserves_known_pixels(
    tmp_path, tiny_config, stage2_config, batch
):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    image, mask = batch
    original = image.clone()
    # Force large raw logits so a merely linear head fails this contract.
    convolutions = [
        m for m in trainer.generator.refine.modules() if isinstance(m, torch.nn.ConvTranspose2d)
    ]
    with torch.no_grad():
        convolutions[-1].weight.zero_()
        convolutions[-1].bias.fill_(50)
        output = trainer.generator(image, mask)
    assert output.refined.min() >= -1 and output.refined.max() <= 1
    assert output.refined.min() > 0.99
    torch.testing.assert_close(
        output.completed[mask.expand_as(image)],
        output.refined[mask.expand_as(image)],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output.completed[~mask.expand_as(image)], image[~mask.expand_as(image)], atol=0, rtol=0
    )
    torch.testing.assert_close(image, original, atol=0, rtol=0)


@pytest.mark.parametrize("route", ["l1", "lpips", "consistency", "image_gan", "feature_gan"])
def test_each_stage2_loss_reaches_refinement_without_rough_gradients(
    tmp_path, tiny_config, stage2_config, batch, route
):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    image, mask = batch
    output = trainer.generator(image, mask)
    trainer.image_d.requires_grad_(False)
    trainer.feature_d.requires_grad_(False)
    with torch.no_grad():
        real_features = trainer.features(image)
    if route == "l1":
        loss = F.l1_loss(output.refined, image)
    elif route == "lpips":
        loss = trainer.perceptual(output.completed, image)
    elif route == "consistency":
        loss = consistency(output, real_features["relu4_3"])
    elif route == "image_gan":
        loss = relativistic_generator(trainer.image_d(image), trainer.image_d(output.refined))
    else:
        fake_features = trainer.features(output.refined)
        loss = relativistic_generator(
            trainer.feature_d(real_features["pool3"]), trainer.feature_d(fake_features["pool3"])
        )
    loss.backward()
    grads = [p.grad for p in trainer.generator.refine.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None for p in trainer.generator.rough.parameters())
    assert all(p.grad is None for p in trainer.features.parameters())
    assert all(p.grad is None for p in trainer.perceptual.parameters())


def test_requested_reconstruction_weights_are_used_in_full_objective(
    tmp_path, tiny_config, stage2_config, batch
):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    with torch.no_grad():
        output = trainer.generator(*batch)
        l1 = F.l1_loss(output.refined, batch[0]).item()
        lpips = trainer.perceptual(output.completed, batch[0]).item()
    result = trainer.step(*batch)
    assert result["reconstruction"] == pytest.approx(l1)
    assert result["lpips"] == pytest.approx(lpips)
    assert result["reconstruction_weighted"] == pytest.approx(0.1 * l1)
    assert result["lpips_weighted"] == pytest.approx(lpips)
    expected = 0.1 * l1 + lpips + stage2_config.train.consistency_weight * result["consistency"]
    expected += stage2_config.train.adversarial_weight * (
        result["adversarial_image"] + result["adversarial_feature"]
    )
    assert result["generator"] == pytest.approx(expected)


def test_resume_restores_exact_state_and_numerically_equivalent_next_update(
    tmp_path, tiny_config, stage2_config, batch
):
    trainer, _, parent_path = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    trainer.step(*batch)
    trainer.finish_epoch()
    trainer.data_progress = {"epoch": 1, "samples_in_epoch": 2, "samples_seen": 4}
    path = tmp_path / "stage2.pt"
    trainer.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    parent_path.unlink()
    restored = Stage2Trainer(stage2_config, test_fixture=True, restoring=True)
    restored.restore(path)
    assert torch.equal(torch.get_rng_state(), state["rng_cpu"])
    assert random.getstate() == state["rng_python"]
    assert restored.global_step == 1 and restored.completed_epochs == 1
    assert restored.data_progress == trainer.data_progress
    states = (
        "generator",
        "image_d",
        "feature_d",
        "features",
        "perceptual",
        "g_optimizer",
        "d_optimizer",
        "g_scheduler",
        "d_scheduler",
        "scaler",
    )
    # Serialization must preserve every network, optimizer and scheduler tensor
    # exactly, independently of the numerical behavior of the next CPU update.
    for name in states:
        torch.testing.assert_close(
            getattr(trainer, name).state_dict(),
            getattr(restored, name).state_dict(),
            atol=0,
            rtol=0,
        )
    model, loaded_config = load_stage2_generator(path, torch.device("cpu"))
    assert not model.training and loaded_config == stage2_config
    with torch.inference_mode():
        torch.testing.assert_close(
            model(*batch).completed, trainer.generator(*batch).completed, atol=0, rtol=0
        )
    expected, actual = trainer.step(*batch), restored.step(*batch)
    # Independent in-memory copies (no checkpoint loading) also show CPU reduction
    # roundoff. Adam can amplify near-zero gradients of biases before normalization,
    # so compare the next update's losses and actual outputs, not raw bias/moment
    # equality. Every restored tensor was already required to match bitwise above.
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
    for name in ("features", "perceptual", "g_scheduler", "d_scheduler", "scaler"):
        torch.testing.assert_close(
            getattr(trainer, name).state_dict(),
            getattr(restored, name).state_dict(),
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(
        trainer.generator.rough.state_dict(),
        restored.generator.rough.state_dict(),
        atol=0,
        rtol=0,
    )
    for name in ("g_optimizer", "d_optimizer"):
        left, right = getattr(trainer, name).state_dict(), getattr(restored, name).state_dict()
        assert left["param_groups"] == right["param_groups"]
        for parameter in left["state"]:
            assert torch.equal(left["state"][parameter]["step"], right["state"][parameter]["step"])
    with torch.inference_mode():
        expected_output = trainer.generator.eval()(*batch)
        actual_output = restored.generator.eval()(*batch)
        for name in ("refined", "completed"):
            torch.testing.assert_close(
                getattr(actual_output, name),
                getattr(expected_output, name),
                atol=1e-6,
                rtol=1e-5,
            )
    assert restored.global_step == 2


def test_stage1_cannot_be_resumed_as_stage2_or_the_reverse(
    tmp_path, tiny_config, stage2_config, batch
):
    trainer, _, parent_path = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    path = tmp_path / "stage2.pt"
    trainer.save(path)
    with pytest.raises(ValueError):
        trainer.restore(parent_path)
    assert_pending(trainer, batch, tmp_path / "wrong-resume.pt")
    with pytest.raises(ValueError):
        Stage1Trainer(tiny_config, restoring=True).restore(path)
    with pytest.raises(ValueError):
        load_stage2_generator(parent_path, torch.device("cpu"))


@pytest.mark.parametrize(
    "field", ["feature_weights_sha256", "metric_weights_sha256", "saved_feature_weights"]
)
def test_lpips_provenance_drift_is_rejected(tmp_path, tiny_config, stage2_config, batch, field):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    path = tmp_path / "stage2.pt"
    trainer.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if field == "saved_feature_weights":
        key = next(key for key in state["perceptual"] if "net.weight" in key)
        state["perceptual"][key] = state["perceptual"][key] + 0.1
    else:
        state["lpips_provenance"][field] = "unexpected replacement"
    torch.save(state, path)
    with pytest.raises(ValueError):
        trainer.restore(path)
    assert_pending(trainer, batch, tmp_path / "provenance-drift.pt")


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 0.001),
        ("lpips_net", "alex"),
        ("reconstruction_weight", 1.0),
        ("lpips_weight", 0.1),
    ],
)
def test_resume_keeps_loss_and_optimizer_configuration_strict(
    tmp_path, tiny_config, stage2_config, batch, field, value
):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    path = tmp_path / "stage2.pt"
    trainer.save(path)
    changed = Config.from_dict(stage2_config.as_dict())
    setattr(changed.train, field, value)
    restored = Stage2Trainer(changed, test_fixture=True, restoring=True)
    with pytest.raises(ValueError):
        restored.restore(path)
    assert_pending(restored, batch, tmp_path / "changed-settings.pt")


def test_partial_restore_failure_is_fail_closed_and_can_be_retried(
    tmp_path, tiny_config, stage2_config, batch
):
    trainer, _, _ = ready_trainer(tmp_path, tiny_config, stage2_config, batch)
    trainer.step(*batch)
    valid_path = tmp_path / "valid.pt"
    trainer.save(valid_path)
    state = torch.load(valid_path, map_location="cpu", weights_only=True)
    bad = copy.deepcopy(state)
    bad["d_optimizer"]["param_groups"] = []
    bad_path = tmp_path / "partly-loadable.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError):
        trainer.restore(bad_path)
    assert_pending(trainer, batch, tmp_path / "partial.pt")
    trainer.restore(valid_path)
    assert trainer.global_step == state["global_step"]
    assert math.isfinite(trainer.step(*batch)["generator"])
