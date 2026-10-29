import builtins
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from csa_inpainting.config import Config
from csa_inpainting.perceptual import LPIPSLoss
from csa_inpainting.stage1 import (
    LPIPS_TRAINING_SEMANTICS,
    TRAINING_SEMANTICS,
    Stage1Trainer,
    load_stage1_checkpoint,
    load_stage1_generator,
)


class FakeLPIPS(nn.Module):
    """Small deterministic perceptual network; no package or weights download."""

    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["net"] in {"vgg", "alex", "squeeze"}
        assert kwargs == {
            "net": kwargs["net"],
            "version": "0.1",
            "pretrained": True,
            "pnet_rand": False,
            "pnet_tune": False,
            "eval_mode": True,
            "verbose": False,
        }
        self.net = nn.Conv2d(3, 4, 3, padding=1)
        self.dropout = nn.Dropout2d(0.5)
        self.calibration = nn.Parameter(torch.tensor([0.2, 0.3, 0.4, 0.5]).view(1, 4, 1, 1))
        with torch.no_grad():
            self.net.weight.copy_(torch.arange(108).reshape_as(self.net.weight) / 1080)
            self.net.bias.fill_(0.1)
        self.calls = []

    def forward(self, prediction, target, normalize=False):
        assert prediction.dtype == target.dtype == torch.float32
        assert normalize is False
        self.calls.append((len(prediction), self.training, torch.is_grad_enabled()))
        difference = self.dropout(self.net(prediction)).square() - self.net(target).square()
        return (difference.square() * self.calibration).mean((1, 2, 3), keepdim=True)


@pytest.fixture
def fake_lpips(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=FakeLPIPS))


def test_zero_weight_does_not_import_or_initialize_lpips(monkeypatch, tiny_config, batch):
    original = builtins.__import__

    def import_without_lpips(name, *args, **kwargs):
        if name == "lpips":
            raise AssertionError("LPIPS must remain optional at zero weight")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lpips)
    trainer = Stage1Trainer(tiny_config)
    assert trainer.perceptual is None and trainer.lpips_provenance is None
    assert set(trainer.step(*batch)) == {"generator", "reconstruction"}


def test_frozen_eval_fp32_metric_preserves_prediction_gradient(fake_lpips):
    loss = LPIPSLoss(batch_size=2).train()
    prediction = torch.rand(3, 3, 12, 12, dtype=torch.bfloat16).requires_grad_()
    target = torch.zeros_like(prediction)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = loss(prediction, target)
    assert value.dtype == torch.float32
    value.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert not loss.training and not loss.network.training
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in loss.parameters()
    )
    assert all(not training for _, training, _ in loss.network.calls)
    assert any(enabled for _, _, enabled in loss.network.calls)


def test_uneven_chunks_match_full_value_and_gradient(fake_lpips):
    chunked = LPIPSLoss(batch_size=4)
    direct = LPIPSLoss(batch_size=10)
    prediction = (torch.rand(10, 3, 12, 12) * 2 - 1).requires_grad_()
    reference = prediction.detach().clone().requires_grad_()
    target = torch.rand_like(prediction) * 2 - 1
    chunked_value = chunked(prediction, target)
    direct_value = direct.network(reference, target, normalize=False).mean()
    chunked_value.backward()
    direct_value.backward()
    torch.testing.assert_close(chunked_value, direct_value)
    torch.testing.assert_close(prediction.grad, reference.grad)
    assert [count for count, _, _ in chunked.network.calls[:3]] == [4, 4, 2]
    assert len(chunked.network.calls) > 3  # Backward recomputes feature activations.


def test_completed_image_perceptual_gradient_is_limited_to_hole(fake_lpips, batch):
    image, mask = batch
    rough = torch.zeros_like(image, requires_grad=True)
    completed = torch.where(mask, rough, image)
    LPIPSLoss(batch_size=1)(completed, image).backward()
    known_gradient = rough.grad[~mask.expand_as(image)]
    assert torch.equal(known_gradient, torch.zeros_like(known_gradient))
    assert rough.grad[mask.expand_as(image)].abs().sum() > 0


def test_stage1_combines_full_image_l1_and_completed_lpips(fake_lpips, tiny_config, batch):
    tiny_config.train.lpips_weight = 0.1
    tiny_config.train.lpips_batch_size = 1
    trainer = Stage1Trainer(tiny_config)
    with torch.no_grad():
        output = trainer.generator(*batch)
        reconstruction = F.l1_loss(output.rough, batch[0]).item()
        perceptual = trainer.perceptual(output.completed, batch[0]).item()
    before = next(trainer.generator.parameters()).detach().clone()
    result = trainer.step(*batch)
    assert result["reconstruction"] == pytest.approx(reconstruction)
    assert result["reconstruction_weighted"] == pytest.approx(reconstruction)
    assert result["lpips"] == pytest.approx(perceptual)
    assert result["lpips_weighted"] == pytest.approx(0.1 * perceptual)
    assert result["generator"] == pytest.approx(reconstruction + 0.1 * perceptual)
    assert not torch.equal(before, next(trainer.generator.parameters()))
    assert all(parameter.grad is None for parameter in trainer.perceptual.parameters())
    optimizer_ids = {id(p) for group in trainer.p_optimizer.param_groups for p in group["params"]}
    assert not optimizer_ids.intersection(id(p) for p in trainer.perceptual.parameters())


def test_explicit_loss_transition_preserves_state_and_can_resume_same_loss(
    fake_lpips,
    tmp_path,
    tiny_config,
    batch,
):
    old = Stage1Trainer(tiny_config)
    old.step(*batch)
    old.finish_epoch()
    old.data_progress = {"epoch": 1, "samples_in_epoch": 17, "samples_seen": 217}
    path = tmp_path / "l1.pt"
    old.save(path)
    state = load_stage1_checkpoint(path)
    config = Config.from_dict(tiny_config.as_dict())
    config.train.lpips_weight = 0.1
    restored = Stage1Trainer(config, restoring=True)
    with pytest.raises(ValueError, match="Resume config differs"):
        restored.restore(path)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        restored.step(*batch)
    restored.restore(path, allow_loss_change=True)
    torch.testing.assert_close(restored.generator.state_dict(), state["generator"], atol=0, rtol=0)
    torch.testing.assert_close(
        restored.p_optimizer.state_dict(), state["p_optimizer"], atol=0, rtol=0
    )
    assert restored.p_scheduler.state_dict() == state["p_scheduler"]
    assert restored.scaler.state_dict() == state["scaler"]
    assert restored.global_step == state["global_step"]
    assert restored.completed_epochs == state["completed_epochs"]
    assert restored.data_progress == state["data_progress"]
    assert torch.equal(torch.get_rng_state(), state["rng_cpu"])
    assert random.getstate() == state["rng_python"]
    assert len(restored.loss_transitions) == 1
    assert restored.loss_transitions[0]["parent_global_step"] == state["global_step"]
    assert restored.loss_transitions[0]["parent_training_semantics"] == TRAINING_SEMANTICS
    assert restored.loss_transitions[0]["loss_config"]["lpips_weight"] == 0.1
    perceptual_path = tmp_path / "lpips.pt"
    restored.save(perceptual_path)
    perceptual_state = load_stage1_checkpoint(perceptual_path)
    assert perceptual_state["training_semantics"] == LPIPS_TRAINING_SEMANTICS
    assert perceptual_state["feature_mode"] == "lpips_vgg_v0.1_stage1"
    assert perceptual_state["lpips_provenance"] == restored.lpips_provenance
    again = Stage1Trainer(config, restoring=True)
    again.restore(perceptual_path)  # Opt-in is only needed for the initial change.
    assert again.loss_transitions == restored.loss_transitions
    assert restored.step(*batch) == again.step(*batch)
    torch.testing.assert_close(
        restored.generator.state_dict(), again.generator.state_dict(), atol=0, rtol=0
    )
    model, loaded_config = load_stage1_generator(perceptual_path, torch.device("cpu"))
    assert not model.training and loaded_config.train.lpips_weight == 0.1


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("train", "lr", 0.001),
        ("train", "reconstruction_weight", 0.5),
        ("data", "mask_mode", "center"),
        ("train", "seed", 5),
    ],
)
def test_loss_opt_in_does_not_allow_unrelated_changes(
    fake_lpips,
    tmp_path,
    tiny_config,
    section,
    field,
    value,
):
    path = tmp_path / "l1.pt"
    Stage1Trainer(tiny_config).save(path)
    config = Config.from_dict(tiny_config.as_dict())
    config.train.lpips_weight = 0.1
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Resume config differs"):
        Stage1Trainer(config, restoring=True).restore(path, allow_loss_change=True)


def test_existing_lpips_objective_cannot_silently_change(fake_lpips, tmp_path, tiny_config):
    tiny_config.train.lpips_weight = 0.1
    path = tmp_path / "lpips.pt"
    Stage1Trainer(tiny_config).save(path)
    for weight in (0.0, 0.2):
        config = Config.from_dict(tiny_config.as_dict())
        config.train.lpips_weight = weight
        with pytest.raises(ValueError, match="Resume config differs"):
            Stage1Trainer(config, restoring=True).restore(path)
        if weight == 0.0:
            with pytest.raises(ValueError, match="Resume config differs"):
                Stage1Trainer(config, restoring=True).restore(path, allow_loss_change=True)
    state = load_stage1_checkpoint(path)
    state["lpips_provenance"]["metric_weights_sha256"] = "changed weights"
    torch.save(state, path)
    with pytest.raises(ValueError, match="Resume config differs"):
        Stage1Trainer(tiny_config, restoring=True).restore(path)
    changed_config = Config.from_dict(tiny_config.as_dict())
    changed_config.train.reconstruction_weight = 0.1
    changed_config.train.lpips_weight = 1.0
    with pytest.raises(ValueError, match="Resume config differs"):
        Stage1Trainer(changed_config, restoring=True).restore(path, allow_loss_change=True)


def test_requested_coefficients_preserve_state_history_and_resume(
    fake_lpips,
    tmp_path,
    tiny_config,
    batch,
):
    # Include the earlier L1 -> LPIPS transition in the checkpoint history.
    l1_path = tmp_path / "l1.pt"
    source = Stage1Trainer(tiny_config)
    source.step(*batch)
    source.save(l1_path)
    old_config = Config.from_dict(tiny_config.as_dict())
    old_config.train.lpips_weight = 0.1
    old = Stage1Trainer(old_config, restoring=True)
    old.restore(l1_path, allow_loss_change=True)
    old.step(*batch)
    old.finish_epoch()
    old.data_progress = {"epoch": 1, "samples_in_epoch": 512, "samples_seen": 4096}
    path = tmp_path / "old-ratio.pt"
    old.save(path)
    state = load_stage1_checkpoint(path)

    config = Config.from_dict(old_config.as_dict())
    config.train.reconstruction_weight = 0.1
    config.train.lpips_weight = 1.0
    restored = Stage1Trainer(config, restoring=True)
    with pytest.raises(ValueError, match="Resume config differs"):
        restored.restore(path)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        restored.step(*batch)
    restored.restore(path, allow_loss_change=True)
    torch.testing.assert_close(restored.generator.state_dict(), state["generator"], atol=0, rtol=0)
    torch.testing.assert_close(
        restored.p_optimizer.state_dict(), state["p_optimizer"], atol=0, rtol=0
    )
    assert restored.p_scheduler.state_dict() == state["p_scheduler"]
    assert restored.scaler.state_dict() == state["scaler"]
    assert restored.global_step == state["global_step"]
    assert restored.completed_epochs == state["completed_epochs"]
    assert restored.data_progress == state["data_progress"]
    assert torch.equal(torch.get_rng_state(), state["rng_cpu"])
    assert random.getstate() == state["rng_python"]
    assert len(restored.loss_transitions) == 2
    assert restored.loss_transitions[:-1] == state["loss_transitions"]
    transition = restored.loss_transitions[-1]
    assert transition["change_kind"] == "loss_coefficients"
    assert transition["parent_global_step"] == state["global_step"]
    assert transition["parent_loss_config"]["reconstruction_weight"] == 1.0
    assert transition["parent_loss_config"]["lpips_weight"] == 0.1
    assert transition["loss_config"]["reconstruction_weight"] == 0.1
    assert transition["loss_config"]["lpips_weight"] == 1.0
    assert transition["parent_training_semantics"] == transition["training_semantics"]
    assert restored.lpips_provenance == state["lpips_provenance"]

    new_path = tmp_path / "new-ratio.pt"
    restored.save(new_path)
    again = Stage1Trainer(config, restoring=True)
    again.restore(new_path)
    again.restore(new_path, allow_loss_change=True)
    assert again.loss_transitions == restored.loss_transitions  # No duplicate records.
    with torch.no_grad():
        output = restored.generator(*batch)
        reconstruction = F.l1_loss(output.rough, batch[0]).item()
        perceptual = restored.perceptual(output.completed, batch[0]).item()
    result = restored.step(*batch)
    assert result["reconstruction_weighted"] == pytest.approx(0.1 * reconstruction)
    assert result["lpips_weighted"] == pytest.approx(perceptual)
    assert result["generator"] == pytest.approx(0.1 * reconstruction + perceptual)
    assert result == again.step(*batch)
    torch.testing.assert_close(
        restored.generator.state_dict(), again.generator.state_dict(), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("train", "lr", 0.001),
        ("data", "mask_mode", "center"),
        ("train", "seed", 5),
        ("train", "lpips_batch_size", 1),
        ("train", "lpips_net", "alex"),
    ],
)
def test_coefficient_opt_in_keeps_other_settings_strict(
    fake_lpips,
    tmp_path,
    tiny_config,
    section,
    field,
    value,
):
    tiny_config.train.lpips_weight = 0.1
    path = tmp_path / "lpips.pt"
    Stage1Trainer(tiny_config).save(path)
    config = Config.from_dict(tiny_config.as_dict())
    config.train.reconstruction_weight = 0.1
    config.train.lpips_weight = 1.0
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Resume config differs"):
        Stage1Trainer(config, restoring=True).restore(path, allow_loss_change=True)


def test_older_l1_checkpoint_without_new_fields_loads(tmp_path, tiny_config):
    path = tmp_path / "old.pt"
    Stage1Trainer(tiny_config).save(path)
    state = load_stage1_checkpoint(path)
    for key in ("lpips_weight", "lpips_net", "lpips_batch_size"):
        state["config"]["train"].pop(key)
    state.pop("loss_transitions")
    state.pop("lpips_provenance")
    torch.save(state, path)
    restored = Stage1Trainer(tiny_config, restoring=True)
    restored.restore(path)
    assert restored.loss_transitions == [] and restored.lpips_provenance is None
    _, config = load_stage1_generator(path, torch.device("cpu"))
    assert config.train.lpips_weight == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("lpips_weight", -0.1),
        ("lpips_weight", float("nan")),
        ("lpips_weight", float("inf")),
        ("lpips_weight", True),
        ("lpips_batch_size", 0),
        ("lpips_batch_size", -1),
        ("lpips_batch_size", 1.5),
        ("lpips_net", "untrained"),
    ],
)
def test_invalid_lpips_settings_rejected(tiny_config, field, value):
    setattr(tiny_config.train, field, value)
    with pytest.raises(ValueError):
        tiny_config.validate()
