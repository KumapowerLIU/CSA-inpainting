import math
import random

import pytest
import torch
from torch.nn import functional as F

from csa_inpainting.config import Config
from csa_inpainting.engine import load_checkpoint
from csa_inpainting.networks import RoughNetwork
from csa_inpainting.stage1 import (
    TRAINING_SEMANTICS,
    RoughInpaintingModel,
    Stage1Trainer,
    load_stage1_checkpoint,
    load_stage1_generator,
)


def test_only_rough_network_and_full_image_l1(tiny_config, batch):
    trainer = Stage1Trainer(tiny_config)
    assert list(dict(trainer.generator.named_children())) == ["rough"]
    assert isinstance(trainer.generator.rough, RoughNetwork)
    assert not any(hasattr(trainer, name) for name in ("image_d", "feature_d", "features"))
    parameter_ids = {id(p) for p in trainer.generator.parameters()}
    assert parameter_ids == {
        id(p) for group in trainer.p_optimizer.param_groups for p in group["params"]
    }
    with torch.no_grad():
        expected = F.l1_loss(trainer.generator(*batch).rough, batch[0]).item()
    before = next(trainer.generator.parameters()).detach().clone()
    result = trainer.step(*batch)
    assert result["reconstruction"] == pytest.approx(expected)
    assert result["generator"] == pytest.approx(expected * tiny_config.train.reconstruction_weight)
    assert not torch.equal(before, next(trainer.generator.parameters()))
    assert trainer.global_step == 1


def test_rough_output_preserves_known_region_and_input(batch):
    image, mask = batch
    before = image.clone()
    model = RoughInpaintingModel(width=4, depth=6)
    output = model(image, mask)
    assert output.rough.shape == image.shape
    assert torch.equal(output.completed[~mask.expand_as(image)], image[~mask.expand_as(image)])
    assert torch.equal(image, before)
    assert all(key == "fill" or key.startswith("rough.") for key in model.state_dict())


def test_stage1_checkpoint_exact_next_update(tmp_path, tiny_config, batch):
    trainer = Stage1Trainer(tiny_config)
    trainer.step(*batch)
    trainer.finish_epoch()
    checkpoint = tmp_path / "stage1.pt"
    trainer.save(checkpoint)
    restored = Stage1Trainer(tiny_config, restoring=True)
    restored.restore(checkpoint)
    expected, actual = trainer.step(*batch), restored.step(*batch)
    assert actual == expected
    torch.testing.assert_close(
        trainer.generator.state_dict(), restored.generator.state_dict(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        trainer.p_optimizer.state_dict(), restored.p_optimizer.state_dict(), atol=0, rtol=0
    )
    assert trainer.p_scheduler.state_dict() == restored.p_scheduler.state_dict()
    assert restored.global_step == 2 and restored.completed_epochs == 1
    state = load_stage1_checkpoint(checkpoint)
    assert state["training_semantics"] == TRAINING_SEMANTICS and state["stage"] == 1
    loaded, config = load_stage1_generator(checkpoint, torch.device("cpu"))
    assert not loaded.training and config == tiny_config
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(checkpoint)
    tiny_config.data.batch_size = 3
    with pytest.raises(ValueError, match="Resume config differs"):
        Stage1Trainer(tiny_config, restoring=True).restore(checkpoint)


@pytest.mark.parametrize("field,value", [("kind", "joint"), ("format_version", 2), ("stage", 2)])
def test_reject_other_stage_and_pending_restore(tmp_path, tiny_config, batch, field, value):
    trainer = Stage1Trainer(tiny_config)
    checkpoint = tmp_path / "bad.pt"
    trainer.save(checkpoint)
    state = load_stage1_checkpoint(checkpoint)
    state[field] = value
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="stage-1 rough-only"):
        trainer.restore(checkpoint)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        trainer.step(*batch)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        trainer.save(tmp_path / "partial.pt")
    assert not (tmp_path / "partial.pt").exists()


def test_stage1_cpu_amp_rejected(tiny_config):
    tiny_config.train.amp = "bf16"
    with pytest.raises(ValueError, match="CUDA only"):
        Stage1Trainer(tiny_config)


def test_explicit_rebatch_preserves_training_state_and_data_progress(tmp_path, tiny_config, batch):
    trainer = Stage1Trainer(tiny_config)
    trainer.step(*batch)
    trainer.finish_epoch()
    trainer.data_progress = {"epoch": 1, "samples_in_epoch": 17, "total_samples": 217}
    checkpoint = tmp_path / "before-rebatch.pt"
    trainer.save(checkpoint)
    saved = load_stage1_checkpoint(checkpoint)
    config = Config.from_dict(tiny_config.as_dict())
    config.data.batch_size = 8
    restored = Stage1Trainer(config, restoring=True)
    with pytest.raises(ValueError, match="Resume config differs"):
        restored.restore(checkpoint)
    restored.restore(checkpoint, allow_batch_size_change=True)
    assert restored.config.data.batch_size == 8
    assert restored.global_step == trainer.global_step
    assert restored.completed_epochs == trainer.completed_epochs
    assert restored.data_progress == trainer.data_progress
    torch.testing.assert_close(restored.generator.state_dict(), saved["generator"], atol=0, rtol=0)
    torch.testing.assert_close(
        restored.p_optimizer.state_dict(), saved["p_optimizer"], atol=0, rtol=0
    )
    assert restored.p_scheduler.state_dict() == saved["p_scheduler"]
    assert restored.scaler.state_dict() == saved["scaler"]
    assert torch.equal(torch.get_rng_state(), saved["rng_cpu"])
    assert random.getstate() == saved["rng_python"]
    resumed_checkpoint = tmp_path / "after-rebatch.pt"
    restored.save(resumed_checkpoint)
    assert load_stage1_checkpoint(resumed_checkpoint)["data_progress"] == trainer.data_progress


@pytest.mark.parametrize(
    "section,field,value", [("data", "mask_mode", "center"), ("train", "lr", 0.001)]
)
def test_rebatch_does_not_allow_other_training_changes(
    tmp_path, tiny_config, section, field, value
):
    trainer = Stage1Trainer(tiny_config)
    checkpoint = tmp_path / "source.pt"
    trainer.save(checkpoint)
    config = Config.from_dict(tiny_config.as_dict())
    config.data.batch_size = 8
    setattr(getattr(config, section), field, value)
    restored = Stage1Trainer(config, restoring=True)
    with pytest.raises(ValueError, match="Resume config differs"):
        restored.restore(checkpoint, allow_batch_size_change=True)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        restored.save(tmp_path / "invalid.pt")


def test_older_stage1_checkpoint_has_unknown_data_progress(tmp_path, tiny_config):
    trainer = Stage1Trainer(tiny_config)
    checkpoint = tmp_path / "older.pt"
    trainer.save(checkpoint)
    saved = load_stage1_checkpoint(checkpoint)
    saved.pop("data_progress")
    torch.save(saved, checkpoint)
    restored = Stage1Trainer(tiny_config, restoring=True)
    restored.restore(checkpoint)
    assert restored.data_progress is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_stage1_cuda_amp(tiny_config, batch, amp):
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    tiny_config.train.device, tiny_config.train.amp = "cuda", amp
    trainer = Stage1Trainer(tiny_config)
    assert all(math.isfinite(v) for v in trainer.step(*batch).values())
