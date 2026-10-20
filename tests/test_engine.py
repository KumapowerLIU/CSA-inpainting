import math
from contextlib import contextmanager

import pytest
import torch

from csa_inpainting.engine import Trainer, load_checkpoint, load_generator


def test_training_updates_both_stages_and_discriminators(tiny_config, batch):
    trainer = Trainer(tiny_config, test_fixture=True)
    modules = [
        trainer.generator.rough,
        trainer.generator.refine,
        trainer.image_d,
        trainer.feature_d,
    ]
    before = [next(module.parameters()).detach().clone() for module in modules]
    result = trainer.step(*batch)
    assert all(math.isfinite(value) for value in result.values())
    for old, module in zip(before, modules, strict=True):
        assert not torch.equal(old, next(module.parameters()))


def test_checkpoint_exact_next_step(tmp_path, tiny_config, batch):
    trainer = Trainer(tiny_config, test_fixture=True)
    trainer.step(*batch)
    trainer.finish_epoch()
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    restored = Trainer(tiny_config, test_fixture=True, restoring=True)
    restored.restore(path)
    expected = trainer.step(*batch)
    actual = restored.step(*batch)
    assert actual == expected
    for key, value in trainer.generator.state_dict().items():
        torch.testing.assert_close(value, restored.generator.state_dict()[key], atol=0, rtol=0)
    assert restored.global_step == 2 and restored.completed_epochs == 1
    assert restored.g_scheduler.state_dict() == trainer.g_scheduler.state_dict()
    model, config = load_generator(path, torch.device("cpu"))
    assert config.model == tiny_config.model
    assert not model.training
    assert load_checkpoint(path)["test_fixture"] is True
    tiny_config.train.lr *= 2
    incompatible = Trainer(tiny_config, test_fixture=True, restoring=True)
    with pytest.raises(ValueError, match="Resume config differs"):
        incompatible.restore(path)


def test_real_training_rejects_tiny_features(tiny_config):
    with pytest.raises(ValueError, match="requires model.width=64"):
        Trainer(tiny_config)


def test_cpu_amp_rejected(tiny_config):
    tiny_config.train.amp = "fp16"
    with pytest.raises(ValueError, match="CUDA only"):
        Trainer(tiny_config, test_fixture=True)


def test_pending_restore_cannot_train_or_save(tmp_path, tiny_config, batch):
    pending = Trainer(tiny_config, test_fixture=True, restoring=True)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        pending.step(*batch)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        pending.finish_epoch()
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        pending.save(tmp_path / "invalid.pt")
    assert not (tmp_path / "invalid.pt").exists()

    checkpoint = tmp_path / "valid.pt"
    Trainer(tiny_config, test_fixture=True).save(checkpoint)
    pending.restore(checkpoint)
    assert math.isfinite(pending.step(*batch)["generator"])

    # A failure after model loading must not permit a partly restored run.
    state = load_checkpoint(checkpoint)
    state["g_optimizer"]["param_groups"] = []
    torch.save(state, checkpoint)
    with pytest.raises(ValueError):
        pending.restore(checkpoint)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        pending.step(*batch)


def test_random_vgg_diagnostic_is_explicit_in_checkpoint(tmp_path, tiny_config, monkeypatch):
    from csa_inpainting import engine
    from csa_inpainting.diagnostics import TinyFeatures

    # Test mode selection and checkpoint provenance without allocating full VGG.
    tiny_config.model.width = 64
    observed = []

    def vgg(pretrained):
        observed.append(pretrained)
        return TinyFeatures(512)

    monkeypatch.setattr(engine, "VGGFeatures", vgg)
    monkeypatch.setattr(engine, "InpaintingModel", lambda **kwargs: torch.nn.Conv2d(3, 3, 1))
    Trainer(tiny_config)
    assert observed == [True]
    diagnostic = Trainer(tiny_config, diagnostic="random_vgg")
    assert observed == [True, False]
    checkpoint = tmp_path / "random-vgg.pt"
    diagnostic.save(checkpoint)
    state = load_checkpoint(checkpoint)
    assert state["feature_mode"] == "random_vgg_diagnostic"
    assert state["test_fixture"] is True
    normal = Trainer(tiny_config, restoring=True)
    with pytest.raises(ValueError, match="Resume config differs"):
        normal.restore(checkpoint)
    restored = Trainer(tiny_config, diagnostic="random_vgg", restoring=True)
    restored.restore(checkpoint)
    restored.save(tmp_path / "resumed-diagnostic.pt")
    assert (
        load_checkpoint(tmp_path / "resumed-diagnostic.pt")["feature_mode"] == state["feature_mode"]
    )


def test_bf16_checks_selected_cuda_device(tiny_config, monkeypatch):
    from csa_inpainting import engine

    tiny_config.train.amp = "bf16"
    selected = []

    @contextmanager
    def device_context(device):
        selected.append(str(device))
        yield

    monkeypatch.setattr(engine, "choose_device", lambda _: torch.device("cuda:1"))
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="cuda:1 does not support bf16"):
        Trainer(tiny_config, test_fixture=True)
    assert selected == ["cuda:1"]


def test_cuda_rng_topology_mismatch_rejected_before_loading(
    tmp_path, tiny_config, batch, monkeypatch
):
    trainer = Trainer(tiny_config, test_fixture=True)
    checkpoint = tmp_path / "one-gpu.pt"
    trainer.save(checkpoint)
    state = load_checkpoint(checkpoint)
    state["rng_cuda"] = [torch.get_rng_state()]
    torch.save(state, checkpoint)
    trainer.device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="different GPU counts"):
        trainer.restore(checkpoint)
    with pytest.raises(RuntimeError, match="awaiting a successful restore"):
        trainer.step(*batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_cuda_amp(tiny_config, batch, amp):
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    tiny_config.train.device = "cuda"
    tiny_config.train.amp = amp
    trainer = Trainer(tiny_config, test_fixture=True)
    result = trainer.step(*batch)
    assert all(math.isfinite(v) for v in result.values())
