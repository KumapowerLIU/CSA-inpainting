import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "remote_training_progress",
    Path(__file__).parents[1] / "scripts/remote/training_progress.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trainer(step=290, epochs=0, batch_size=512, progress=None):
    return SimpleNamespace(
        global_step=step,
        completed_epochs=epochs,
        data_progress=progress,
        config=SimpleNamespace(data=SimpleNamespace(batch_size=batch_size)),
    )


def test_convert_old_batch_position_and_reuse_cursor_after_rebatch():
    state = trainer()
    progress = module.stage1_progress(state, 296145, "digest", saved_batch_size=32)
    assert progress["samples_in_epoch"] == 9280
    assert progress["samples_seen"] == 9280
    state.data_progress = progress
    # The optimizer update count no longer determines the data cursor.
    state.global_step += 10
    state.data_progress["samples_in_epoch"] += 5120
    state.data_progress["samples_seen"] += 5120
    assert module.stage1_progress(state, 296145, "digest")["samples_in_epoch"] == 14400


def test_legacy_tail_and_finished_epoch_positions():
    progress = module.stage1_progress(trainer(step=4), 100, "digest", saved_batch_size=32)
    assert progress["samples_in_epoch"] == progress["samples_seen"] == 100
    progress = module.stage1_progress(trainer(step=4, epochs=1), 100, "digest", 32)
    assert progress["samples_in_epoch"] == 0 and progress["samples_seen"] == 100


def test_reject_changed_dataset_or_inconsistent_cursor():
    state = trainer()
    state.data_progress = module.stage1_progress(state, 296145, "digest", 32)
    with pytest.raises(ValueError, match="cursor"):
        module.stage1_progress(state, 296145, "changed")
    state.data_progress["samples_in_epoch"] += 1
    with pytest.raises(ValueError, match="cursor"):
        module.stage1_progress(state, 296145, "digest")


def test_dataset_switch_starts_new_shuffle_and_preserves_training_state():
    state = trainer(step=6, epochs=1, batch_size=32)
    state.data_progress = module.stage1_progress(state, 100, "old")
    before = deepcopy(state.__dict__)
    # Model/optimizer/RNG etc. are deliberately opaque to the data transition.
    state.model = object()
    state.optimizer = object()
    model, optimizer = state.model, state.optimizer
    progress = module.switch_stage1_dataset(state, 301, "mixed")
    assert state.data_progress == before["data_progress"]
    assert state.global_step == before["global_step"]
    assert state.completed_epochs == before["completed_epochs"]
    assert state.config == before["config"]
    assert state.model is model and state.optimizer is optimizer
    assert progress["samples_in_epoch"] == 0
    assert progress["samples_seen"] == 164
    assert progress["epoch"] == progress["dataset_start_epoch"] == 1
    assert progress["parent_data_progress"] == before["data_progress"]
    state.data_progress = progress
    assert module.stage1_progress(state, 301, "mixed") == progress
    with pytest.raises(ValueError, match="cursor"):
        module.stage1_progress(state, 100, "old")


def test_switched_progress_survives_partial_batch_epoch_boundary_and_another_switch():
    state = trainer(step=6, epochs=1, batch_size=32)
    state.data_progress = module.stage1_progress(state, 100, "old")
    state.data_progress = module.switch_stage1_dataset(state, 301, "mixed")
    state.data_progress["samples_in_epoch"] = 301
    state.data_progress["samples_seen"] += 301
    assert module.stage1_progress(state, 301, "mixed")["samples_seen"] == 465
    state.completed_epochs += 1
    state.data_progress.update(epoch=state.completed_epochs, samples_in_epoch=0)
    module.stage1_progress(state, 301, "mixed")
    state.data_progress["samples_in_epoch"] += 64
    state.data_progress["samples_seen"] += 64
    assert module.stage1_progress(state, 301, "mixed")["samples_seen"] == 529
    state.data_progress = module.switch_stage1_dataset(state, 400, "third")
    assert state.data_progress["samples_in_epoch"] == 0
    assert state.data_progress["samples_seen"] == 529
    assert state.data_progress["parent_data_progress"]["index_sha256"] == "mixed"
    assert module.stage1_progress(state, 400, "third") == state.data_progress


def test_dataset_switch_requires_old_identity_for_legacy_cursor():
    state = trainer(step=6, epochs=1)
    with pytest.raises(ValueError, match="previous dataset"):
        module.switch_stage1_dataset(state, 301, "mixed")
    progress = module.switch_stage1_dataset(
        state,
        301,
        "mixed",
        previous_dataset_size=100,
        previous_index_sha256="old",
        saved_batch_size=32,
    )
    assert progress["samples_seen"] == 164
    assert progress["parent_data_progress"]["samples_in_epoch"] == 64
    assert state.data_progress is None


def test_dataset_switch_does_not_bypass_invalid_old_cursor():
    state = trainer(step=6, epochs=1, batch_size=32)
    state.data_progress = module.stage1_progress(state, 100, "old")
    with pytest.raises(ValueError, match="cursor"):
        module.switch_stage1_dataset(state, 301, "mixed", previous_index_sha256="wrong")
    with pytest.raises(ValueError, match="different dataset"):
        module.switch_stage1_dataset(state, 100, "old")
    state.data_progress["samples_in_epoch"] += 1
    with pytest.raises(ValueError, match="cursor"):
        module.switch_stage1_dataset(state, 301, "mixed")


@pytest.mark.parametrize(
    "key,value",
    [
        ("samples_seen", 165),
        ("dataset_start_epoch", 2),
        ("dataset_start_samples_seen", 165),
        ("parent_data_progress", None),
        ("samples_in_epoch", -1),
        ("epoch", 0),
    ],
)
def test_switched_cursor_rejects_inconsistent_accounting(key, value):
    state = trainer(step=6, epochs=1, batch_size=32)
    state.data_progress = module.stage1_progress(state, 100, "old")
    state.data_progress = module.switch_stage1_dataset(state, 301, "mixed")
    state.data_progress[key] = value
    with pytest.raises(ValueError, match="cursor"):
        module.stage1_progress(state, 301, "mixed")


def test_switched_cursor_provenance_is_independent_of_returned_copies():
    state = trainer(step=6, epochs=1, batch_size=32)
    state.data_progress = module.stage1_progress(state, 100, "old")
    previous = state.data_progress
    state.data_progress = module.switch_stage1_dataset(state, 301, "mixed")
    previous["samples_seen"] = 0
    restored = module.stage1_progress(state, 301, "mixed")
    restored["parent_data_progress"]["samples_seen"] = 0
    assert state.data_progress["parent_data_progress"]["samples_seen"] == 164


def test_stage2_dataset_switch_preserves_full_training_state_and_requires_exact_cursor():
    state = trainer(
        step=33559,
        epochs=5,
        batch_size=192,
        progress={
            "format_version": 1,
            "dataset_size": 1197875,
            "index_sha256": "old",
            "epoch": 5,
            "samples_in_epoch": 453888,
            "samples_seen": 6443263,
        },
    )
    for name in (
        "generator",
        "image_d",
        "feature_d",
        "g_optimizer",
        "d_optimizer",
        "g_scheduler",
        "d_scheduler",
        "scaler",
        "rng_cpu",
        "rng_cuda",
        "rng_python",
        "parent_provenance",
    ):
        setattr(state, name, object())
    before = state.__dict__.copy()
    old_cursor = deepcopy(state.data_progress)
    progress = module.switch_stage2_dataset(state, 1297875, "faces-mixed")
    assert state.__dict__ == before
    assert state.data_progress == old_cursor
    assert progress["epoch"] == progress["dataset_start_epoch"] == 5
    assert progress["samples_in_epoch"] == 0
    assert progress["samples_seen"] == progress["dataset_start_samples_seen"] == 6443263
    assert progress["parent_data_progress"] == old_cursor
    assert progress["parent_data_progress"] is not state.data_progress
    state.data_progress = progress
    assert module.stage2_progress(state, 1297875, "faces-mixed") == progress
    with pytest.raises(ValueError, match="cursor"):
        module.stage2_progress(state, 1197875, "old")


def test_stage2_switched_cursor_survives_epoch_resume_and_further_transition():
    state = trainer(
        step=37,
        epochs=2,
        batch_size=2,
        progress={
            "format_version": 1,
            "dataset_size": 5,
            "index_sha256": "old",
            "epoch": 2,
            "samples_in_epoch": 2,
            "samples_seen": 12,
        },
    )
    state.data_progress = module.switch_stage2_dataset(state, 7, "mixed")
    # Consume 2+2+2+1 records, preserving the final partial batch.
    for count in (2, 2, 2, 1):
        state.global_step += 1
        state.data_progress["samples_in_epoch"] += count
        state.data_progress["samples_seen"] += count
        assert module.stage2_progress(state, 7, "mixed") == state.data_progress
    state.completed_epochs += 1
    state.data_progress.update(epoch=state.completed_epochs, samples_in_epoch=0)
    resumed = deepcopy(state)
    resumed.data_progress = module.stage2_progress(resumed, 7, "mixed")
    assert resumed.data_progress["samples_seen"] == 19
    resumed.data_progress["samples_in_epoch"] += 2
    resumed.data_progress["samples_seen"] += 2
    resumed.global_step += 1
    resumed.data_progress = module.switch_stage2_dataset(resumed, 9, "more-faces")
    assert resumed.global_step == 42 and resumed.completed_epochs == 3
    assert resumed.data_progress["samples_in_epoch"] == 0
    assert resumed.data_progress["samples_seen"] == 21
    assert resumed.data_progress["parent_data_progress"]["index_sha256"] == "mixed"
    assert module.stage2_progress(resumed, 9, "more-faces") == resumed.data_progress


@pytest.mark.parametrize("step,epochs", [(0, 0), (37, 2)])
def test_stage2_dataset_switch_never_infers_a_missing_cursor(step, epochs):
    state = trainer(step=step, epochs=epochs)
    with pytest.raises(ValueError, match="saved data cursor"):
        module.switch_stage2_dataset(state, 7, "mixed")


@pytest.mark.parametrize(
    "change",
    [
        {"samples_seen": 13},
        {"epoch": 1},
        {"index_sha256": ""},
        {"dataset_size": 0},
        {"samples_in_epoch": 6},
    ],
)
def test_stage2_dataset_switch_cannot_bypass_corrupt_cursor(change):
    state = trainer(
        step=37,
        epochs=2,
        progress={
            "format_version": 1,
            "dataset_size": 5,
            "index_sha256": "old",
            "epoch": 2,
            "samples_in_epoch": 2,
            "samples_seen": 12,
        },
    )
    with pytest.raises(ValueError, match="different dataset"):
        module.switch_stage2_dataset(state, 5, "old")
    state.data_progress.update(change)
    with pytest.raises(ValueError):
        module.switch_stage2_dataset(state, 7, "mixed")
