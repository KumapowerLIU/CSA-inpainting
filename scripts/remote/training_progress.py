"""Sample cursors for exact resumption and explicit dataset transitions."""

import math
from copy import deepcopy


def _validate_progress(progress, dataset_size, index_sha256, epoch):
    error = "Saved data cursor does not match this dataset/epoch"
    if not isinstance(progress, dict) or (
        progress.get("dataset_size") != dataset_size
        or progress.get("index_sha256") != index_sha256
        or progress.get("epoch") != epoch
        or any(
            type(progress.get(key)) is not int
            for key in ("epoch", "samples_in_epoch", "samples_seen")
        )
        or epoch < 0
        or not 0 <= progress["samples_in_epoch"] <= dataset_size
    ):
        raise ValueError(error)
    if progress.get("format_version") == 1:
        expected_seen = epoch * dataset_size + progress["samples_in_epoch"]
    elif progress.get("format_version") == 2:
        start_epoch = progress.get("dataset_start_epoch")
        start_seen = progress.get("dataset_start_samples_seen")
        parent = progress.get("parent_data_progress")
        if (
            type(start_epoch) is not int
            or not 0 <= start_epoch <= epoch
            or type(start_seen) is not int
            or start_seen < 0
            or not isinstance(parent, dict)
            or parent.get("samples_seen") != start_seen
            or type(parent.get("dataset_size")) is not int
            or parent["dataset_size"] < 1
        ):
            raise ValueError(error)
        _validate_progress(
            parent,
            parent["dataset_size"],
            parent.get("index_sha256"),
            start_epoch,
        )
        expected_seen = (
            start_seen + (epoch - start_epoch) * dataset_size + progress["samples_in_epoch"]
        )
    else:
        raise ValueError(error)
    if progress["samples_seen"] != expected_seen:
        raise ValueError(error)


def stage1_progress(trainer, dataset_size, index_sha256, saved_batch_size=None):
    if type(dataset_size) is not int or dataset_size < 1:
        raise ValueError("Training dataset must be nonempty")
    progress = trainer.data_progress
    if progress is None:
        # Historical checkpoints count updates at one fixed batch size. Convert
        # that position once; subsequent checkpoints store the actual sample cursor.
        batch_size = saved_batch_size or trainer.config.data.batch_size
        steps_per_epoch = math.ceil(dataset_size / batch_size)
        batches = trainer.global_step - trainer.completed_epochs * steps_per_epoch
        if not 0 <= batches <= steps_per_epoch:
            raise ValueError("Cannot infer the legacy checkpoint's data position")
        consumed = min(batches * batch_size, dataset_size)
        progress = {
            "format_version": 1,
            "dataset_size": dataset_size,
            "index_sha256": index_sha256,
            "epoch": trainer.completed_epochs,
            "samples_in_epoch": consumed,
            "samples_seen": trainer.completed_epochs * dataset_size + consumed,
        }
    _validate_progress(progress, dataset_size, index_sha256, trainer.completed_epochs)
    return deepcopy(progress)


def stage2_progress(trainer, dataset_size, index_sha256):
    """Stage 2 starts a new sample stream; resume requires its saved exact cursor.

    Stage-1 epoch/cursor state belongs only in parent provenance. Never infer a
    missing stage-2 cursor from optimizer steps, which can hide a wrong parent or
    repeat records after a partial epoch.
    """
    if type(dataset_size) is not int or dataset_size < 1:
        raise ValueError("Training dataset must be nonempty")
    if not isinstance(index_sha256, str) or not index_sha256:
        raise ValueError("The dataset fingerprint must be provided")
    progress = trainer.data_progress
    if progress is None:
        if trainer.global_step != 0 or trainer.completed_epochs != 0:
            raise ValueError("Stage-2 resume requires its saved data cursor")
        progress = {
            "format_version": 1,
            "dataset_size": dataset_size,
            "index_sha256": index_sha256,
            "epoch": 0,
            "samples_in_epoch": 0,
            "samples_seen": 0,
        }
    _validate_progress(progress, dataset_size, index_sha256, trainer.completed_epochs)
    return deepcopy(progress)


def switch_stage2_dataset(trainer, dataset_size, index_sha256):
    """Start a new dataset at the current epoch without changing training state.

    A stage-2 transition requires its saved exact cursor; unlike stage 1, never
    infer the previous data position from optimizer steps. Keep the abandoned
    partial epoch in lifetime sample accounting and the nested cursor history.
    """
    if type(dataset_size) is not int or dataset_size < 1:
        raise ValueError("Training dataset must be nonempty")
    if not isinstance(index_sha256, str) or not index_sha256:
        raise ValueError("The new dataset fingerprint must be provided")
    saved = trainer.data_progress
    if not isinstance(saved, dict):
        raise ValueError("Stage-2 dataset switching requires its saved data cursor")
    previous = stage2_progress(
        trainer,
        saved.get("dataset_size"),
        saved.get("index_sha256"),
    )
    if previous["dataset_size"] == dataset_size and previous["index_sha256"] == index_sha256:
        raise ValueError(
            "Dataset switching requires a different dataset; resume its cursor instead"
        )
    return {
        "format_version": 2,
        "dataset_size": dataset_size,
        "index_sha256": index_sha256,
        "epoch": trainer.completed_epochs,
        "samples_in_epoch": 0,
        "samples_seen": previous["samples_seen"],
        "dataset_start_epoch": trainer.completed_epochs,
        "dataset_start_samples_seen": previous["samples_seen"],
        "parent_data_progress": previous,
    }


def switch_stage1_dataset(
    trainer,
    dataset_size,
    index_sha256,
    *,
    previous_dataset_size=None,
    previous_index_sha256=None,
    saved_batch_size=None,
):
    """Return a fresh cursor for an explicitly requested dataset switch.

    The caller assigns this to ``trainer.data_progress`` after restoring the
    trainer normally. No model, optimizer, scheduler, scaler, RNG, step, or
    completed-epoch state is changed here. The new dataset starts at the beginning
    of its shuffle for the current epoch; an old dataset's cursor is never reused.
    Lifetime samples seen includes the abandoned partial epoch of the old dataset.

    Legacy checkpoints without a cursor require the old dataset identity and
    size, plus the saved batch size if it differs from the current configuration.
    """
    if type(dataset_size) is not int or dataset_size < 1:
        raise ValueError("Training dataset must be nonempty")
    if not isinstance(index_sha256, str) or not index_sha256:
        raise ValueError("The new dataset fingerprint must be provided")
    saved = trainer.data_progress
    if saved is not None:
        if not isinstance(saved, dict):
            raise ValueError("Saved data cursor does not match this dataset/epoch")
        if previous_dataset_size is None:
            previous_dataset_size = saved.get("dataset_size")
        if previous_index_sha256 is None:
            previous_index_sha256 = saved.get("index_sha256")
    if previous_dataset_size is None or previous_index_sha256 is None:
        raise ValueError("Legacy dataset switching requires the previous dataset size/fingerprint")
    previous = stage1_progress(
        trainer,
        previous_dataset_size,
        previous_index_sha256,
        saved_batch_size,
    )
    if previous_dataset_size == dataset_size and previous_index_sha256 == index_sha256:
        raise ValueError(
            "Dataset switching requires a different dataset; resume its cursor instead"
        )
    return {
        "format_version": 2,
        "dataset_size": dataset_size,
        "index_sha256": index_sha256,
        "epoch": trainer.completed_epochs,
        "samples_in_epoch": 0,
        "samples_seen": previous["samples_seen"],
        "dataset_start_epoch": trainer.completed_epochs,
        "dataset_start_samples_seen": previous["samples_seen"],
        "parent_data_progress": previous,
    }
