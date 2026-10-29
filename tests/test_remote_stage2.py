"""Stage-2 runner boundaries without CUDA, pretrained weights, or real training."""

import contextlib
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from csa_inpainting.config import Config

REMOTE = Path(__file__).parents[1] / "scripts/remote"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REMOTE))
    # These tests exercise scheduling and provenance, not image metrics. Keep
    # the runner import usable in the small CPU test environment without skimage.
    monkeypatch.setitem(
        sys.modules,
        "csa_inpainting.evaluation",
        SimpleNamespace(evaluate=lambda *args, **kwargs: {}),
    )
    spec = importlib.util.spec_from_file_location(
        "remote_stage2_runner",
        REMOTE / "csa_remote_train.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(runner, *extra, mode="train"):
    return runner.parse_args([mode, "--runtime", "/runtime", "--stage", "stage2", *extra])


def test_stage2_requires_one_parent_and_preserves_existing_defaults(runner):
    for mode in ("train", "probe"):
        with pytest.raises(SystemExit):
            arguments(runner, mode=mode)
        with pytest.raises(SystemExit):
            arguments(runner, "--init-stage1", "/parent.pt", "--resume", "/latest.pt", mode=mode)
        args = arguments(runner, "--init-stage1", "/parent.pt", mode=mode)
        assert (args.epochs, args.decay_start, args.log_every) == (120, 20, 10)
        assert (args.reconstruction_weight, args.lpips_weight) == (1.0, 0.0)
        assert args.resume is None
    for stage in ("stage1", "joint"):
        with pytest.raises(SystemExit):
            runner.parse_args(
                ["train", "--runtime", "/runtime", "--stage", stage, "--init-stage1", "/parent.pt"]
            )
    # Indexing does not create training state and needs no checkpoint.
    assert arguments(runner, mode="index").init_stage1 is None


def test_stage2_explicit_loss_schedule_and_strict_resume_gates(runner):
    args = arguments(
        runner,
        "--init-stage1",
        "/parent.pt",
        "--reconstruction-weight",
        "0.1",
        "--lpips-weight",
        "1",
        "--epochs",
        "30",
        "--decay-start",
        "10",
        "--log-every",
        "1",
        "--memory-budget-gib",
        "13",
    )
    assert (args.reconstruction_weight, args.lpips_weight) == (0.1, 1.0)
    assert (args.epochs, args.decay_start, args.log_every) == (30, 10, 1)
    assert args.memory_budget_gib == 13
    for extra in (
        ["--allow-loss-change"],
        ["--allow-batch-size-change"],
        ["--diagnostic"],
        ["--log-every", "0"],
        ["--epochs", "20"],
        ["--decay-start", "-1"],
        ["--memory-budget-gib", "nan"],
        ["--batch-size", "0"],
        ["--steps", "0"],
    ):
        with pytest.raises(SystemExit):
            arguments(runner, "--resume", "/latest.pt", *extra)


def test_dataset_change_requires_explicit_stage1_or_stage2_resume(runner):
    for stage in ("stage1", "stage2"):
        args = runner.parse_args(
            [
                "train",
                "--runtime",
                "/runtime",
                "--stage",
                stage,
                "--resume",
                "/latest.pt",
                "--allow-dataset-change",
            ]
        )
        assert args.allow_dataset_change
        with pytest.raises(SystemExit):
            runner.parse_args(
                ["train", "--runtime", "/runtime", "--stage", stage, "--allow-dataset-change"]
            )
    with pytest.raises(SystemExit):
        arguments(runner, "--init-stage1", "/parent.pt", "--allow-dataset-change")
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "train",
                "--runtime",
                "/runtime",
                "--stage",
                "joint",
                "--resume",
                "/latest.pt",
                "--allow-dataset-change",
            ]
        )


def test_realesrgan_requires_explicit_fresh_weights_and_stage2(runner):
    variant = ["--image-discriminator", "realesrgan_unet_sn"]
    with pytest.raises(SystemExit):
        arguments(runner, "--init-stage1", "/parent.pt", *variant)
    args = arguments(
        runner,
        "--init-stage1",
        "/parent.pt",
        *variant,
        "--image-discriminator-pretrained",
        "/netD.pth",
        "--image-discriminator-chunk-size",
        "4",
    )
    assert args.image_discriminator_pretrained == "/netD.pth"
    assert args.image_discriminator_chunk_size == 4
    # A complete stage-2 checkpoint contains the discriminator; no external file is needed.
    resumed = arguments(runner, "--resume", "/latest.pt", *variant)
    assert resumed.image_discriminator_pretrained == ""
    for stage in ("stage1", "joint"):
        with pytest.raises(SystemExit):
            runner.parse_args(["train", "--runtime", "/runtime", "--stage", stage, *variant])
    for extra in (
        ["--image-discriminator-pretrained", "/netD.pth"],
        ["--image-discriminator-chunk-size", "0"],
    ):
        with pytest.raises(SystemExit):
            arguments(runner, "--init-stage1", "/parent.pt", *extra)


def test_each_probe_constructs_and_initializes_its_own_trainer(runner, monkeypatch):
    instances = []

    class FakeTrainer:
        def __init__(self, config, *, restoring):
            self.config, self.restoring = config, restoring
            self.actions = []
            instances.append(self)

        def init_from_stage1(self, path):
            self.actions.append(("init", path))

        def restore(self, path):
            self.actions.append(("restore", path))

    monkeypatch.setitem(
        sys.modules,
        "csa_inpainting.stage2",
        SimpleNamespace(Stage2Trainer=FakeTrainer),
    )
    for batch in (2, 4):
        config = Config()
        config.data.batch_size = batch
        args = arguments(
            runner, "--init-stage1", "/parent.pt", "--batch-size", str(batch), mode="probe"
        )
        trainer = runner.create_trainer(config, args)
        assert trainer.actions == [("init", "/parent.pt")]
        assert not trainer.restoring
    assert instances[0] is not instances[1]
    assert [trainer.config.data.batch_size for trainer in instances] == [2, 4]
    resumed = runner.create_trainer(Config(), arguments(runner, "--resume", "/latest.pt"))
    assert resumed.restoring and resumed.actions == [("restore", "/latest.pt")]


def test_stage2_cursor_is_new_or_exact_and_does_not_infer_from_steps(runner):
    trainer = SimpleNamespace(global_step=0, completed_epochs=0, data_progress=None)
    cursor = runner.stage2_progress(trainer, 5, "digest")
    assert cursor == {
        "format_version": 1,
        "dataset_size": 5,
        "index_sha256": "digest",
        "epoch": 0,
        "samples_in_epoch": 0,
        "samples_seen": 0,
    }
    # A parent checkpoint's counts must not be mistaken for new-stage progress.
    trainer.global_step, trainer.completed_epochs = 154200, 33
    with pytest.raises(ValueError, match="saved data cursor"):
        runner.stage2_progress(trainer, 5, "digest")
    trainer.global_step, trainer.completed_epochs = 37, 2
    cursor.update(epoch=2, samples_in_epoch=3, samples_seen=13)
    trainer.data_progress = cursor
    restored = runner.stage2_progress(trainer, 5, "digest")
    assert restored == cursor and restored is not cursor
    for size, fingerprint in [(6, "digest"), (5, "changed")]:
        with pytest.raises(ValueError, match="cursor"):
            runner.stage2_progress(trainer, size, fingerprint)


def test_stage2_eval_uses_autocast_and_small_batch_without_changing_stage1(runner):
    context = contextlib.nullcontext
    trainer = SimpleNamespace(autocast=context)
    assert runner.evaluation_settings(trainer, "stage2") == (context, 2)
    assert runner.evaluation_settings(trainer, "stage1") == (context, 4)
    assert runner.evaluation_settings(trainer, "joint")[1] == 8


@pytest.mark.parametrize(
    "mode,resume,dataset_case",
    [
        ("train", False, "same"),
        ("train", True, "same"),
        ("probe", False, "same"),
        ("train", True, "changed"),
        ("train", True, "same_with_flag"),
        ("train", True, "changed_without_flag"),
        ("train", True, "switched_with_flag"),
    ],
)
def test_runner_trains_exact_cursor_and_records_stage_boundary(
    runner,
    monkeypatch,
    tmp_path,
    mode,
    resume,
    dataset_case,
):
    runtime = tmp_path / "runtime"
    (runtime / "data").mkdir(parents=True)
    changed = dataset_case in {"changed", "changed_without_flag"}
    already_switched = dataset_case == "switched_with_flag"
    fingerprint, dataset_size = ("mixed", 7) if changed or already_switched else ("digest", 5)
    (runtime / "data/dataset.json").write_text(json.dumps({"index_sha256": fingerprint}))
    contexts, load_calls, instances, budgets = [], [], [], []

    class FakeDataset:
        def __init__(self, index, training, config, seed):
            self.training, self.seed, self.epoch = training, seed, 0

        def __len__(self):
            return dataset_size

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __getitem__(self, index):
            return {
                "image": torch.zeros(3, 8, 8),
                "mask": torch.zeros(1, 8, 8, dtype=torch.bool),
                "path": str(index),
            }

    class FakeTrainer:
        feature_mode = "test_stage2"
        training_semantics = "test_stage2_v1"
        lpips_provenance = None

        def __init__(self, config, *, restoring):
            self.config = config
            self.global_step, self.completed_epochs, self.data_progress = 0, 0, None
            self.generator = torch.nn.Linear(1, 1)
            self.device = torch.device("cpu")
            self.saved = []
            instances.append(self)

        def init_from_stage1(self, path):
            self.parent_provenance = {
                "path": path,
                "sha256": "parent-sha",
                "global_step": 154200,
                "completed_epochs": 33,
            }

        def restore(self, path):
            self.init_from_stage1("/parent.pt")
            self.global_step, self.completed_epochs = 37, 2
            self.data_progress = {
                "format_version": 1,
                "dataset_size": 5,
                "index_sha256": "digest",
                "epoch": 2,
                "samples_in_epoch": 2,
                "samples_seen": 12,
            }
            if already_switched:
                self.data_progress = {
                    "format_version": 2,
                    "dataset_size": 7,
                    "index_sha256": "mixed",
                    "epoch": 2,
                    "samples_in_epoch": 2,
                    "samples_seen": 14,
                    "dataset_start_epoch": 2,
                    "dataset_start_samples_seen": 12,
                    "parent_data_progress": self.data_progress,
                }

        @contextlib.contextmanager
        def autocast(self):
            contexts.append(True)
            try:
                yield
            finally:
                contexts.pop()

        def step(self, image, mask):
            self.global_step += 1
            return {"generator": 0.5}

        def finish_epoch(self):
            self.completed_epochs += 1

        def save(self, path):
            self.saved.append(copy.deepcopy(self.data_progress))
            path.touch()

    def fake_loader(dataset, batch_size, workers, epoch=0, start_sample=0):
        load_calls.append((dataset.training, epoch, start_sample, batch_size))
        for start in range(start_sample, len(dataset), batch_size):
            count = min(batch_size, len(dataset) - start)
            yield {
                "image": torch.zeros(count, 3, 8, 8),
                "mask": torch.zeros(count, 1, 8, 8, dtype=torch.bool),
            }

    def fake_evaluate(generator, data, device, output=None):
        assert contexts == [True]
        assert max(batch["image"].shape[0] for batch in data) <= 2
        return {"hole_psnr": 12.0}

    monkeypatch.setitem(
        sys.modules,
        "csa_inpainting.stage2",
        SimpleNamespace(Stage2Trainer=FakeTrainer),
    )
    monkeypatch.setattr(runner, "RecordDataset", FakeDataset)
    monkeypatch.setattr(runner, "loader", fake_loader)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=32 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", budgets.append)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test_gpu")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    checkpoint = ["--resume", "/latest.pt"] if resume else ["--init-stage1", "/parent.pt"]
    if dataset_case in {"changed", "same_with_flag", "switched_with_flag"}:
        checkpoint.append("--allow-dataset-change")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            mode,
            "--runtime",
            str(runtime),
            "--stage",
            "stage2",
            *checkpoint,
            "--batch-size",
            "2",
            "--epochs",
            "3" if resume else "1",
            "--decay-start",
            "0",
            "--log-every",
            "1",
            "--memory-budget-gib",
            "13",
            "--steps",
            "2",
        ],
    )
    if dataset_case == "changed_without_flag":
        with pytest.raises(ValueError, match="cursor"):
            runner.main()
        assert instances[0].global_step == 37
        assert instances[0].data_progress["samples_in_epoch"] == 2
        assert load_calls == []
        return
    runner.main()
    trainer = instances[0]
    assert budgets == [13 / 32]
    expected_start = (True, 2, 0 if changed else 2, 2) if resume else (True, 0, 0, 2)
    assert load_calls[0] == expected_start
    if mode == "probe":
        report = json.loads((runtime / "probe-b2-bf16.json").read_text())
        assert report["probe_updates_discarded"]
        assert report["evaluation_batch_size"] == 2
        assert report["parent_provenance"]["global_step"] == 154200
        assert report["data_progress"]["samples_seen"] == 0
        assert trainer.global_step == 2 and trainer.completed_epochs == 0
        assert trainer.saved == [] and not (runtime / "run").exists()
        return
    expected_updates = (dataset_size - expected_start[2] + 1) // 2
    assert trainer.global_step == (37 if resume else 0) + expected_updates
    assert trainer.completed_epochs == (3 if resume else 1)
    assert trainer.data_progress["samples_in_epoch"] == 0
    assert trainer.data_progress["samples_seen"] == (
        (19 if changed or already_switched else 15) if resume else 5
    )
    assert trainer.saved[-1] == trainer.data_progress
    manifest = json.loads((runtime / "run/stage2-manifest.json").read_text())
    assert manifest["parent_provenance"]["global_step"] == 154200
    assert manifest["initial_data_progress"]["samples_seen"] == (
        (14 if already_switched else 12) if resume else 0
    )
    if changed or already_switched:
        assert manifest["initial_data_progress"]["format_version"] == 2
        assert manifest["initial_data_progress"]["parent_data_progress"]["samples_in_epoch"] == 2
    else:
        assert manifest["initial_data_progress"]["format_version"] == 1
    assert manifest["resumed_from"] == ("/latest.pt" if resume else None)
    assert json.loads((runtime / "training-state.json").read_text())["state"] == "COMPLETED"
    # A later invocation must preserve the initial manifest and append history.
    report = json.loads((runtime / "run/environment.json").read_text())
    report["resumed_from"] = "/later.pt"
    runner.write_stage2_manifest(runtime / "run", trainer.config, report)
    assert json.loads((runtime / "run/stage2-manifest.json").read_text()) == manifest
    invocations = (runtime / "run/stage2-invocations.jsonl").read_text().splitlines()
    assert len(invocations) == 2 and json.loads(invocations[-1])["resumed_from"] == "/later.pt"
