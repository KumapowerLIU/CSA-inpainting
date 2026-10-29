"""Single-GPU CSA runner for indexed Unsplash TFRecords; no dataset unpickling globals."""

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from batching import training_loader
from PIL import Image, ImageOps
from record_images import read_record_image
from torch.utils.data import DataLoader, Dataset, Subset
from training_progress import (
    stage1_progress,
    stage2_progress,
    switch_stage1_dataset,
    switch_stage2_dataset,
)

from csa_inpainting.config import Config, DataConfig
from csa_inpainting.data import brush_mask, rgb_tensor
from csa_inpainting.engine import TRAINING_SEMANTICS, Trainer, seed_everything
from csa_inpainting.evaluation import evaluate
from csa_inpainting.experiment import check_output_directory
from csa_inpainting.masks import center_mask, random_mask
from csa_inpainting.stage1 import Stage1Trainer, load_stage1_checkpoint


def varint(data, pos):
    value, shift = 0, 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("Invalid protobuf varint")


def fields(data):
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        wire = tag & 7
        if wire == 2:
            size, pos = varint(data, pos)
            value = data[pos : pos + size]
            pos += size
        elif wire == 0:
            value, pos = varint(data, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[pos : pos + size]
            pos += size
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield tag >> 3, value


class BytesOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Pickle globals are forbidden")


def raw_image(record):
    features = dict(fields(record))[1]
    for _, entry in fields(features):
        mapping = dict(fields(entry))
        if mapping[1] == b"state_dict":
            byte_list = dict(fields(mapping[2]))[1]
            payload = dict(fields(byte_list))[1]
            state = BytesOnlyUnpickler(io.BytesIO(payload)).load()
            value = state["raw_image"]
            if not isinstance(value, bytes):
                raise TypeError("raw_image is not bytes")
            return value
    raise ValueError("Missing state_dict")


def make_index(root, output):
    records, digest = [], hashlib.sha256()
    for shard in sorted(Path(root).glob("*.tfrecord")):
        idx = shard.with_suffix(".tfindex").read_bytes()
        digest.update(shard.name.encode() + idx)
        for line in idx.splitlines():
            offset, length = map(int, line.split())
            records.append((str(shard), offset, length))
    if len(records) < 3000:
        raise ValueError("Unexpectedly small dataset")
    order = list(range(len(records)))
    random.Random(2019).shuffle(order)
    val_ids = set(order[:2000])
    train = [r for i, r in enumerate(records) if i not in val_ids]
    val = [r for i, r in enumerate(records) if i in val_ids]
    metadata = dict(
        total=len(records),
        train=len(train),
        validation=len(val),
        seed=2019,
        index_sha256=digest.hexdigest(),
        split="fixed record-level split; not content deduplicated",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2))
    for name, rows in [("train", train), ("validation", val)]:
        (output / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(metadata), flush=True)


class RecordDataset(Dataset):
    def __init__(self, index, training, config=None, seed=2019):
        self.index_path = str(index)
        self.records = []
        for line_number, line in enumerate(Path(index).read_text().splitlines(), start=1):
            try:
                self.records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid image index {index}, line {line_number}: {exc}") from exc
        self.training, self.epoch = training, 0
        self.config = config if config is not None else DataConfig()
        self.seed = seed
        if self.config.mask_mode not in {"center", "brush", "random"}:
            raise ValueError("TFRecord runner supports center, brush and random masks")

    def set_epoch(self, epoch):
        self.epoch = epoch if self.training else 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        try:
            image_bytes, path = read_record_image(self.records[index], raw_image)
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                size = self.config.image_size
                image = image.resize((size, size), Image.Resampling.BICUBIC)
                rng = random.Random(self.seed + self.epoch * 1000003 + index * 97)
                if self.training and self.config.horizontal_flip and rng.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = rgb_tensor(image)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise ValueError(
                f"Cannot read image index {self.index_path}, row {index}: {exc}"
            ) from exc
        if self.config.mask_mode == "random":
            mask = random_mask(
                size, size, rng, self.config.random_min_fraction, self.config.random_max_fraction
            )
        elif self.config.mask_mode == "brush":
            mask = brush_mask(size, rng)
        else:
            mask = center_mask(size, size, self.config.center_overlap)
        return {"image": image, "mask": mask, "path": path}


def loader(dataset, batch_size, workers, epoch=0, start_sample=0):
    return training_loader(dataset, batch_size, workers, epoch, start_sample)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["index", "probe", "train"])
    p.add_argument("--source")
    p.add_argument("--runtime", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", default="bf16")
    p.add_argument("--steps", type=int, default=3)
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume")
    checkpoint.add_argument(
        "--init-stage1", help="Stage 2 only: initialize frozen rough weights from this checkpoint"
    )
    p.add_argument(
        "--allow-batch-size-change",
        action="store_true",
        help="Stage 1 only: keep training state and resume by sample position",
    )
    p.add_argument(
        "--allow-dataset-change",
        action="store_true",
        help="Stage 1/2 resume: retain training state and restart a changed dataset",
    )
    p.add_argument(
        "--allow-loss-change",
        action="store_true",
        help="Stage 1 resume only: explicitly change the loss and record the transition",
    )
    p.add_argument("--stage", choices=["stage1", "stage2", "joint"], default="joint")
    p.add_argument(
        "--reconstruction-weight",
        type=float,
        default=1.0,
        help="Full-image reconstruction L1 coefficient",
    )
    p.add_argument(
        "--lpips-weight",
        type=float,
        default=0.0,
        help="Stage 1 or 2 completed-image LPIPS coefficient (0 disables LPIPS)",
    )
    p.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="vgg")
    p.add_argument(
        "--lpips-batch-size",
        type=int,
        default=8,
        help="Maximum images per FP32 LPIPS chunk; does not change optimizer batch size",
    )
    p.add_argument(
        "--image-discriminator",
        choices=["patch", "realesrgan_unet_sn"],
        default="patch",
        help="Stage-2 image discriminator architecture",
    )
    p.add_argument(
        "--image-discriminator-pretrained",
        default="",
        help="Official Real-ESRGAN netD checkpoint for fresh stage-2 initialization",
    )
    p.add_argument(
        "--image-discriminator-chunk-size",
        type=int,
        default=8,
        help="Maximum images per checkpointed U-Net discriminator chunk",
    )
    p.add_argument(
        "--memory-budget-gib", type=float, help="Maximum PyTorch allocator budget on a shared GPU"
    )
    p.add_argument("--mask-mode", choices=["random", "center", "brush"], default="random")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--preview-every", type=int, default=500)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--decay-start", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--diagnostic", action="store_true", help="Probe only, with random VGG weights")
    args = p.parse_args(argv)
    if args.save_every < 1 or args.preview_every < 1:
        p.error("Checkpoint and preview intervals must be positive")
    if args.workers < 0:
        p.error("Worker count must be nonnegative")
    if args.batch_size < 1 or args.steps < 1:
        p.error("Batch size and probe steps must be positive")
    if args.epochs < 1 or not 0 <= args.decay_start < args.epochs:
        p.error("Require 0 <= decay-start < epochs")
    if args.log_every < 1:
        p.error("Log interval must be positive")
    if args.memory_budget_gib is not None and (
        not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0
    ):
        p.error("Memory budget must be finite and positive")
    if not math.isfinite(args.reconstruction_weight) or args.reconstruction_weight < 0:
        p.error("Reconstruction weight must be finite and nonnegative")
    if not math.isfinite(args.lpips_weight) or args.lpips_weight < 0:
        p.error("LPIPS weight must be finite and nonnegative")
    if args.lpips_batch_size < 1:
        p.error("LPIPS batch size must be positive")
    if args.image_discriminator_chunk_size < 1:
        p.error("Image discriminator chunk size must be positive")
    if args.image_discriminator != "patch" and args.stage != "stage2":
        p.error("Real-ESRGAN image discriminator requires --stage stage2")
    if args.image_discriminator_pretrained and args.image_discriminator != "realesrgan_unet_sn":
        p.error("Pretrained image discriminator requires --image-discriminator realesrgan_unet_sn")
    if (
        args.image_discriminator == "realesrgan_unet_sn"
        and args.mode != "index"
        and not args.resume
        and not args.image_discriminator_pretrained
    ):
        p.error("Fresh Real-ESRGAN stage 2 requires --image-discriminator-pretrained")
    if args.lpips_weight > 0 and args.stage not in {"stage1", "stage2"}:
        p.error("--lpips-weight is supported only with --stage stage1 or stage2")
    if args.stage == "stage1" and args.diagnostic:
        p.error("Stage 1 does not allow random perceptual weights or --diagnostic")
    if args.stage == "stage2" and args.diagnostic:
        p.error("Stage 2 does not allow random perceptual weights or --diagnostic")
    if args.init_stage1 and args.stage != "stage2":
        p.error("--init-stage1 requires --stage stage2")
    if args.stage == "stage2" and args.mode != "index" and not (args.init_stage1 or args.resume):
        p.error("Stage 2 requires either --init-stage1 or --resume")
    if args.allow_batch_size_change and (args.stage != "stage1" or not args.resume):
        p.error("--allow-batch-size-change requires --stage stage1 and --resume")
    if args.allow_dataset_change and (args.stage not in {"stage1", "stage2"} or not args.resume):
        p.error("--allow-dataset-change requires --stage stage1 or stage2 and --resume")
    if args.allow_loss_change and (args.stage != "stage1" or not args.resume):
        p.error("--allow-loss-change requires --stage stage1 and --resume")
    return args


def create_trainer(config, args):
    """Each probe/train invocation owns fresh training state and its own parent load."""
    if args.stage == "stage1":
        trainer = Stage1Trainer(config, restoring=bool(args.resume))
    elif args.stage == "stage2":
        from csa_inpainting.stage2 import Stage2Trainer

        trainer = Stage2Trainer(config, restoring=bool(args.resume))
    else:
        trainer = Trainer(
            config,
            restoring=bool(args.resume),
            diagnostic="random_vgg" if args.diagnostic else None,
        )
    if args.init_stage1:
        trainer.init_from_stage1(args.init_stage1)
    elif args.resume:
        if args.stage == "stage1":
            trainer.restore(
                args.resume,
                allow_batch_size_change=args.allow_batch_size_change,
                allow_loss_change=args.allow_loss_change,
            )
        else:
            trainer.restore(args.resume)
    return trainer


def evaluation_settings(trainer, stage):
    # Refine/CSA plus two VGG paths need much more memory than rough inference.
    # Keep evaluation under the same AMP policy and bound its independent batch.
    if stage == "stage2":
        return trainer.autocast, 2
    if stage == "stage1":
        return trainer.autocast, 4
    return nullcontext, 8


def write_stage2_manifest(destination, config, report):
    """Keep the original stage boundary, plus append each fresh/resume invocation."""
    entry = {
        "stage": "stage2",
        "created_unix": time.time(),
        "parent_provenance": report["parent_provenance"],
        "initialized_from_stage1": report["initialized_from_stage1"],
        "resumed_from": report["resumed_from"],
        "training_semantics": report["training_semantics"],
        "config": config.as_dict(),
        "dataset": report["dataset"],
        "initial_data_progress": report["data_progress"],
        "source_sha256": report["source_sha256"],
        "runner_sha256": report["runner_sha256"],
        "lpips_provenance": report["lpips_provenance"],
        "image_d_provenance": report.get("image_d_provenance"),
    }
    manifest = destination / "stage2-manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(entry, indent=2))
    with (destination / "stage2-invocations.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")


def main():
    args = parse_args()
    runtime = Path(args.runtime)
    if args.mode == "index":
        make_index(args.source, runtime / "data")
        return
    torch.set_num_threads(8)
    seed_everything(2019)
    if args.memory_budget_gib is None:
        memory_fraction = 0.90
    else:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_fraction = args.memory_budget_gib * 1024**3 / total_memory
        if memory_fraction > 1:
            raise ValueError("Memory budget exceeds the selected GPU capacity")
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    config = Config()
    config.data.batch_size = args.batch_size
    config.data.mask_mode = args.mask_mode
    config.data.workers = args.workers
    config.data.train_images = str(runtime / "data/train.jsonl")
    config.data.val_images = str(runtime / "data/validation.jsonl")
    config.train.device = "cuda:0"
    config.train.amp = args.amp
    config.train.reconstruction_weight = args.reconstruction_weight
    config.train.lpips_weight = args.lpips_weight
    config.train.lpips_net = args.lpips_net
    config.train.lpips_batch_size = args.lpips_batch_size
    config.train.image_discriminator = args.image_discriminator
    config.train.image_discriminator_pretrained = args.image_discriminator_pretrained
    config.train.image_discriminator_chunk_size = args.image_discriminator_chunk_size
    config.train.epochs = args.epochs
    config.train.decay_start = args.decay_start
    config.train.log_every = args.log_every
    config.train.output = str(runtime / "run")
    destination = Path(config.train.output)
    train = RecordDataset(config.data.train_images, True, config.data, config.train.seed)
    if args.mode == "train":
        check_output_directory(destination, args.resume)
    if args.diagnostic and args.mode != "probe":
        raise ValueError("Diagnostic features are forbidden for training")
    trainer = create_trainer(config, args)
    dataset_metadata = json.loads((runtime / "data/dataset.json").read_text())
    if args.stage == "stage1":
        saved_batch_size = None
        if args.resume and trainer.data_progress is None:
            saved_state = load_stage1_checkpoint(args.resume)
            saved_batch_size = saved_state["config"]["data"]["batch_size"]
            del saved_state
        saved_progress = trainer.data_progress
        dataset_changed = saved_progress is not None and (
            saved_progress.get("dataset_size") != len(train)
            or saved_progress.get("index_sha256") != dataset_metadata["index_sha256"]
        )
        if args.allow_dataset_change and (dataset_changed or saved_progress is None):
            trainer.data_progress = switch_stage1_dataset(
                trainer, len(train), dataset_metadata["index_sha256"]
            )
        else:
            trainer.data_progress = stage1_progress(
                trainer, len(train), dataset_metadata["index_sha256"], saved_batch_size
            )
    elif args.stage == "stage2":
        saved_progress = trainer.data_progress
        dataset_changed = isinstance(saved_progress, dict) and (
            saved_progress.get("dataset_size") != len(train)
            or saved_progress.get("index_sha256") != dataset_metadata["index_sha256"]
        )
        if args.allow_dataset_change and (dataset_changed or saved_progress is None):
            trainer.data_progress = switch_stage2_dataset(
                trainer, len(train), dataset_metadata["index_sha256"]
            )
        else:
            trainer.data_progress = stage2_progress(
                trainer, len(train), dataset_metadata["index_sha256"]
            )
    report = dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        batch_size=args.batch_size,
        amp=args.amp,
        feature_mode=trainer.feature_mode,
        stage=args.stage,
        memory_budget_gib=args.memory_budget_gib,
        initialized_from_stage1=args.init_stage1,
        parent_provenance=getattr(trainer, "parent_provenance", None),
        resumed_from=args.resume,
        batch_size_change_allowed=args.allow_batch_size_change,
        dataset_change_allowed=args.allow_dataset_change,
        loss_change_allowed=args.allow_loss_change,
        reconstruction_weight=config.train.reconstruction_weight,
        lpips_weight=config.train.lpips_weight,
        lpips_net=config.train.lpips_net,
        lpips_batch_size=config.train.lpips_batch_size,
        image_discriminator=config.train.image_discriminator,
        image_discriminator_chunk_size=config.train.image_discriminator_chunk_size,
        image_d_provenance=getattr(trainer, "image_d_provenance", None),
        epochs=config.train.epochs,
        decay_start=config.train.decay_start,
        log_every=config.train.log_every,
        lpips_provenance=getattr(trainer, "lpips_provenance", None),
        torch_hub_dir=torch.hub.get_dir(),
        loss_transitions=getattr(trainer, "loss_transitions", []),
        data_progress=getattr(trainer, "data_progress", None),
        training_semantics=getattr(trainer, "training_semantics", TRAINING_SEMANTICS),
        mask_mode=config.data.mask_mode,
        mask_fraction_range=[config.data.random_min_fraction, config.data.random_max_fraction],
        generator_parameters=sum(p.numel() for p in trainer.generator.parameters()),
    )
    report["source_sha256"] = {
        str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((runtime / "source").rglob("*.py"))
    }
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["dataset"] = dataset_metadata
    if args.stage == "stage2":
        report["initial_step"] = trainer.global_step
        report["initial_completed_epochs"] = trainer.completed_epochs
        if hasattr(trainer, "g_optimizer") and hasattr(trainer, "d_optimizer"):
            report["initial_optimizer_state_entries"] = {
                "generator": len(trainer.g_optimizer.state),
                "discriminators": len(trainer.d_optimizer.state),
            }
    print(json.dumps(report), flush=True)
    validation = RecordDataset(config.data.val_images, False, config.data, config.train.seed)
    evaluation_context, evaluation_batch_size = evaluation_settings(trainer, args.stage)
    if args.mode == "probe":
        batch = next(iter(loader(train, args.batch_size, min(config.data.workers, 8))))
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            torch.cuda.synchronize()
            start = time.monotonic()
            loss = trainer.step(batch["image"], batch["mask"])
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            print(json.dumps(dict(probe_step=step + 1, seconds=times[-1], losses=loss)), flush=True)
        with evaluation_context():
            probe_validation = evaluate(
                trainer.generator,
                DataLoader(
                    Subset(validation, range(min(len(validation), evaluation_batch_size))),
                    batch_size=evaluation_batch_size,
                    num_workers=0,
                ),
                trainer.device,
            )
        report.update(
            seconds=times,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            feature_mode=trainer.feature_mode,
            losses=loss,
            probe_validation=probe_validation,
            probe_updates_discarded=True,
            evaluation_batch_size=evaluation_batch_size,
        )
        (runtime / f"probe-b{args.batch_size}-{args.amp}.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
    (destination / "environment.json").write_text(json.dumps(report, indent=2))
    if args.stage == "stage2":
        write_stage2_manifest(destination, config, report)
    stop_requested = []

    def request_stop(signum, _frame):
        stop_requested.append(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def write_status(state, **extra):
        status = dict(
            state=state,
            pid=os.getpid(),
            updated_unix=time.time(),
            step=trainer.global_step,
            **extra,
        )
        temporary = runtime / "training-state.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(runtime / "training-state.json")

    write_status("RUNNING")
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    print(
        json.dumps(
            dict(
                train_images=len(train),
                validation_images=len(validation),
                steps_per_epoch=steps_per_epoch,
            )
        ),
        flush=True,
    )
    for epoch in range(trainer.completed_epochs, config.train.epochs):
        train.set_epoch(epoch)
        cursor_stage = args.stage in {"stage1", "stage2"}
        start_sample = trainer.data_progress["samples_in_epoch"] if cursor_stage else 0
        skip = 0 if cursor_stage else trainer.global_step - epoch * steps_per_epoch
        epoch_loader = loader(train, args.batch_size, config.data.workers, epoch, start_sample)
        ready_time = time.monotonic()
        for batch_id, batch in enumerate(epoch_loader):
            if batch_id < skip:
                continue
            start = time.monotonic()
            data_seconds = start - ready_time
            losses = trainer.step(batch["image"], batch["mask"])
            compute_seconds = time.monotonic() - start
            if cursor_stage:
                trainer.data_progress["samples_in_epoch"] += len(batch["image"])
                trainer.data_progress["samples_seen"] += len(batch["image"])
            if trainer.global_step % config.train.log_every == 0 or trainer.global_step == 1:
                row = dict(
                    epoch=epoch + 1,
                    step=trainer.global_step,
                    seconds=compute_seconds,
                    data_seconds=data_seconds,
                    images_per_second=len(batch["image"]) / (data_seconds + compute_seconds),
                    samples_seen=getattr(trainer, "data_progress", {}).get("samples_seen"),
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                    **losses,
                )
                print(json.dumps(row), flush=True)
                with (destination / "train.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                write_status("RUNNING", epoch=epoch + 1)
            if stop_requested:
                trainer.save(destination / "latest.pt")
                write_status("STOPPED", signal=stop_requested[0])
                raise SystemExit(128 + stop_requested[0])
            if trainer.global_step == 10 or trainer.global_step % args.save_every == 0:
                trainer.save(destination / "latest.pt")
                print(json.dumps(dict(checkpoint_step=trainer.global_step)), flush=True)
            if trainer.global_step == 10 or trainer.global_step % args.preview_every == 0:
                preview_loader = DataLoader(
                    Subset(validation, range(min(64, len(validation)))),
                    batch_size=evaluation_batch_size,
                    num_workers=4,
                    generator=torch.Generator().manual_seed(2019),
                )
                with evaluation_context():
                    summary = evaluate(
                        trainer.generator,
                        preview_loader,
                        trainer.device,
                        destination / f"validation-step-{trainer.global_step:07d}",
                    )
                print(
                    json.dumps(dict(validation_step=trainer.global_step, validation=summary)),
                    flush=True,
                )
            ready_time = time.monotonic()
        trainer.finish_epoch()
        if cursor_stage:
            trainer.data_progress.update(epoch=trainer.completed_epochs, samples_in_epoch=0)
        trainer.save(destination / "latest.pt")
        with evaluation_context():
            summary = evaluate(
                trainer.generator,
                loader(validation, min(evaluation_batch_size, args.batch_size), 4),
                trainer.device,
                destination / f"validation-{epoch + 1:04d}",
            )
        print(json.dumps(dict(epoch=epoch + 1, validation=summary)), flush=True)
    (destination / "COMPLETE").write_text(str(trainer.global_step))
    write_status("COMPLETED")


if __name__ == "__main__":
    main()
