import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import InpaintingDataset, read_mask, rgb_tensor
from .engine import Trainer, choose_device, load_generator, seed_everything
from .evaluation import evaluate, save_image
from .experiment import check_output_directory, record_invocation
from .masks import center_mask


def train(args: argparse.Namespace) -> None:
    config = Config.load(args.config, args.set)
    if args.device:
        config.train.device = args.device
    config.validate()
    if args.dry_run:
        print(yaml.safe_dump(config.as_dict(), sort_keys=False), end="")
        return
    destination = check_output_directory(config.train.output, args.resume)
    seed_everything(config.train.seed)
    dataset = InpaintingDataset(config.data, training=True, seed=config.train.seed)
    validation = (
        InpaintingDataset(config.data, training=False, seed=config.train.seed)
        if config.data.val_images
        else None
    )
    trainer = Trainer(config, restoring=bool(args.resume))
    if args.resume:
        trainer.restore(args.resume)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
    record_invocation(
        destination, args.config, args.set, args.resume, trainer.device, config.as_dict()
    )
    print(
        json.dumps(
            {
                "torch": str(torch.__version__),
                "device": str(trainer.device),
                "images": len(dataset),
                "batch_size": config.data.batch_size,
                "start_epoch": trainer.completed_epochs,
            }
        ),
        flush=True,
    )
    for epoch in range(trainer.completed_epochs, config.train.epochs):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.workers,
            pin_memory=trainer.device.type == "cuda",
            generator=torch.Generator().manual_seed(config.train.seed + epoch),
            drop_last=False,
        )
        for batch in loader:
            losses = trainer.step(batch["image"], batch["mask"])
            if trainer.global_step % config.train.log_every == 0 or trainer.global_step == 1:
                record = {"epoch": epoch + 1, "step": trainer.global_step, **losses}
                print(json.dumps(record), flush=True)
                with (destination / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        trainer.finish_epoch()
        trainer.save(destination / "latest.pt")
        if validation is not None:
            summary = evaluate(
                trainer.generator,
                DataLoader(
                    validation, batch_size=config.data.batch_size, num_workers=config.data.workers
                ),
                trainer.device,
                destination / f"validation-{epoch + 1:04d}",
            )
            print(json.dumps({"epoch": epoch + 1, "validation": summary}), flush=True)


def evaluate_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, config = load_generator(args.checkpoint, device)
    if args.images:
        config.data.val_images = args.images
    if args.masks:
        config.data.val_masks = args.masks
        config.data.mask_mode = "files"
    if args.mask_mode:
        config.data.mask_mode = args.mask_mode
    if args.black_is_hole:
        config.data.mask_white_is_hole = False
    dataset = InpaintingDataset(config.data, training=False, seed=config.train.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    summary = evaluate(model, loader, device, args.output, args.save_limit)
    print(json.dumps(summary, indent=2))


def export_command(args: argparse.Namespace) -> None:
    from .inference import export_checkpoint

    metadata = export_checkpoint(args.checkpoint, args.output)
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


@torch.inference_mode()
def infer(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, _ = load_generator(args.checkpoint, device)
    with Image.open(args.image) as source:
        image = rgb_tensor(ImageOps.exif_transpose(source))
    h, w = image.shape[-2:]
    with Image.open(args.mask) as mask_source:
        if ImageOps.exif_transpose(mask_source).size != (w, h):
            raise ValueError("Inference mask must have the same dimensions as the input image")
    mask = read_mask(args.mask, (w, h), not args.black_is_hole)
    if bool(mask.all()):
        raise ValueError("A fully missing image has no context for CSA")
    divisor = 2**model.depth
    ph, pw = (-h) % divisor, (-w) % divisor
    padded_image = F.pad(image.unsqueeze(0), (0, pw, 0, ph), mode="replicate").to(device)
    # Replicate both image and mask; do not turn unknown padding into valid context.
    padded_mask = F.pad(mask.float().unsqueeze(0), (0, pw, 0, ph), mode="replicate")
    output = model(padded_image, padded_mask.bool().to(device))
    save_image(output.completed[0, :, :h, :w], args.output)
    print(f"Saved {args.output}")


def smoke(args: argparse.Namespace) -> None:
    if args.batch_size < 2 or args.steps < 1:
        raise ValueError("Smoke requires batch_size>=2 and steps>=1")
    config = Config.from_dict(
        {
            "model": {"width": 4, "depth": 6},
            "data": {"image_size": 64, "batch_size": args.batch_size, "workers": 0},
            "train": {
                "device": args.device,
                "epochs": 2,
                "decay_start": 1,
                "discriminator_width": 4,
                "feature_discriminator_width": 16,
                "output": args.output,
                "amp": args.amp,
            },
        }
    )
    seed_everything(config.train.seed)
    trainer = Trainer(config, test_fixture=True)
    image = torch.rand(args.batch_size, 3, 64, 64) * 2 - 1
    masks = torch.stack(
        [torch.roll(center_mask(64, 64), shifts=i * 5, dims=-1) for i in range(args.batch_size)]
    )
    for _ in range(args.steps):
        losses = trainer.step(image, masks)
        if not all(math.isfinite(value) for value in losses.values()):
            raise FloatingPointError("Smoke produced nonfinite losses")
    destination = Path(args.output)
    trainer.finish_epoch()
    trainer.save(destination / "smoke.pt")
    restored = Trainer(config, test_fixture=True, restoring=True)
    restored.restore(destination / "smoke.pt")
    trainer.generator.eval()
    restored.generator.eval()
    with torch.inference_mode():
        expected = trainer.generator(image.to(trainer.device), masks.to(trainer.device)).completed
        actual = restored.generator(image.to(trainer.device), masks.to(trainer.device)).completed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for i in range(args.batch_size):
        save_image(torch.cat((image[i], actual[i].cpu()), dim=-1), destination / f"smoke-{i}.png")
    report = {
        "purpose": "Synthetic software smoke test; NOT a trained inpainting result",
        "torch": str(torch.__version__),
        "device": str(trainer.device),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "checkpoint_roundtrip_exact": True,
        "losses": losses,
    }
    (destination / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CSA image inpainting")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Train with pretrained VGG supervision")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--resume")
    p.add_argument("--device")
    p.add_argument(
        "--set",
        nargs="+",
        action="extend",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override known YAML settings; may be repeated",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved config without loading data or model weights",
    )
    p.set_defaults(run=train)
    p = commands.add_parser("evaluate", help="Evaluate a checkpoint on a fixed validation set")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images")
    p.add_argument("--masks")
    p.add_argument("--mask-mode", choices=["random", "center", "files", "brush"])
    p.add_argument("--black-is-hole", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save-limit", type=int, default=8)
    p.add_argument("--output", default="runs/evaluation")
    p.set_defaults(run=evaluate_command)
    p = commands.add_parser("infer", help="Inpaint one image, preserving its dimensions")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--black-is-hole", action="store_true")
    p.add_argument("--device", default="auto")
    p.set_defaults(run=infer)
    p = commands.add_parser("export", help="Export a portable stage-2 inference checkpoint on CPU")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--output",
        required=True,
        help="New output file; existing files are never replaced",
    )
    p.set_defaults(run=export_command)
    p = commands.add_parser("smoke", help="Offline synthetic check, no dataset or VGG download")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    p.add_argument("--output", default="runs/smoke")
    p.set_defaults(run=smoke)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
