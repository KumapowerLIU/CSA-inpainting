"""Evaluate a pinned stage-2 checkpoint against its own frozen rough model."""

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from csa_remote_train import RecordDataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from csa_inpainting.config import Config
from csa_inpainting.evaluation import metrics, save_image
from csa_inpainting.stage2 import Stage2InpaintingModel, load_stage2_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--memory-budget-gib", type=float, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.memory_budget_gib <= 0:
        parser.error("Batch size and memory budget must be positive")
    runtime = Path(args.runtime)
    output = runtime / ("quick-eval-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    # This shared filesystem rejects hard links and can invalidate an open file
    # when latest.pt is replaced. Copy to local temporary storage, then verify
    # that the source identity stayed unchanged during the copy.
    fd, name = tempfile.mkstemp(prefix="csa-stage2-eval-", suffix=".pt")
    os.close(fd)
    pinned = Path(name)
    try:
        source_path = runtime / "run/latest.pt"
        for attempt in range(3):
            before = source_path.stat()
            try:
                shutil.copyfile(source_path, pinned)
                after = source_path.stat()

                def signature(stat):
                    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

                if signature(before) == signature(after) and pinned.stat().st_size == after.st_size:
                    break
            except OSError:
                if attempt == 2:
                    raise
        else:
            raise RuntimeError("Checkpoint changed during all snapshot attempts")
        digest = hashlib.sha256()
        with pinned.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024**2), b""):
                digest.update(block)
        state = load_stage2_checkpoint(pinned)
        config = Config.from_dict(state["config"])
        step, parent = state["global_step"], state["parent_provenance"]
        progress = state.get("data_progress")
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(
            args.memory_budget_gib * 1024**3 / torch.cuda.get_device_properties(device).total_memory
        )
        model = Stage2InpaintingModel(**asdict(config.model))
        model.load_state_dict(state["generator"], strict=True)
        del state
        model = model.to(device).float().eval()
    finally:
        pinned.unlink()

    validation = RecordDataset(
        runtime / "data/validation.jsonl", False, config.data, config.train.seed
    )
    shown = [582, 605, 1659, 477, 97, 1001, 1492, 1850]
    others = [index for index in range(len(validation)) if index not in shown]
    indices = shown + random.Random(20261023).sample(others, 56)
    loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, num_workers=4)
    rows = []
    for batch in loader:
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        prediction = model(image, mask)
        rough = torch.where(mask, prediction.rough, image)
        assert torch.equal(
            prediction.completed[~mask.expand_as(image)], image[~mask.expand_as(image)]
        )
        rough_metrics = metrics(rough, image, mask)
        refined_metrics = metrics(prediction.completed, image, mask)
        for item, (coarse, refined) in enumerate(zip(rough_metrics, refined_metrics, strict=True)):
            index = indices[len(rows)]
            row = {
                "validation_index": index,
                "path": batch["path"][item],
                "rough": coarse,
                "refined": refined,
                "delta_hole_psnr": refined["hole_psnr"] - coarse["hole_psnr"],
            }
            rows.append(row)
            if index in shown:
                strip = torch.cat(
                    (prediction.masked[item], rough[item], prediction.completed[item], image[item]),
                    dim=-1,
                )
                save_image(strip, output / f"sample-{index:04d}.png")
                Image.fromarray((mask[item, 0].cpu().numpy() * 255).astype(np.uint8)).save(
                    output / f"mask-{index:04d}.png"
                )

    summary = {
        name: {key: float(np.mean([row[name][key] for row in rows])) for key in rows[0][name]}
        for name in ("rough", "refined")
    }
    summary.update(
        images=len(rows),
        improved_hole_psnr=sum(row["delta_hole_psnr"] > 0 for row in rows),
        improved_hole_l1=sum(row["refined"]["hole_l1"] < row["rough"]["hole_l1"] for row in rows),
    )
    (output / "metrics.json").write_text(
        json.dumps({"summary": summary, "per_image": rows}, indent=2) + "\n"
    )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for part in range(2):
        canvas = Image.new("RGB", (1048, 1208), "#20242a")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            f"Stage 2 step {step} | frozen rough step {parent['global_step']}",
            fill="white",
            font=font,
        )
        for column, label in enumerate(
            ("Masked input", "Stage 1 / rough", "Stage 2 / refined", "Ground truth")
        ):
            draw.text((12 + column * 256, 33), label, fill="white", font=font)
        for offset, row in enumerate(rows[part * 4 : (part + 1) * 4]):
            y = 60 + offset * 286
            with Image.open(output / f"sample-{row['validation_index']:04d}.png") as strip:
                canvas.paste(strip, (12, y))
            caption = (
                f"sample {row['validation_index']} | hole PSNR "
                f"{row['rough']['hole_psnr']:.2f} -> {row['refined']['hole_psnr']:.2f} dB "
                f"({row['delta_hole_psnr']:+.2f})"
            )
            draw.text((12, y + 260), caption, fill="white", font=font)
        canvas.save(output / f"comparison-{part + 1}.png")
    report = {
        "stage": 2,
        "checkpoint_step": step,
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_retained": False,
        "parent_provenance": parent,
        "data_progress": progress,
        "validation_indices": indices,
        "shown_indices": shown,
        "mask_mode": config.data.mask_mode,
        "precision": "FP32",
        "inference_batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(device),
        "summary": summary,
        "elapsed_seconds": time.monotonic() - start,
        "note": "Matched images/masks; rough and refined share one checkpoint and forward pass. "
        "64 held-out records; this is an early-training spot check, not convergence evidence.",
    }
    (output / "test.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(__file__, output / "evaluate.py")
    files = sorted(
        [
            *output.glob("*.png"),
            output / "metrics.json",
            output / "test.json",
            output / "evaluate.py",
        ]
    )
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (output / "COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETE", "files": hashes}, indent=2) + "\n"
    )
    print(
        json.dumps({"output": str(output), "checkpoint_step": step, "summary": summary}), flush=True
    )


if __name__ == "__main__":
    main()
