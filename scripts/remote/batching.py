"""Stable sample cursors and bounded worker transfers for large training batches."""

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset


def legacy_epoch_order(length: int, seed: int, epoch: int = 0) -> list[int]:
    """Match the old runner's shuffled DataLoader order (PyTorch 2.6).

    DataLoader draws its worker base seed before RandomSampler draws randperm,
    using the same generator even when num_workers=0. Preserve that first draw.
    """
    if length < 0 or epoch < 0:
        raise ValueError("length and epoch must be nonnegative")
    generator = torch.Generator().manual_seed(seed + epoch)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.randperm(length, generator=generator).tolist()


def training_loader(
    dataset,
    batch_size: int,
    workers: int,
    epoch: int = 0,
    start_sample: int = 0,
    micro_batch_size: int = 32,
    *,
    pin_memory: bool | None = None,
) -> Iterator[dict[str, Tensor | list[str]]]:
    """Regroup one epoch's remaining samples without skips or repeated records.

    Workers transfer at most 32 images at once and prefetch one microbatch each.
    The parent copies each transfer into a final, optionally pinned CPU buffer;
    it never accumulates a full training batch in multiprocessing shared memory.
    The final partial batch is retained. ``start_sample`` counts trained samples,
    not prefetched records or optimizer updates.
    """
    if batch_size < 1 or micro_batch_size < 1 or workers < 0:
        raise ValueError("batch sizes must be positive and workers nonnegative")
    length = len(dataset)
    if not 0 <= start_sample <= length:
        raise ValueError("start_sample must be within the dataset")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    order = (
        legacy_epoch_order(length, dataset.seed, epoch) if dataset.training else list(range(length))
    )
    remaining = length - start_sample
    if remaining == 0:
        return
    options = {"prefetch_factor": 1} if workers else {}
    micro_loader = DataLoader(
        Subset(dataset, order[start_sample:]),
        batch_size=min(32, micro_batch_size, batch_size),
        num_workers=workers,
        shuffle=False,
        pin_memory=False,
        generator=torch.Generator().manual_seed(dataset.seed + epoch),
        **options,
    )
    # Pin only final buffers in the parent, avoiding a second worker-sized pool.
    use_pin = pin_memory is not False and torch.cuda.is_available()
    micro_iterator = iter(micro_loader)
    current = None
    offset = 0
    while remaining:
        target_size = min(batch_size, remaining)
        images = masks = None
        paths = []
        filled = 0
        while filled < target_size:
            if current is None or offset == current["image"].shape[0]:
                current = next(micro_iterator)
                offset = 0
            if images is None:
                images = torch.empty(
                    (target_size, *current["image"].shape[1:]),
                    dtype=current["image"].dtype,
                    pin_memory=use_pin,
                )
                masks = torch.empty(
                    (target_size, *current["mask"].shape[1:]),
                    dtype=current["mask"].dtype,
                    pin_memory=use_pin,
                )
            count = min(target_size - filled, current["image"].shape[0] - offset)
            images[filled : filled + count].copy_(current["image"][offset : offset + count])
            masks[filled : filled + count].copy_(current["mask"][offset : offset + count])
            paths.extend(current["path"][offset : offset + count])
            filled += count
            offset += count
        remaining -= target_size
        yield {"image": images, "mask": masks, "path": paths}
