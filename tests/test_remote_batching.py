import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

_path = Path(__file__).resolve().parents[1] / "scripts/remote/batching.py"
_spec = importlib.util.spec_from_file_location("remote_batching", _path)
_batching = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batching)
legacy_epoch_order = _batching.legacy_epoch_order
training_loader = _batching.training_loader


class Records(Dataset):
    def __init__(self, length=103, training=True, epoch=0):
        self.length, self.training, self.epoch, self.seed = length, training, epoch, 2019

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.full((3, 8, 8), float(index))
        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask.flatten()[(index * 7 + self.epoch * 11) % 64] = True
        return {"image": image, "mask": mask, "path": str(index)}


def old_loader(dataset, batch_size, workers, epoch):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        generator=torch.Generator().manual_seed(dataset.seed + epoch),
    )


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("epoch", [0, 3])
def test_order_and_sample_contents_match_original_loader(workers, epoch):
    dataset = Records(epoch=epoch)
    original = list(old_loader(dataset, 32, workers, epoch))
    new = list(training_loader(dataset, 32, workers, epoch, pin_memory=False))
    expected_order = [int(path) for batch in original for path in batch["path"]]
    assert legacy_epoch_order(len(dataset), dataset.seed, epoch) == expected_order
    for old_batch, new_batch in zip(original, new, strict=True):
        assert old_batch["path"] == new_batch["path"]
        assert torch.equal(old_batch["image"], new_batch["image"])
        assert torch.equal(old_batch["mask"], new_batch["mask"])
        assert not new_batch["image"].is_shared()
        assert not new_batch["mask"].is_shared()


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("start,batch_size", [(64, 33), (96, 64), (0, 7), (103, 32)])
def test_rebatch_preserves_cursor_and_tail(workers, start, batch_size):
    dataset = Records(epoch=2)
    original = list(old_loader(dataset, 32, workers, 2))
    original_paths = [path for batch in original for path in batch["path"]]
    batches = list(
        training_loader(
            dataset, batch_size, workers, 2, start, micro_batch_size=8, pin_memory=False
        )
    )
    paths = [path for batch in batches for path in batch["path"]]
    assert original_paths[:start] + paths == original_paths
    assert len(set(original_paths[:start] + paths)) == len(dataset)
    if batches:
        assert all(len(batch["path"]) == batch_size for batch in batches[:-1])
        assert len(batches[-1]["path"]) == (len(dataset) - start - 1) % batch_size + 1
    for batch in batches:
        for i, path in enumerate(batch["path"]):
            expected = dataset[int(path)]
            assert torch.equal(batch["image"][i], expected["image"])
            assert torch.equal(batch["mask"][i], expected["mask"])


def test_validation_is_sequential_and_tail_is_retained():
    dataset = Records(length=35, training=False)
    batches = list(training_loader(dataset, 32, 2, epoch=7, pin_memory=False))
    assert [path for batch in batches for path in batch["path"]] == list(map(str, range(35)))
    assert [len(batch["path"]) for batch in batches] == [32, 3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"workers": -1},
        {"micro_batch_size": 0},
        {"start_sample": -1},
        {"start_sample": 104},
    ],
)
def test_invalid_loader_request(kwargs):
    arguments = {"batch_size": 32, "workers": 0, **kwargs}
    with pytest.raises(ValueError):
        list(training_loader(Records(), **arguments))
