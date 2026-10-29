import importlib.util
import io
import json
import struct
import tarfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from csa_inpainting.config import DataConfig

REMOTE = Path(__file__).resolve().parents[1] / "scripts/remote"
_spec = importlib.util.spec_from_file_location("remote_record_images", REMOTE / "record_images.py")
_reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader)
read_record_image = _reader.read_record_image


def no_legacy_decoder(_payload):
    raise AssertionError("New image formats must not invoke the legacy pickle decoder")


@pytest.fixture
def image_records(tmp_path):
    image = Image.new("RGB", (19, 13), (14, 180, 62))
    # Asymmetric contents also exercise the shared horizontal-flip behavior.
    image.paste((210, 51, 80), (0, 0, 7, 13))
    jpeg = io.BytesIO()
    image.save(jpeg, format="JPEG")
    payload = jpeg.getvalue()
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(payload)
    archive = tmp_path / "samples.tar"
    name = "scenes/sample.jpg"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    with tarfile.open(archive, "r:") as stream:
        member = stream.getmember(name)
        offset = member.offset_data
    tfrecord = tmp_path / "sample.tfrecord"
    tfrecord.write_bytes(b"prefix" + struct.pack("<Q", len(payload)) + b"CRC!" + payload + b"CRC!")
    return payload, [
        [str(tfrecord), 6, len(payload) + 16],
        {"format": "image", "path": str(image_path)},
        {
            "format": "tar",
            "path": str(archive),
            "offset": offset,
            "length": len(payload),
            "name": name,
        },
    ]


def test_all_formats_read_identical_jpeg_bytes_and_stable_ids(image_records):
    payload, records = image_records
    decoded_payloads = []

    def legacy_decoder(value):
        decoded_payloads.append(value)
        return value

    for record in records:
        decoder = legacy_decoder if isinstance(record, list) else no_legacy_decoder
        image_bytes, stable_id = read_record_image(record, decoder)
        assert image_bytes == payload
        assert isinstance(stable_id, str) and stable_id
        assert read_record_image(record, decoder) == (image_bytes, stable_id)
        with Image.open(io.BytesIO(image_bytes)) as image:
            assert image.size == (19, 13)
    assert decoded_payloads == [payload, payload]
    assert read_record_image(records[0], legacy_decoder)[1] == f"{records[0][0]}:6"


@pytest.mark.parametrize(
    "changes",
    [
        {"offset": -1},
        {"offset": True},
        {"offset": 1.5},
        {"offset": 2**80},
        {"length": 0},
        {"length": -1},
        {"length": True},
        {"length": "20"},
        {"length": 64 * 1024**2 + 1},
        {"name": ""},
        {"name": None},
        {"path": "relative.tar"},
        {"format": "zip"},
        {"format": []},
    ],
)
def test_invalid_tar_index_is_rejected(image_records, changes):
    _, records = image_records
    with pytest.raises(ValueError):
        read_record_image({**records[2], **changes}, no_legacy_decoder)


def test_tar_short_read_is_reported(image_records):
    _, records = image_records
    record = records[2]
    with open(record["path"], "r+b") as stream:
        stream.truncate(record["offset"] + record["length"] - 1)
    with pytest.raises(ValueError, match="Truncated TAR member"):
        read_record_image(record, no_legacy_decoder)


@pytest.mark.parametrize("size,match", [(10, "header"), (20, "Truncated TFRecord")])
def test_legacy_short_read_is_reported(image_records, size, match):
    _, records = image_records
    record = records[0]
    with open(record[0], "r+b") as stream:
        stream.truncate(size)
    with pytest.raises(ValueError, match=match):
        read_record_image(record, no_legacy_decoder)


def test_legacy_length_mismatch_and_invalid_rows(image_records):
    _, records = image_records
    path, offset, length = records[0]
    invalid = [
        [path, offset, length + 1],
        [path, -1, length],
        [path, offset, 2],
        [path, offset],
        None,
        "invalid",
    ]
    for record in invalid:
        with pytest.raises(ValueError):
            read_record_image(record, no_legacy_decoder)


def test_empty_image_and_missing_files_are_reported(tmp_path):
    path = tmp_path / "empty.jpg"
    path.touch()
    with pytest.raises(ValueError, match="empty"):
        read_record_image({"format": "image", "path": str(path)}, no_legacy_decoder)
    with pytest.raises(FileNotFoundError):
        read_record_image({"format": "image", "path": str(path) + "missing"}, no_legacy_decoder)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REMOTE))
    spec = importlib.util.spec_from_file_location(
        "remote_mixed_runner", REMOTE / "csa_remote_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "raw_image", lambda payload: payload)
    return module


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mask_mode", ["random", "brush", "center"])
def test_dataset_preprocessing_and_masks_match_across_formats(
    tmp_path,
    image_records,
    runner,
    training,
    mask_mode,
):
    _, records = image_records
    config = DataConfig(image_size=64, mask_mode=mask_mode, horizontal_flip=True)
    samples = []
    for number, record in enumerate(records):
        index = tmp_path / f"index-{number}.jsonl"
        index.write_text(json.dumps(record) + "\n")
        dataset = runner.RecordDataset(index, training, config, seed=42)
        dataset.set_epoch(7)
        assert dataset.epoch == (7 if training else 0)
        samples.append(dataset[0])
        assert torch.equal(dataset[0]["mask"], samples[-1]["mask"])
    for sample in samples[1:]:
        assert torch.equal(sample["image"], samples[0]["image"])
        assert torch.equal(sample["mask"], samples[0]["mask"])
    assert len({sample["path"] for sample in samples}) == 3


def test_dataset_bad_rows_include_index_location(tmp_path, runner):
    index = tmp_path / "invalid.jsonl"
    index.write_text('{"format": "tar"}\n')
    dataset = runner.RecordDataset(index, False)
    with pytest.raises(ValueError, match=r"invalid.jsonl, row 0"):
        dataset[0]
    index.write_text("invalid-json\n")
    with pytest.raises(ValueError, match=r"invalid.jsonl, line 1"):
        runner.RecordDataset(index, False)
