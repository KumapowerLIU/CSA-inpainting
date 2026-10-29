import gzip
import hashlib
import importlib.util
import io
import json
import os
import random
import tarfile
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "prepare_places_mix",
    Path(__file__).resolve().parents[1] / "scripts/remote/prepare_places_mix.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
prepare_mix = _module.prepare_mix


def make_archive(path, names, bad_index=None):
    buffer = io.BytesIO()
    Image.new("RGB", (13, 9), (23, 160, 51)).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()
    with tarfile.open(path, "w") as tar:
        for index, name in enumerate(names):
            payload = b"broken jpeg" if index == bad_index else jpeg
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        metadata = tarfile.TarInfo("data_256/README.txt")
        metadata.size = 4
        tar.addfile(metadata, io.BytesIO(b"text"))


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "unsplash"
    source.mkdir()
    # Validation whitespace/CRLF and its missing final newline must survive.
    train = b'["/shard.tfrecord", 0, 22]\n["/shard.tfrecord", 22, 33]'
    validation = b'  ["/shard.tfrecord", 55, 31] \r\n["/other.tfrecord", 0, 25]'
    (source / "train.jsonl").write_bytes(train)
    (source / "validation.jsonl").write_bytes(validation)
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "total": 4,
                "train": 2,
                "validation": 2,
                "index_sha256": "legacy-source-digest",
            }
        )
    )
    archive = tmp_path / "places.tar"
    names = [
        f"data_256/{category[0]}/{category}/{index:04}.jpg"
        for category in ("airfield", "church/indoor", "church/outdoor")
        for index in range(4)
    ]
    make_archive(archive, names)
    return source, archive, names, train, validation


def build(inputs, output, **kwargs):
    source, archive, *_ = inputs
    defaults = {"seed": 20261023, "expected_count": 12, "take": 6, "expected_categories": 3}
    return prepare_mix(archive, source, output, **{**defaults, **kwargs})


def test_uniform_sample_is_reproducible_and_preserves_validation_bytes(inputs, tmp_path):
    source, archive, names, train, validation = inputs
    originals = {path.name: path.read_bytes() for path in source.iterdir()}
    first, second = tmp_path / "first", tmp_path / "second"
    metadata = build(inputs, first, validate_images=True)
    build(inputs, second, validate_images=True)
    for filename in ("train.jsonl", "validation.jsonl", "places-selection.jsonl", "dataset.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert (first / "validation.jsonl").read_bytes() == validation
    mixed_train = (first / "train.jsonl").read_bytes()
    assert mixed_train.startswith(train + b"\n")
    assert metadata["index_sha256"] == hashlib.sha256(mixed_train).hexdigest()
    assert (metadata["total"], metadata["train"], metadata["validation"]) == (10, 8, 2)
    manifest_bytes = (first / "places-selection.jsonl").read_bytes()
    manifest = [json.loads(line) for line in manifest_bytes.splitlines()]
    expected = sorted(random.Random(20261023).sample(range(12), 6))
    assert [row["ordinal"] for row in manifest] == expected
    assert [row["name"] for row in manifest] == [names[index] for index in expected]
    assert metadata["selection"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert metadata["sources"]["places365_standard"]["validated_images"] == 6
    categories = metadata["sources"]["places365_standard"]["category_counts"]
    assert len(categories) == 3
    assert all(count["full"] == 4 for count in categories.values())
    assert sum(count["selected"] for count in categories.values()) == 6
    assert {path.name: path.read_bytes() for path in source.iterdir()} == originals
    selected_rows = [json.loads(line) for line in mixed_train.splitlines()][2:]
    with archive.open("rb") as stream:
        for row in selected_rows:
            assert row["path"] == str(archive)
            stream.seek(row["offset"])
            with Image.open(io.BytesIO(stream.read(row["length"]))) as image:
                image.load()
                assert image.format == "JPEG"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"expected_count": 13}, "JPEG count"),
        ({"expected_count": 11}, "JPEG count"),
        ({"expected_categories": 4}, "category count"),
        ({"take": 13}, "exceed"),
    ],
)
def test_count_or_category_mismatch_never_publishes_partial_output(inputs, tmp_path, kwargs, match):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        build(inputs, output, **kwargs)
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid.building-*"))


def test_missing_jpeg_member_fails_exact_count(inputs, tmp_path):
    _, archive, names, *_ = inputs
    make_archive(archive, names[:-1])
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="JPEG count"):
        build(inputs, output)
    assert not output.exists()


def test_bad_selected_image_aborts_full_validation(inputs, tmp_path):
    _, archive, names, *_ = inputs
    selected = random.Random(20261023).sample(range(12), 6)
    make_archive(archive, names, bad_index=selected[0])
    output = tmp_path / "bad-jpeg"
    output.mkdir()
    with pytest.raises(ValueError, match="Cannot decode selected JPEG"):
        build(inputs, output, validate_images=True)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".bad-jpeg.building-*"))


def test_existing_nonempty_output_is_untouched(inputs, tmp_path):
    output = tmp_path / "already-built"
    output.mkdir()
    (output / "keep.txt").write_text("existing results")
    with pytest.raises(ValueError, match="absent or empty"):
        build(inputs, output)
    assert (output / "keep.txt").read_text() == "existing results"
    assert list(output.iterdir()) == [output / "keep.txt"]


@pytest.mark.parametrize("fault,match", [("train_count", "train count"), ("overlap", "overlap")])
def test_source_count_and_split_overlap_are_checked(inputs, tmp_path, fault, match):
    source, *_ = inputs
    if fault == "train_count":
        metadata = json.loads((source / "dataset.json").read_text())
        metadata["train"] = 3
        (source / "dataset.json").write_text(json.dumps(metadata))
    else:
        (source / "validation.jsonl").write_bytes(
            b'["/shard.tfrecord", 0, 22]\n["/other.tfrecord", 0, 25]\n',
        )
    output = tmp_path / "bad-source"
    with pytest.raises(ValueError, match=match):
        build(inputs, output)
    assert not output.exists()


def test_external_md5_is_recorded_only_as_caller_supplied(inputs, tmp_path):
    metadata = build(inputs, tmp_path / "with-md5", verified_md5="a" * 32)
    verification = metadata["sources"]["places365_standard"]["external_archive_verification"]
    assert verification["md5"] == "a" * 32
    assert "did not compute or verify" in verification["provenance"]


def test_compressed_archive_is_rejected_without_partial_output(inputs, tmp_path):
    source, archive, *_ = inputs
    compressed = tmp_path / "places.tar.gz"
    with gzip.open(compressed, "wb") as stream:
        stream.write(archive.read_bytes())
    output = tmp_path / "compressed-output"
    with pytest.raises(tarfile.ReadError):
        prepare_mix(compressed, source, output, expected_count=12, take=6, expected_categories=3)
    assert not output.exists()
    assert not list(tmp_path.glob(".compressed-output.building-*"))


def test_buffered_scan_avoids_per_member_raw_seeks(inputs, tmp_path, monkeypatch):
    _, archive, *_ = inputs
    contents = archive.read_bytes()

    class CountingRaw(io.BytesIO):
        reads = 0
        seeks = 0

        def readinto(self, buffer):
            self.reads += 1
            return super().readinto(buffer)

        def seek(self, offset, whence=0):
            self.seeks += 1
            return super().seek(offset, whence)

    raw = CountingRaw(contents)
    original_open = Path.open
    archive_opens = []

    def counted_open(path, mode="r", buffering=-1, *args, **kwargs):
        if path == archive:
            archive_opens.append((mode, buffering))
            return io.BufferedReader(raw, buffer_size=buffering)
        return original_open(path, mode, buffering, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    metadata = build(inputs, tmp_path / "buffered", validate_images=False, io_workers=0)
    assert metadata["sources"]["places365_standard"]["selected_train"] == 6
    assert archive_opens == [("rb", 8 * 1024**2)]
    # This tiny archive fits in one prefetch. TAR's per-member logical seeks
    # should never reach the underlying filesystem, and no validation handle opens.
    assert raw.reads == 1
    assert raw.seeks == 0


@pytest.mark.parametrize("workers", [2, 4])
def test_prefetch_supports_buffered_reads_seeks_and_resource_close(tmp_path, workers):
    contents = bytes(range(256)) * 41 + b"final bytes"
    path = tmp_path / "bytes.bin"
    path.write_bytes(contents)
    raw = _module.PrefetchReader(path, workers=workers, block_size=127)
    fd = raw._fd
    with io.BufferedReader(raw, buffer_size=333) as stream:
        assert stream.read(511) == contents[:511]
        assert stream.seek(1000) == 1000
        assert stream.read(73) == contents[1000:1073]
        assert stream.seek(-21, os.SEEK_END) == len(contents) - 21
        assert stream.read() == contents[-21:]
        assert stream.seek(0) == 0
        assert stream.read() == contents
        assert stream.read(1) == b""
        assert stream.seek(len(contents) + 100) == len(contents) + 100
        assert stream.read(1) == b""
        threads = list(raw._executor._threads)
    assert raw.closed and not raw._pending
    assert all(not thread.is_alive() for thread in threads)
    with pytest.raises(OSError):
        os.fstat(fd)
    with pytest.raises(ValueError, match="closed"):
        raw.readinto(bytearray(1))
    raw.close()  # A second close must remain harmless.


@pytest.mark.parametrize("workers", [2, 4])
def test_prefetch_cache_and_in_flight_work_stay_bounded(tmp_path, monkeypatch, workers):
    path = tmp_path / "bounded.bin"
    contents = bytes(range(256)) * 100
    path.write_bytes(contents)
    original_pread = os.pread
    lock = threading.Lock()
    active = peak = 0

    def counted_pread(fd, length, offset):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.001)
            return original_pread(fd, length, offset)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(os, "pread", counted_pread)
    with _module.PrefetchReader(path, workers=workers, block_size=256) as reader:
        for offset in (0, 512, 12000, 256, 24000, 9000):
            reader.seek(offset)
            result = bytearray(111)
            assert reader.readinto(result) == 111
            assert result == contents[offset : offset + 111]
            assert len(reader._pending) <= workers
            cached = [future.result() for future in reader._pending.values() if future.done()]
            assert sum(map(len, cached)) <= workers * 256
    assert 1 < peak <= workers
    assert active == 0


@pytest.mark.parametrize("short", [False, True])
def test_prefetch_read_failures_propagate_and_close_resources(tmp_path, monkeypatch, short):
    path = tmp_path / "read-error.bin"
    path.write_bytes(bytes(range(256)) * 4)
    original_pread = os.pread

    def failing_pread(fd, length, offset):
        if offset == 0:
            if short:
                return original_pread(fd, length - 1, offset)
            raise OSError("injected storage error")
        return original_pread(fd, length, offset)

    monkeypatch.setattr(os, "pread", failing_pread)
    raw = _module.PrefetchReader(path, workers=2, block_size=128)
    fd = raw._fd
    match = "Short archive read" if short else "injected storage error"
    with pytest.raises(OSError, match=match), raw:
        raw.readinto(bytearray(100))
    assert raw.closed and not raw._pending
    assert all(not thread.is_alive() for thread in raw._executor._threads)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_prefetched_tar_has_identical_indexes_to_synchronous_tar(inputs, tmp_path):
    synchronous, prefetched = tmp_path / "sync", tmp_path / "prefetch"
    build(inputs, synchronous, io_workers=0)
    build(inputs, prefetched, io_workers=4)
    for name in ("train.jsonl", "validation.jsonl", "places-selection.jsonl", "dataset.json"):
        assert (synchronous / name).read_bytes() == (prefetched / name).read_bytes()
