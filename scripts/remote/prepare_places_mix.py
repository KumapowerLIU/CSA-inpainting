"""Build an Unsplash + fixed random Places365-Standard subset without extracting TARs."""

import argparse
import hashlib
import io
import json
import operator
import os
import random
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

from PIL import Image

DEFAULT_COUNT = 1_803_460
DEFAULT_TAKE = 901_730
MAX_IMAGE_BYTES = 64 * 1024**2
ARCHIVE_BLOCK_BYTES = 8 * 1024**2


class PrefetchReader(io.RawIOBase):
    """Seekable read-only file with at most ``workers`` blocks held or in flight."""

    def __init__(self, path, *, workers=8, block_size=ARCHIVE_BLOCK_BYTES):
        super().__init__()
        if type(workers) is not int or workers < 1:
            raise ValueError("Prefetch workers must be a positive integer")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("Prefetch block_size must be a positive integer")
        self.name = str(path)
        self._block_size, self._workers = block_size, workers
        self._position = 0
        self._pending = {}
        self._fd = os.open(path, os.O_RDONLY)
        try:
            self._size = os.fstat(self._fd).st_size
            self._executor = ThreadPoolExecutor(max_workers=workers)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def _ensure_open(self):
        if self.closed:
            raise ValueError("I/O operation on closed prefetch reader")

    def readable(self):
        self._ensure_open()
        return True

    def seekable(self):
        self._ensure_open()
        return True

    def tell(self):
        self._ensure_open()
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        self._ensure_open()
        offset = operator.index(offset)
        if whence == os.SEEK_CUR:
            offset += self._position
        elif whence == os.SEEK_END:
            offset += self._size
        elif whence != os.SEEK_SET:
            raise ValueError("Unsupported seek whence")
        if offset < 0:
            raise ValueError("Negative seek position")
        self._position = offset
        return offset

    def _fetch(self, offset):
        length = min(self._block_size, self._size - offset)
        contents = os.pread(self._fd, length, offset)
        if len(contents) != length:
            raise OSError(f"Short archive read at {offset}: expected {length}, got {len(contents)}")
        return contents

    def _fill_window(self, start):
        desired = set(
            range(
                start, min(self._size, start + self._workers * self._block_size), self._block_size
            )
        )
        stale = [future for offset, future in self._pending.items() if offset not in desired]
        for future in stale:
            future.cancel()
        if stale:
            # Drain displaced work before replacing it. Repeated distant seeks
            # cannot accumulate extra in-flight blocks beyond the window limit.
            wait(stale)
            del future
        self._pending = {
            offset: future for offset, future in self._pending.items() if offset in desired
        }
        stale.clear()
        for offset in sorted(desired):
            if offset not in self._pending:
                self._pending[offset] = self._executor.submit(self._fetch, offset)

    def readinto(self, buffer):
        self._ensure_open()
        view = memoryview(buffer).cast("B")
        copied = 0
        while copied < len(view) and self._position < self._size:
            block_start = self._position // self._block_size * self._block_size
            self._fill_window(block_start)
            contents = self._pending[block_start].result()
            in_block = self._position - block_start
            count = min(len(view) - copied, len(contents) - in_block)
            view[copied : copied + count] = contents[in_block : in_block + count]
            copied += count
            self._position += count
            del contents
        return copied

    def close(self):
        if self.closed:
            return
        try:
            executor = getattr(self, "_executor", None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            getattr(self, "_pending", {}).clear()
        finally:
            fd = getattr(self, "_fd", None)
            if fd is not None:
                os.close(fd)
                self._fd = None
            super().close()


def _archive_reader(path, io_workers):
    if io_workers == 0:
        return path.open("rb", buffering=ARCHIVE_BLOCK_BYTES)
    raw = PrefetchReader(path, workers=io_workers)
    try:
        return io.BufferedReader(raw, buffer_size=ARCHIVE_BLOCK_BYTES)
    except BaseException:
        raw.close()
        raise


def _json_line(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _legacy_rows(contents, label):
    rows = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid {label}, line {line_number}: {exc}") from exc
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or type(row[1]) is not int
            or row[1] < 0
            or type(row[2]) is not int
            or row[2] < 16
        ):
            raise ValueError(f"Invalid legacy TFRecord row in {label}, line {line_number}")
        rows.append(tuple(row))
    return rows


def _source_indexes(directory):
    metadata_bytes = (directory / "dataset.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    train_bytes = (directory / "train.jsonl").read_bytes()
    validation_bytes = (directory / "validation.jsonl").read_bytes()
    train = _legacy_rows(train_bytes, "Unsplash train.jsonl")
    validation = _legacy_rows(validation_bytes, "Unsplash validation.jsonl")
    for split, rows in (("train", train), ("validation", validation)):
        count = metadata.get(split)
        if type(count) is not int or count != len(rows) or count < 1:
            raise ValueError(f"Unsplash {split} count does not match dataset.json")
        if len(set(rows)) != len(rows):
            raise ValueError(f"Duplicate Unsplash {split} index rows")
    if metadata.get("total") != len(train) + len(validation):
        raise ValueError("Unsplash total count does not match dataset.json")
    overlap = set(train).intersection(validation)
    if overlap:
        raise ValueError(f"Unsplash train/validation overlap: {next(iter(overlap))!r}")
    return (
        metadata,
        train_bytes,
        validation_bytes,
        {
            "dataset_json_sha256": _sha256(metadata_bytes),
            "train_jsonl_sha256": _sha256(train_bytes),
            "validation_jsonl_sha256": _sha256(validation_bytes),
        },
    )


def _category(member_name):
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or str(path.parent) == ".":
        raise ValueError(f"JPEG member has no valid category directory: {member_name!r}")
    # Full parent paths retain subtype directories such as church/indoor and
    # church/outdoor. The official archive has one parent directory per class.
    return str(path.parent)


def _validate_image(stream, record):
    stream.seek(record["offset"])
    contents = stream.read(record["length"])
    if len(contents) != record["length"]:
        raise ValueError(f"Truncated selected JPEG: {record['name']}")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.format != "JPEG":
                raise ValueError(f"Expected JPEG bytes, found {image.format}")
            image.load()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode selected JPEG {record['name']}: {exc}") from exc


def prepare_mix(
    archive,
    unsplash_data,
    output,
    *,
    seed=20261023,
    expected_count=DEFAULT_COUNT,
    take=DEFAULT_TAKE,
    expected_categories=365,
    validate_images=False,
    verified_md5=None,
    io_workers=8,
):
    """Create a complete new index directory, or leave no partial output on failure."""
    for name, value in (
        ("seed", seed),
        ("expected_count", expected_count),
        ("take", take),
        ("expected_categories", expected_categories),
    ):
        if type(value) is not int or (name != "seed" and value < 1):
            qualifier = "an" if name == "seed" else "a positive"
            raise ValueError(f"Invalid {name}: expected {qualifier} integer")
    if take > expected_count:
        raise ValueError("take cannot exceed expected_count")
    if type(io_workers) is not int or io_workers < 0:
        raise ValueError("io_workers must be a nonnegative integer")
    if verified_md5 is not None and re.fullmatch(r"[0-9a-fA-F]{32}", verified_md5) is None:
        raise ValueError("verified_md5 must be a 32-character hexadecimal digest")
    archive = Path(archive).resolve(strict=True)
    unsplash_data = Path(unsplash_data).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    source_metadata, train_bytes, validation_bytes, source_hashes = _source_indexes(unsplash_data)
    selected = set(random.Random(seed).sample(range(expected_count), take))
    archive_size = archive.stat().st_size
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        (staging / "validation.jsonl").write_bytes(validation_bytes)
        train_hash, manifest_hash = hashlib.sha256(), hashlib.sha256()
        category_full, category_selected = Counter(), Counter()
        seen, selected_count = 0, 0
        with (
            (staging / "train.jsonl").open("wb") as train_stream,
            (staging / "places-selection.jsonl").open("wb") as manifest_stream,
            # BufferedReader handles nearby header seeks inside its buffer,
            # avoiding both network seeks and _Stream's large-buffer copies.
            _archive_reader(archive, io_workers) as archive_stream,
            tarfile.open(fileobj=archive_stream, mode="r:") as tar,
            archive.open("rb") if validate_images else nullcontext() as images,
        ):
            # Preserve the original training lines, adding a separator only when
            # the source's final JSON line has no newline terminator.
            separator = b"\n" if train_bytes and not train_bytes.endswith(b"\n") else b""
            prefix = train_bytes + separator
            train_stream.write(prefix)
            train_hash.update(prefix)
            while True:
                member = tar.next()
                if member is None:
                    break
                # TarFile otherwise retains millions of TarInfo objects.
                tar.members.clear()
                extension = PurePosixPath(member.name).suffix.lower()
                if not member.isfile() or extension not in {".jpg", ".jpeg"}:
                    continue
                ordinal = seen
                seen += 1
                if seen > expected_count:
                    raise ValueError(f"Archive JPEG count exceeds expected_count={expected_count}")
                category = _category(member.name)
                category_full[category] += 1
                if member.offset_data + member.size > archive_size:
                    raise ValueError(f"Truncated TAR member: {member.name}")
                if ordinal not in selected:
                    continue
                if not 1 <= member.size <= MAX_IMAGE_BYTES:
                    raise ValueError(f"Invalid selected JPEG length: {member.name}: {member.size}")
                record = {
                    "format": "tar",
                    "path": str(archive),
                    "offset": member.offset_data,
                    "length": member.size,
                    "name": member.name,
                }
                if validate_images:
                    _validate_image(images, record)
                row_bytes = _json_line(record)
                train_stream.write(row_bytes)
                train_hash.update(row_bytes)
                manifest_bytes = _json_line(
                    {
                        "ordinal": ordinal,
                        "name": member.name,
                        "category": category,
                        "offset": member.offset_data,
                        "length": member.size,
                    }
                )
                manifest_stream.write(manifest_bytes)
                manifest_hash.update(manifest_bytes)
                category_selected[category] += 1
                selected_count += 1
        if seen != expected_count:
            raise ValueError(
                f"Archive JPEG count {seen} does not match expected_count={expected_count}"
            )
        if len(category_full) != expected_categories:
            raise ValueError(
                f"Archive category count {len(category_full)} does not match "
                f"expected_categories={expected_categories}"
            )
        if selected_count != take:
            raise ValueError(f"Selected JPEG count {selected_count} does not match take={take}")
        train_count = source_metadata["train"] + selected_count
        validation_count = source_metadata["validation"]
        metadata = {
            "total": train_count + validation_count,
            "train": train_count,
            "validation": validation_count,
            "index_sha256": train_hash.hexdigest(),
            "index_sha256_scope": "exact bytes of mixed train.jsonl",
            "validation_sha256": _sha256(validation_bytes),
            "seed": seed,
            "split": "Unsplash validation unchanged; sampled Places training images added to train",
            "sources": {
                "unsplash": {
                    "data_directory": str(unsplash_data),
                    "total": source_metadata["total"],
                    "train": source_metadata["train"],
                    "validation": validation_count,
                    "source_index_sha256": source_metadata.get("index_sha256"),
                    **source_hashes,
                },
                "places365_standard": {
                    "archive": str(archive),
                    "archive_bytes": archive_size,
                    "available_train": seen,
                    "selected_train": selected_count,
                    "validation": 0,
                    "categories": len(category_full),
                    "category_counts": {
                        category: {"full": count, "selected": category_selected[category]}
                        for category, count in sorted(category_full.items())
                    },
                    "validated_images": selected_count if validate_images else 0,
                    "image_validation": "full JPEG decode" if validate_images else "not requested",
                },
            },
            "selection": {
                "algorithm": "random.Random(seed).sample(range(expected_count), take)",
                "seed": seed,
                "expected_count": expected_count,
                "take": take,
                "ordinal_basis": "zero-based regular JPEG member order in the uncompressed TAR",
                "manifest": "places-selection.jsonl",
                "sha256": manifest_hash.hexdigest(),
            },
        }
        if verified_md5 is not None:
            metadata["sources"]["places365_standard"]["external_archive_verification"] = {
                "md5": verified_md5.lower(),
                "provenance": "supplied by caller; this script did not compute or verify the MD5",
            }
        (staging / "dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        # Empty directories are permitted, but never replace an existing result.
        # rmdir also rejects a directory that became nonempty while we prepared.
        if output.exists():
            output.rmdir()
        os.rename(staging, output)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", required=True, help="Official uncompressed Places training TAR"
    )
    parser.add_argument("--unsplash-data", required=True, help="Existing Unsplash runtime/data")
    parser.add_argument("--output", required=True, help="New runtime/data directory")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--take", type=int, default=DEFAULT_TAKE)
    parser.add_argument("--expected-categories", type=int, default=365)
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Fully decode every selected JPEG; any failure aborts publication",
    )
    parser.add_argument(
        "--verified-md5", help="Externally verified MD5, recorded without recomputation"
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=8,
        help="Bounded 8 MiB archive prefetch blocks; 0 uses synchronous buffering",
    )
    args = parser.parse_args()
    metadata = prepare_mix(**vars(args))
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
