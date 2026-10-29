"""Read image bytes from legacy TFRecords, image files, or uncompressed TAR offsets."""

import struct
from pathlib import Path

MAX_RECORD_BYTES = 64 * 1024**2


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"Invalid {name}: expected a 64-bit integer >= {minimum}, got {value!r}")
    return value


def _path(value, absolute=False):
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid image index path: expected a nonempty string")
    if absolute and not Path(value).is_absolute():
        raise ValueError(f"Invalid image index path: expected an absolute path, got {value!r}")
    return value


def _read_exact(stream, length, label):
    value = stream.read(length)
    if len(value) != length:
        raise ValueError(f"Truncated {label}: expected {length} bytes, got {len(value)}")
    return value


def read_record_image(record, decode_tfrecord):
    """Return ``(image_bytes, stable_id)`` without decoding new formats as pickle.

    Legacy rows are ``[path, record_offset, total_record_length]``. TAR offsets
    address member contents directly in an *uncompressed* archive, not a header.
    Only legacy TFRecord payloads are passed to ``decode_tfrecord``.
    """
    if isinstance(record, (list, tuple)):
        if len(record) != 3:
            raise ValueError("Invalid TFRecord index: expected [path, offset, total_length]")
        path = _path(record[0])
        offset = _integer(record[1], "TFRecord offset")
        expected_length = _integer(record[2], "TFRecord total_length", minimum=16)
        if expected_length > MAX_RECORD_BYTES + 16:
            raise ValueError(f"Invalid TFRecord index {path}:{offset}: record is too large")
        label = f"TFRecord {path}:{offset}"
        with open(path, "rb") as stream:
            stream.seek(offset)
            header = _read_exact(stream, 12, f"{label} header")
            length = struct.unpack("<Q", header[:8])[0]
            if length + 16 != expected_length or length > MAX_RECORD_BYTES:
                raise ValueError(f"Invalid TFRecord index {path}:{offset}: length mismatch")
            payload = _read_exact(stream, length, label)
        return decode_tfrecord(payload), f"{path}:{offset}"

    if not isinstance(record, dict):
        raise ValueError("Invalid image index: expected a TFRecord row or a format dictionary")
    image_format = record.get("format")
    if not isinstance(image_format, str) or image_format not in {"image", "tar"}:
        raise ValueError(f"Unsupported image index format: {image_format!r}")
    path = _path(record.get("path"), absolute=True)
    if image_format == "image":
        with open(path, "rb") as stream:
            image_bytes = stream.read(MAX_RECORD_BYTES + 1)
        if not image_bytes or len(image_bytes) > MAX_RECORD_BYTES:
            raise ValueError(f"Invalid image file {path}: empty or larger than 64 MiB")
        return image_bytes, path

    offset = _integer(record.get("offset"), "TAR member offset")
    length = _integer(record.get("length"), "TAR member length", minimum=1)
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid TAR member name: expected a nonempty string")
    if length > MAX_RECORD_BYTES:
        raise ValueError(f"Invalid TAR member {path}:{offset}: member is larger than 64 MiB")
    stable_id = f"{path}:{offset}:{name}"
    with open(path, "rb") as stream:
        stream.seek(offset)
        image_bytes = _read_exact(stream, length, f"TAR member {stable_id}")
    return image_bytes, stable_id
