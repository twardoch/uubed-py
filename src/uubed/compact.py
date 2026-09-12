# this_file: src/uubed/compact.py
"""UB1: self-contained signed vectors for text files, independent of legacy MQ64.

Truncate, normalize, then quantize. Scalar quantization uses a per-vector
absolute maximum so decoding needs no external calibration corpus. Binary
vectors reconstruct signs; they approximate angular similarity, not magnitude.
"""

from __future__ import annotations

import base64
import binascii
import json
import zlib
from dataclasses import dataclass

import numpy as np

PRECISIONS = {"float32": 32, "int8": 8, "int4": 4, "binary": 1}
MAX_DIMENSIONS = 65536
MAX_SPACE_BYTES = 4096
MAX_RECORD_CHARS = 400000


def _normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values.astype(np.float64))
    if not np.isfinite(norm) or norm == 0:
        msg = "embedding must have a finite nonzero norm"
        raise ValueError(msg)
    return (values / norm).astype(np.float32)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    result = base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_", validate=True)
    if _b64(result) != text:
        msg = "noncanonical base64"
        raise ValueError(msg)
    return result


@dataclass(frozen=True)
class PackedEmbedding:
    """Immutable quantized payload with its vector-space identity and scale."""

    space: str
    dimensions: int
    precision: str
    scale: float
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or not self.space or len(self.space.encode()) > MAX_SPACE_BYTES:
            msg = "space must be a nonempty model/revision/task identity of at most 4096 bytes"
            raise ValueError(msg)
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= MAX_DIMENSIONS:
            msg = "invalid dimensions"
            raise ValueError(msg)
        if self.precision not in PRECISIONS:
            msg = "unsupported precision"
            raise ValueError(msg)
        if not isinstance(self.scale, (int, float)) or not np.isfinite(self.scale) or self.scale <= 0:
            msg = "scale must be finite and positive"
            raise ValueError(msg)
        bits = PRECISIONS[self.precision]
        if not isinstance(self.payload, bytes) or len(self.payload) != (self.dimensions * bits + 7) // 8:
            msg = "payload size does not match dimensions and precision"
            raise ValueError(msg)
        if bits == 1 and self.dimensions % 8 and self.payload[-1] & ((1 << (8 - self.dimensions % 8)) - 1):
            msg = "nonzero binary padding"
            raise ValueError(msg)
        if bits == PRECISIONS["int4"] and self.dimensions % 2 and self.payload[-1] & 15:
            msg = "nonzero int4 padding"
            raise ValueError(msg)

    def vector(self) -> np.ndarray:
        """Decode to a normalized float32 direction, retaining negative components."""
        raw = np.frombuffer(self.payload, dtype=np.uint8)
        if self.precision == "float32":
            values = np.frombuffer(self.payload, dtype="<f4")
        elif self.precision == "int8":
            values = raw.view(np.int8).astype(np.float32) * self.scale
        elif self.precision == "int4":
            nibbles = np.stack((raw >> 4, raw & 15), axis=1).ravel()[: self.dimensions]
            values = (nibbles.astype(np.float32) - 7) * self.scale
        else:
            values = np.unpackbits(raw)[: self.dimensions].astype(np.float32) * 2 - 1
        return _normalize(values)

    def dumps(self) -> str:
        """ASCII record with a CRC32 for accidental corruption detection."""
        meta = {"s": self.space, "d": self.dimensions, "q": self.precision, "a": self.scale}
        body = f"UB1.{_b64(json.dumps(meta, separators=(',', ':'), ensure_ascii=True).encode())}.{_b64(self.payload)}"
        return f"{body}.{zlib.crc32(body.encode()):08x}"

    @classmethod
    def loads(cls, text: str) -> PackedEmbedding:
        """Reject malformed records, inconsistent metadata and damaged payloads."""
        try:
            if not isinstance(text, str) or len(text) > MAX_RECORD_CHARS:
                msg = "record too large or not text"
                raise ValueError(msg)
            version, header, data, checksum = text.split(".")
            body = f"{version}.{header}.{data}"
            if version != "UB1" or checksum != f"{zlib.crc32(body.encode()):08x}":
                msg = "unsupported version or checksum mismatch"
                raise ValueError(msg)
            meta = json.loads(_unb64(header))
            if not isinstance(meta, dict) or set(meta) != {"s", "d", "q", "a"}:
                msg = "invalid metadata"
                raise ValueError(msg)
            result = cls(meta["s"], meta["d"], meta["q"], meta["a"], _unb64(data))
            result.vector()
            return result
        except (TypeError, KeyError, binascii.Error, UnicodeError) as error:
            msg = "invalid UB1 record"
            raise ValueError(msg) from error


def pack(
    values: np.ndarray | list[float], *, space: str, dimensions: int | None = None, precision: str = "int8"
) -> PackedEmbedding:
    """Encode a vector prefix; callers enforce the model's supported MRL dimensions."""
    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind not in "fiu" or not np.all(np.isfinite(source)):
        msg = "embedding must be a one-dimensional finite real vector"
        raise ValueError(msg)
    dimensions = len(source) if dimensions is None else dimensions
    if type(dimensions) is not int or not 1 <= dimensions <= min(len(source), MAX_DIMENSIONS):
        msg = "dimensions must be a positive integer no greater than the input length"
        raise ValueError(msg)
    vector = _normalize(source[:dimensions].astype(np.float64))
    scale = 1.0
    if precision == "float32":
        payload = vector.astype("<f4").tobytes()
    elif precision in {"int8", "int4"}:
        maximum = 127 if precision == "int8" else 7
        scale = float(np.max(np.abs(vector))) / maximum
        quantized = np.rint(vector / scale).clip(-maximum, maximum).astype(np.int8)
        if precision == "int8":
            payload = quantized.tobytes()
        else:
            unsigned = (quantized + 7).astype(np.uint8)
            unsigned = np.pad(unsigned, (0, dimensions % 2))
            payload = ((unsigned[::2] << 4) | unsigned[1::2]).tobytes()
    elif precision == "binary":
        payload = np.packbits(vector > 0).tobytes()
    else:
        msg = "unsupported precision"
        raise ValueError(msg)
    return PackedEmbedding(space, dimensions, precision, scale, payload)


def similarity(left: PackedEmbedding, right: PackedEmbedding) -> float:
    """Cosine of reconstructed vectors; refuse mixed vector spaces or dimensions."""
    if (left.space, left.dimensions) != (right.space, right.dimensions):
        msg = "embedding spaces and dimensions must match"
        raise ValueError(msg)
    return float(np.clip(np.dot(left.vector(), right.vector()), -1.0, 1.0))
