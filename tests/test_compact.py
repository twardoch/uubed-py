# this_file: tests/test_compact.py
"""Portable signed-vector contracts, independent of an inference runtime."""

import base64
import json
import zlib

import numpy as np
import pytest

from uubed.compact import PackedEmbedding, pack, similarity


@pytest.mark.parametrize(
    ("precision", "threshold"), [("float32", 0.999999), ("int8", 0.9999), ("int4", 0.98), ("binary", 0.7)]
)
def test_pack_when_signed_vector_then_roundtrip_preserves_direction(precision, threshold):
    vector = np.random.default_rng(301).normal(size=768).astype(np.float32)
    encoded = pack(vector, space="model@revision:similarity", dimensions=256, precision=precision)
    restored = PackedEmbedding.loads(encoded.dumps())
    expected = vector[:256] / np.linalg.norm(vector[:256])
    assert restored.dimensions == 256
    assert np.dot(expected, restored.vector()) > threshold
    assert encoded.dumps().isascii()
    assert similarity(encoded, encoded) == pytest.approx(1.0, abs=1e-6)
    assert similarity(encoded, pack(-vector, space=encoded.space, dimensions=256, precision=precision)) < -0.999


@pytest.mark.parametrize(("precision", "nbytes"), [("float32", 1024), ("int8", 256), ("int4", 128), ("binary", 32)])
def test_pack_when_256_dimensions_then_exact_payload_budget(precision, nbytes):
    encoded = pack(np.arange(1, 769), space="m", dimensions=256, precision=precision)
    assert len(encoded.payload) == nbytes
    if precision != "float32":
        assert len(encoded.dumps().encode()) < 768 * 4


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [float("inf"), 1], [[1, 2]], [1j, 2j]])
def test_pack_when_invalid_vector_then_rejected(vector):
    with pytest.raises((ValueError, TypeError)):
        pack(vector, space="m")


@pytest.mark.parametrize("dimensions", [0, -1, 4, True, 1.5])
def test_pack_when_invalid_dimensions_then_rejected(dimensions):
    with pytest.raises(ValueError, match=r".+"):
        pack([1, 2, 3], space="m", dimensions=dimensions)


def test_pack_when_prefix_zero_then_rejected():
    with pytest.raises(ValueError, match=r".+"):
        pack([0, 0, 1], space="m", dimensions=2)


@pytest.mark.parametrize("precision", ["int4", "binary"])
def test_pack_when_odd_dimensions_then_padding_not_dimensions(precision):
    encoded = pack([1, -2, 3], space="m", precision=precision)
    assert encoded.vector().shape == (3,)
    assert np.sign(encoded.vector()).tolist() == [1, -1, 1]


def test_similarity_when_space_or_dimensions_differ_then_rejected():
    a = pack([1, 2, 3], space="m")
    with pytest.raises(ValueError, match=r".+"):
        similarity(a, pack([1, 2, 3], space="other"))
    with pytest.raises(ValueError, match=r".+"):
        similarity(a, pack([1, 2], space="m"))


def test_loads_when_payload_damaged_then_rejected():
    text = pack([1, 2, 3], space="m").dumps()
    with pytest.raises(ValueError, match=r".+"):
        PackedEmbedding.loads(text[:-1] + ("0" if text[-1] != "0" else "1"))
    for invalid in ["", "UB2.a.b.c", text + ".extra", text.replace("UB1", "ub1")]:
        with pytest.raises(ValueError, match=r".+"):
            PackedEmbedding.loads(invalid)


def test_loads_when_metadata_tampered_but_checksum_valid_then_rejected():
    parts = pack([1, 2, 3], space="m").dumps().split(".")
    metadata = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    metadata["d"] = 200
    parts[1] = base64.urlsafe_b64encode(json.dumps(metadata).encode()).decode().rstrip("=")
    body = ".".join(parts[:3])
    parts[3] = f"{zlib.crc32(body.encode()):08x}"
    with pytest.raises(ValueError, match=r".+"):
        PackedEmbedding.loads(".".join(parts))


def test_pack_when_float32_reference_then_normalized_components_exact():
    vector = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    assert np.array_equal(pack(vector, space="m", precision="float32").vector(), vector)


def test_pack_when_large_finite_input_then_normalization_does_not_overflow():
    vector = [1e30, -1e30, 1e30]
    result = pack(vector, space="m").vector()
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) == pytest.approx(1.0)
