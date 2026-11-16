import struct

import numpy as np
import pytest

from fmnist.data import parse_idx


def _idx(array: np.ndarray) -> bytes:
    header = struct.pack(">HBB", 0, 0x08, array.ndim) + struct.pack(f">{array.ndim}I", *array.shape)
    return header + array.tobytes()


def test_parse_idx_roundtrip():
    images = np.random.default_rng(0).integers(0, 255, (5, 28, 28), dtype=np.uint8)
    np.testing.assert_array_equal(parse_idx(_idx(images)), images)


def test_parse_idx_rejects_truncated_payload():
    raw = _idx(np.zeros((2, 28, 28), dtype=np.uint8))
    with pytest.raises(ValueError):
        parse_idx(raw[:-10])


def test_parse_idx_rejects_non_ubyte():
    with pytest.raises(ValueError):
        parse_idx(struct.pack(">HBB", 0, 0x0D, 1) + struct.pack(">I", 1) + b"\0\0\0\0")
