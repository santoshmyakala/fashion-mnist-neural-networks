"""Download (with checksum verification) and parse the Fashion-MNIST IDX files."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRRORS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
)
FILES = {  # filename -> md5 published with the dataset
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        target = root / name
        if target.exists() and _md5(target) == md5:
            continue
        for mirror in MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + name, target)  # noqa: S310 - fixed https/http mirrors
            except OSError:
                continue
            if _md5(target) == md5:
                break
        else:
            raise RuntimeError(f"could not download a valid copy of {name}")


def parse_idx(raw: bytes) -> np.ndarray:
    """Parse an (uncompressed) IDX file: magic number encodes dtype and number of dimensions."""
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError("only unsigned-byte IDX files are supported")
    shape = struct.unpack(f">{ndim}I", raw[4 : 4 + 4 * ndim])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError("IDX payload size does not match header")
    return data.reshape(shape)


def load(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return images (N, 28, 28) uint8 and labels (N,) for split 'train' or 'test'."""
    prefix = {"train": "train", "test": "t10k"}[split]
    download(root)
    images = parse_idx(gzip.decompress((root / f"{prefix}-images-idx3-ubyte.gz").read_bytes()))
    labels = parse_idx(gzip.decompress((root / f"{prefix}-labels-idx1-ubyte.gz").read_bytes()))
    return images, labels
