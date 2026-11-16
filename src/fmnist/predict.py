"""Classify image files with a trained model.

    python -m fmnist.predict --model cnn path/to/image.png [...]

Images are converted to 28x28 greyscale. Fashion-MNIST items are light on a dark
background, so pass --invert for photos of dark items on a light background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from fmnist import CLASSES
from fmnist.models import MODELS
from fmnist.train import to_tensor


def load_model(name: str, weights: Path) -> torch.nn.Module:
    model = MODELS[name]()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


def classify(model: torch.nn.Module, images: np.ndarray) -> list[tuple[str, float]]:
    with torch.no_grad():
        probs = model(to_tensor(images)).softmax(1)
    conf, idx = probs.max(1)
    return [(CLASSES[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist(), strict=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", choices=sorted(MODELS), default="cnn")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()
    model = load_model(args.model, args.weights or Path("reports") / args.model / "model.pt")
    arrays = []
    for path in args.images:
        img = ImageOps.grayscale(Image.open(path)).resize((28, 28))
        arrays.append(np.asarray(ImageOps.invert(img) if args.invert else img))
    for path, (label, conf) in zip(args.images, classify(model, np.stack(arrays)), strict=True):
        print(f"{path}: {label} ({conf:.1%})")


if __name__ == "__main__":
    main()
