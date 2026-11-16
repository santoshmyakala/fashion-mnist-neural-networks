"""Three architectures of increasing capacity, to show what each modelling step buys."""

from __future__ import annotations

from torch import nn


def softmax_regression() -> nn.Module:
    """Linear baseline: one weight per pixel per class."""
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


def mlp(hidden: int = 512, dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, 10),
    )


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


def cnn(dropout: float = 0.3) -> nn.Module:
    """Two VGG-style conv blocks (28->14->7) and a small classifier head."""
    return nn.Sequential(
        _block(1, 32), _block(32, 64),
        nn.Flatten(), nn.Dropout(dropout),
        nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 10),
    )


MODELS = {"softmax": softmax_regression, "mlp": mlp, "cnn": cnn}
