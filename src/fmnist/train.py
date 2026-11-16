"""Train a model, evaluate it on the official test set and write metrics, plots and weights.

    python -m fmnist.train --model cnn --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fmnist import CLASSES
from fmnist.data import load
from fmnist.models import MODELS

MEAN, STD = 0.2860, 0.3530  # Fashion-MNIST training-set pixel statistics


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return ((torch.tensor(images, dtype=torch.float32) / 255.0 - MEAN) / STD).unsqueeze(1)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random horizontal flip and up to 2-pixel translation (padding with background)."""
    flip = torch.rand(len(x), generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    padded = nn.functional.pad(x, (2, 2, 2, 2), value=float(-MEAN / STD))
    dx, dy = torch.randint(0, 5, (2,), generator=gen).tolist()
    return padded[..., dy : dy + 28, dx : dx + 28]


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return torch.cat([model(b).argmax(1) for b in x.split(2048)])


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    images, labels = load(args.data_dir, "train")
    test_images, test_labels = load(args.data_dir, "test")
    if args.limit:
        images, labels = images[: args.limit], labels[: args.limit]
    perm = np.random.default_rng(args.seed).permutation(len(images))
    n_val = max(1, int(len(images) * 0.1))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr, y_tr = to_tensor(images[tr_idx]), torch.tensor(labels[tr_idx], dtype=torch.long)
    x_val, y_val = to_tensor(images[val_idx]), torch.tensor(labels[val_idx], dtype=torch.long)
    x_te, y_te = to_tensor(test_images), torch.tensor(test_labels, dtype=torch.long)

    model = MODELS[args.model]()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    steps = args.epochs * int(np.ceil(len(y_tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best_acc, best_state = [], -1.0, None
    for epoch in range(args.epochs):
        model.train()
        start, running = time.perf_counter(), 0.0
        for batch in torch.randperm(len(y_tr), generator=gen).split(args.batch_size):
            xb = x_tr[batch]
            if args.augment:
                xb = augment(xb, gen)
            loss = loss_fn(model(xb), y_tr[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(batch)
        val_acc = float((predict(model, x_val) == y_val).float().mean())
        history.append({"epoch": epoch + 1, "train_loss": running / len(y_tr), "val_acc": val_acc,
                        "seconds": round(time.perf_counter() - start, 1)})
        print(json.dumps(history[-1]))
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pred = predict(model, x_te)
    confusion = torch.zeros(10, 10, dtype=torch.long)
    confusion.index_put_((y_te, pred), torch.ones_like(y_te), accumulate=True)
    per_class = (confusion.diag() / confusion.sum(1)).tolist()
    result = {
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs,
        "augment": args.augment,
        "train_images": len(y_tr),
        "test_accuracy": float((pred == y_te).float().mean()),
        "best_val_accuracy": best_acc,
        "per_class_accuracy": dict(zip(CLASSES, per_class, strict=True)),
        "train_seconds": round(sum(h["seconds"] for h in history), 1),
        "history": history,
        "confusion_matrix": confusion.tolist(),
    }

    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), out / "model.pt")
    _plot_confusion(confusion.numpy(), out / "confusion_matrix.png", f"{args.model} — test accuracy "
                    f"{result['test_accuracy']:.2%}")
    if args.mlflow:
        _log_mlflow(args, result, out)
    return result


def _plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _log_mlflow(args: argparse.Namespace, result: dict, out: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fashion-mnist")
    with mlflow.start_run(run_name=args.model):
        mlflow.log_params({k: v for k, v in vars(args).items() if k not in {"data_dir", "out"}})
        for h in result["history"]:
            mlflow.log_metrics({"train_loss": h["train_loss"], "val_acc": h["val_acc"]}, step=h["epoch"])
        mlflow.log_metrics({"test_accuracy": result["test_accuracy"], "parameters": result["parameters"]})
        mlflow.log_artifacts(str(out))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="cnn")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None,
                    help="flip/shift augmentation (default: on for cnn, off otherwise)")
    ap.add_argument("--limit", type=int, help="use only the first N training images (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--mlflow", action="store_true", help="log params, curves and artifacts to MLflow")
    args = ap.parse_args(argv)
    if args.augment is None:
        args.augment = args.model == "cnn"
    return args


if __name__ == "__main__":
    train(parse_args())
