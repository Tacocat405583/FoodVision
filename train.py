"""Entry point: wire dataset, model, and trainer together to run FoodVision training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import create_dataloaders
from src.model import build_model
from src.trainer import fit


def plot_curves(history: dict[str, list[float]], save_path: Path) -> None:
    """Save a side-by-side loss / accuracy figure to disk.

    Args:
        history: Training history returned by :func:`~src.trainer.fit`.
        save_path: Destination ``.png`` path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"],  label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["test_acc"],  label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Curves saved → {save_path}")


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments.

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(description="Train FoodVision EfficientNet-B0")
    p.add_argument("--data-dir",    default="data",    help="Root data directory")
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--image-size",  type=int,   default=224)
    p.add_argument("--num-workers", type=int,   default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names), freeze_backbone=True)

    history = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_curves(history, results_dir / "loss_curves.png")
    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), results_dir / "model.pth")
    print(f"Artefacts saved to {results_dir}/")


if __name__ == "__main__":
    main()
