"""Training and evaluation loops for FoodVision."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch of forward pass, loss computation, and weight updates.

    Args:
        model: Network to train.
        loader: DataLoader yielding ``(images, labels)`` batches.
        loss_fn: Loss criterion.
        optimizer: Gradient-based optimizer.
        device: Compute device.

    Returns:
        ``(mean_loss, mean_accuracy)`` over the full epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


def eval_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a dataset without updating parameters.

    Args:
        model: Network to evaluate.
        loader: DataLoader yielding ``(images, labels)`` batches.
        loss_fn: Loss criterion.
        device: Compute device.

    Returns:
        ``(mean_loss, mean_accuracy)`` over the full dataset.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, list[float]]:
    """Train and evaluate the model for a fixed number of epochs.

    Only parameters with ``requires_grad=True`` are passed to the optimizer,
    so a frozen backbone does not receive gradient updates.

    Args:
        model: Network to train (moved to ``device`` internally).
        train_loader: DataLoader for training data.
        test_loader: DataLoader for evaluation data.
        epochs: Total training epochs.
        lr: Adam learning rate.
        device: Compute device.

    Returns:
        History dict with keys ``train_loss``, ``train_acc``, ``test_loss``,
        ``test_acc``; each value is a list of per-epoch floats.
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss,  test_acc  = eval_step(model, test_loader,  loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"test  loss: {test_loss:.4f}  acc: {test_acc:.4f}"
        )

    return history
