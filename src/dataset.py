"""DataLoader setup, transforms, and dataset utilities for FoodVision."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool = True, image_size: int = 224) -> transforms.Compose:
    """Return image transforms for training or evaluation.

    Args:
        train: If True, apply augmentation; otherwise use only resize + normalize.
        image_size: Target square image size in pixels.

    Returns:
        A composed transform pipeline.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create train and test DataLoaders from an ImageFolder-structured directory.

    Expected layout::

        data_dir/
            train/
                pizza/
                steak/
                sushi/
            test/
                pizza/
                steak/
                sushi/

    Args:
        data_dir: Root directory containing ``train/`` and ``test/`` subdirectories.
        batch_size: Samples per batch.
        num_workers: Subprocess workers for data loading.
        image_size: Target square image size in pixels.

    Returns:
        Tuple of ``(train_loader, test_loader, class_names)``.
    """
    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=get_transforms(train=True, image_size=image_size),
    )
    test_dataset = datasets.ImageFolder(
        root=data_dir / "test",
        transform=get_transforms(train=False, image_size=image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset.classes
