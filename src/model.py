"""EfficientNet-B0 with a frozen backbone and a custom classifier head."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Construct an EfficientNet-B0 model with a task-specific classifier head.

    The feature extractor is loaded with ImageNet-pretrained weights. Only the
    final classifier is trainable by default; all feature-extractor parameters
    are frozen.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Freeze all ``model.features`` parameters when True.

    Returns:
        EfficientNet-B0 with a ``(Dropout → Linear)`` classifier.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features: int = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model
