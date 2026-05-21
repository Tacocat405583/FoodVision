"""Single-image inference pipeline for FoodVision."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def load_and_transform(
    image_path: str | Path,
    image_size: int = 224,
) -> torch.Tensor:
    """Load a single image from disk and apply inference-time transforms.

    Args:
        image_path: Path to the image file (any PIL-supported format).
        image_size: Target square size for resizing.

    Returns:
        Float tensor of shape ``(1, 3, image_size, image_size)``.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(
    model: nn.Module,
    image_path: str | Path,
    class_names: list[str],
    device: torch.device,
    image_size: int = 224,
) -> tuple[str, float]:
    """Run inference on a single image and return the top-1 prediction.

    Args:
        model: Trained classification model.
        image_path: Path to the input image.
        class_names: Ordered list of class label strings (same order used in training).
        device: Compute device to run the forward pass on.
        image_size: Target square size used during transform.

    Returns:
        ``(predicted_class_name, confidence)`` where confidence is a float in ``[0, 1]``.
    """
    tensor = load_and_transform(image_path, image_size).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.max(dim=1)

    return class_names[top_idx.item()], top_prob.item()


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.model import build_model  # noqa: E402

    parser = argparse.ArgumentParser(description="FoodVision single-image inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--weights", default="results/model.pth")
    parser.add_argument("--classes", nargs="+", default=["pizza", "steak", "sushi"])
    args = parser.parse_args()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = build_model(num_classes=len(args.classes))
    _model.load_state_dict(torch.load(args.weights, map_location=_device))

    _label, _conf = predict(_model, args.image, args.classes, _device)
    print(f"{_label}  ({_conf:.2%})")
