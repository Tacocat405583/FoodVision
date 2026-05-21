# FoodVision

**Motivation** — Automated food recognition unlocks calorie tracking, menu digitization, and dietary logging at scale.

**Approach** — EfficientNet-B0 transfer learning: ImageNet-pretrained backbone frozen, custom `Dropout → Linear` classifier head trained on a 3-class pizza/steak/sushi dataset.

**Results** — 85.6% test accuracy in 5 epochs, ~19 seconds training time on GPU.

**Stack** — PyTorch, torchvision, PIL

**Status** — complete

## Usage

```bash
python train.py --data-dir data --epochs 5 --batch-size 32
```

All flags and their defaults:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Root directory with `train/` and `test/` subfolders |
| `--results-dir` | `results` | Output directory for model weights, curves, and history |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `32` | Samples per batch |
| `--lr` | `1e-3` | Adam learning rate |
| `--image-size` | `224` | Input image size (square) |
| `--num-workers` | `0` | DataLoader worker processes |

## Project Layout

```
foodvision/
├── src/
│   ├── dataset.py        # Transforms and DataLoader factory
│   ├── model.py          # EfficientNet-B0 with custom head
│   ├── trainer.py        # fit(), train_step(), eval_step()
│   └── predict.py        # Single-image inference
├── data/                 # Not committed — add images here
├── results/
│   └── loss_curves.png
├── train.py              # Entry point
├── requirements.txt
└── README.md
```

## Data Format

Place images in an `ImageFolder`-compatible structure:

```
data/
├── train/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
└── test/
    ├── pizza/
    ├── steak/
    └── sushi/
```
