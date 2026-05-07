"""Train fish segmentation model with relative paths.

Usage example:
  python scripts/train_fish_segmentation.py --model yolo11x-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "training_datasets" / "fish_segmentation" / "data.yaml"
RUNS_ROOT = PROJECT_ROOT / "runs" / "training" / "fish_segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fish segmentation model")
    parser.add_argument("--model", default="yolo26x-seg.pt", help="Base model (notebook uses yolo26x-seg.pt)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0", help="Device to use (e.g. 0 or cpu)")
    parser.add_argument("--single-cls", action="store_true", default=True, help="Force single class training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {DATA_YAML}")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    settings.update({"runs_dir": str(RUNS_ROOT)})
    settings.update({"datasets_dir": str(PROJECT_ROOT / "training_datasets")})

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        single_cls=args.single_cls,
        verbose=True,
        project=str(RUNS_ROOT),
        name="train",
        device=args.device,
    )


if __name__ == "__main__":
    main()

