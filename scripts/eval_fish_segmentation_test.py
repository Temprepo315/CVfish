"""Evaluate fish segmentation model on test split and save results.

Usage examples:
  python scripts/eval_fish_segmentation_test.py --weights runs/training/fish_segmentation/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO, settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "training_datasets" / "fish_segmentation" / "data.yaml"
RUNS_ROOT = PROJECT_ROOT / "runs" / "training" / "fish_segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fish segmentation model on test split")
    parser.add_argument(
        "--weights", 
        default="runs/training/fish_segmentation/train/weights/best.pt",
        help="Path to weight file"
    )
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--device", default="0", help="Device to use (e.g. 0 or cpu)")
    parser.add_argument("--single-cls", action="store_true", default=True, help="Force single class evaluation")
    return parser.parse_args()


def save_eval_csv(metrics, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # YOLO gives metrics for both Box and Mask for segmentation tasks
    # We'll collect Mask metrics as they are primary for segmentation
    for i, name in enumerate(metrics.names.values()):
        if i >= len(metrics.seg.p):
            break
        rows.append(
            {
                "class_id": i,
                "class_name": name,
                "num_image": int(metrics.nt_per_image[i]) if hasattr(metrics, "nt_per_image") else "",
                "num_instance": int(metrics.nt_per_class[i]) if hasattr(metrics, "nt_per_class") else "",
                "precision": float(metrics.seg.p[i]),
                "recall": float(metrics.seg.r[i]),
                "f1": float(metrics.seg.f1[i]),
                "mAP50": float(metrics.seg.ap50[i]),
                "mAP50-95": float(metrics.seg.ap[i]),
                "type": "mask"
            }
        )
        # Also add box metrics for reference
        rows.append(
            {
                "class_id": i,
                "class_name": name,
                "num_image": int(metrics.nt_per_image[i]) if hasattr(metrics, "nt_per_image") else "",
                "num_instance": int(metrics.nt_per_class[i]) if hasattr(metrics, "nt_per_class") else "",
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "f1": float(metrics.box.f1[i]),
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
                "type": "box"
            }
        )

    fieldnames = [
        "class_id",
        "class_name",
        "num_image",
        "num_instance",
        "precision",
        "recall",
        "f1",
        "mAP50",
        "mAP50-95",
        "type"
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    
    if not weights_path.exists():
        # Try relative to runs root if not found
        weights_path = RUNS_ROOT / args.weights
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights: {args.weights}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {DATA_YAML}")

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    
    print(f"Starting evaluation on {args.split} split...")
    metrics = model.val(
        data=str(DATA_YAML),
        split=args.split,
        verbose=True,
        single_cls=args.single_cls,
        device=args.device,
    )

    out_csv = RUNS_ROOT / "val" / f"results_{args.split}.csv"
    save_eval_csv(metrics, out_csv)
    
    print("\n--- Evaluation Summary (Mask) ---")
    print(f"mAP50: {metrics.seg.map50:.4f}")
    print(f"mAP50-95: {metrics.seg.map:.4f}")
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
