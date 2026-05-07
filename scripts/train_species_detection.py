"""Train species detection YOLO models with relative paths.

Usage examples:
  python scripts/train_species_detection.py --run run1
  python scripts/train_species_detection.py --run run4 --epochs 300 --batch 8 --imgsz 960
  python scripts/train_species_detection.py --run all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "training_datasets"
RUNS_ROOT = PROJECT_ROOT / "runs" / "training" / "fish_id_bbox"

RUN_CONFIGS = {
    "run1": "species_detection_run1",
    "run2": "species_detection_run2",
    "run3": "species_detection_run3",
    "run4": "species_detection_run4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train species detection models")
    parser.add_argument("--run", choices=["run1", "run2", "run3", "run4", "all"], default="run4")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--exclude-class", type=int, default=9, help="Class id to exclude (unID=9)")
    return parser.parse_args()


def train_one(run_name: str, args: argparse.Namespace, classes: list[int]) -> None:
    dataset_dir = DATASETS_ROOT / RUN_CONFIGS[run_name]
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {data_yaml}")

    project_dir = RUNS_ROOT / run_name
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training {run_name} ===")
    print(f"Dataset: {data_yaml}")
    print(f"Output:  {project_dir}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        classes=classes,
        verbose=True,
        project=str(project_dir),
        name="train",
    )


def main() -> None:
    args = parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    settings.update({"runs_dir": str(RUNS_ROOT)})
    settings.update({"datasets_dir": str(DATASETS_ROOT)})

    classes = [i for i in range(109) if i != args.exclude_class]

    run_list = list(RUN_CONFIGS.keys()) if args.run == "all" else [args.run]
    for run_name in run_list:
        train_one(run_name, args, classes)


if __name__ == "__main__":
    main()
