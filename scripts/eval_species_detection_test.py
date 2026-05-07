"""Evaluate species detection models on test split and save CSV results.

Usage examples:
  python scripts/eval_species_detection_test.py --run run1
  python scripts/eval_species_detection_test.py --run all
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO, settings

import matplotlib.pyplot as plt
import seaborn as sns


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
    parser = argparse.ArgumentParser(description="Evaluate species detection models on test split")
    parser.add_argument("--run", choices=["run1", "run2", "run3", "run4", "all"], default="run4")
    parser.add_argument("--weights", default="best.pt", help="Weight file name under train/weights")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--exclude-class", type=int, default=9, help="Class id to exclude (unID=9)")
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--rare-thr", type=int, default=20, help="Instance threshold for Rare class")
    parser.add_argument("--common-thr", type=int, default=100, help="Instance threshold for Common class")
    parser.add_argument("--plot-only", action="store_true", help="Skip evaluation and only run averaging/plotting")
    parser.add_argument(
        "--label-summary",
        type=str,
        default="metadata/class_summary.csv",
        help="Path to global class summary CSV for categorization",
    )
    return parser.parse_args()


def save_eval_csv(metrics, classes: list[int], out_csv: Path, run_name: str, split: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    num_images = int(sum(1 for _ in (metrics.save_dir / "labels").glob("*.txt"))) if (metrics.save_dir / "labels").exists() else ""
    total_instances = int(metrics.nt_per_class.sum()) if hasattr(metrics, "nt_per_class") else ""

    for i, class_id in enumerate(classes):
        if i >= len(metrics.box.p):
            break
        rows.append(
            {
                "run": run_name,
                "split": split,
                "class_id": class_id,
                "class_name": metrics.names.get(class_id, str(class_id)),
                "num_image": float(metrics.nt_per_image[class_id]) if hasattr(metrics, "nt_per_image") else "",
                "num_instance": int(metrics.nt_per_class[class_id]) if hasattr(metrics, "nt_per_class") else "",
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "f1": float(metrics.box.f1[i]),
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
            }
        )

    rows.append(
        {
            "run": run_name,
            "split": split,
            "class_id": "MACRO-AVERAGING",
            "class_name": "MACRO-AVERAGING",
            "num_image": num_images,
            "num_instance": total_instances,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1": float(np.mean(metrics.box.f1)),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
    )

    fieldnames = [
        "run",
        "split",
        "class_id",
        "class_name",
        "num_image",
        "num_instance",
        "precision",
        "recall",
        "f1",
        "mAP50",
        "mAP50-95",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_map50_among_rarity(
    eval_csv: Path,
    out_pdf: Path,
    rare_thr: int = 20,
    common_thr: int = 100,
    label_summary: Path | None = None,
) -> None:
    df = pd.read_csv(eval_csv)
    
    # Map possible column names to standard names
    col_map = {
        "mAP50_mean": "mAP50",
        "num_instance_mean": "num_instance",
        "class_name": "class_name"
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "class_name" not in df.columns or "mAP50" not in df.columns:
        print(f"Skipping rarity plot (missing columns): {eval_csv}")
        return

    # Filter out averaging rows
    df = df[~df["class_name"].astype(str).str.contains("AVERAGING")].copy()

    # Load label summary for global instance counts if provided
    if label_summary and label_summary.exists():
        dflabel = pd.read_csv(label_summary)
        # Assuming 'name' in dflabel corresponds to 'class_name' in eval_csv
        df = df.merge(dflabel[["name", "n_instances"]], left_on="class_name", right_on="name", how="left")
        # Use n_instances from labels for categorization, fallback to local num_instance
        df["categorization_instances"] = df["n_instances"].fillna(df["num_instance"])
    else:
        df["categorization_instances"] = df["num_instance"]

    df["mAP50"] = pd.to_numeric(df["mAP50"], errors="coerce")
    df = df[df["categorization_instances"].notna() & (df["categorization_instances"] > 0) & df["mAP50"].notna()].copy()

    if df.empty:
        print(f"Skipping rarity plot (no valid data): {eval_csv}")
        return

    df["abundance_class"] = np.where(
        df["categorization_instances"] < rare_thr,
        "Rare",
        np.where(df["categorization_instances"] < common_thr, "Normal", "Common"),
    )

    order = ["Common", "Normal", "Rare"]
    
    # Setup plotting style
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    # Boxplot
    sns.boxplot(
        data=df,
        x="abundance_class",
        y="mAP50",
        order=order,
        showfliers=False,
        width=0.5,
        ax=ax,
        palette="pastel",
        boxprops=dict(facecolor="none", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # Stripplot
    sns.stripplot(
        data=df,
        x="abundance_class",
        y="mAP50",
        order=order,
        jitter=0.15,
        size=4,
        marker="o",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.5,
        ax=ax,
        facecolors="none",
    )

    # Add means as triangles
    for i, cat in enumerate(order):
        cat_data = df[df["abundance_class"] == cat]["mAP50"]
        if not cat_data.empty:
            ax.scatter(i, cat_data.mean(), marker="^", s=48, color="black", zorder=10)

    # Update labels with counts
    counts = df["abundance_class"].value_counts()
    new_labels = [f"{c}\n(n={counts.get(c, 0)})" for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(new_labels)

    ax.set_xlabel("Species abundance category")
    ax.set_ylabel("Detection performance (mAP50)")
    ax.set_ylim(-0.05, 1.05)
    sns.despine()
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved styled rarity plot: {out_pdf}")


def average_results(csv_paths: list[Path], out_csv: Path) -> None:
    """Average per-class results across multiple runs and save to CSV."""
    all_dfs = []
    for path in csv_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Filter out rows that are not per-class (e.g., MACRO-AVERAGING)
        df = df[~df["class_id"].astype(str).str.contains("AVERAGING")].copy()
        # Ensure numeric columns are actually numeric
        for col in ["num_image", "num_instance", "precision", "recall", "f1", "mAP50", "mAP50-95"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        all_dfs.append(df)

    if not all_dfs:
        print("No CSV files found to average.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    metric_cols = ["num_image", "num_instance", "precision", "recall", "f1", "mAP50", "mAP50-95"]
    
    # Calculate mean and standard deviation
    averaged = (
        merged_df.groupby(["class_id", "class_name"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten column names: "metric_mean", "metric_std"
    averaged.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in averaged.columns
    ]

    # For consistency with the plotting function, create a version with clean names
    averaged_clean = averaged.copy()
    for col in metric_cols:
        if f"{col}_mean" in averaged_clean.columns:
            averaged_clean[col] = averaged_clean[f"{col}_mean"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Save the clean version (with standard column names)
    averaged_clean.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved averaged results CSV with standard names: {out_csv}")


def evaluate_one(run_name: str, args: argparse.Namespace, classes: list[int]) -> dict:
    dataset_dir = DATASETS_ROOT / RUN_CONFIGS[run_name]
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {data_yaml}")

    run_root = RUNS_ROOT / run_name
    weights = run_root / "train" / "weights" / args.weights
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights for {run_name}: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        classes=classes,
        verbose=True,
        save_json=args.save_json,
        project=str(run_root),
        name="val",
    )

    run_suffix = run_name.replace("run", "")
    out_csv = run_root / "val" / f"eval_results{run_suffix}.csv"
    save_eval_csv(metrics, classes, out_csv, run_name, args.split)
    print(f"Saved eval CSV: {out_csv}")
    plot_map50_among_rarity(
        eval_csv=out_csv,
        out_pdf=run_root / "val" / "map50_among_cls.pdf",
        rare_thr=args.rare_thr,
        common_thr=args.common_thr,
        label_summary=Path(args.label_summary) if args.label_summary else None,
    )

    return {
        "run": run_name,
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(np.mean(metrics.box.f1)),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def save_summary(rows: list[dict], split: str) -> None:
    out_csv = RUNS_ROOT / f"eval_summary_{split}.csv"
    fieldnames = ["run", "split", "precision", "recall", "f1", "mAP50", "mAP50-95"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV: {out_csv}")


def main() -> None:
    args = parse_args()
    settings.update({"runs_dir": str(RUNS_ROOT)})
    settings.update({"datasets_dir": str(DATASETS_ROOT)})

    classes = [i for i in range(109) if i != args.exclude_class]
    run_list = list(RUN_CONFIGS.keys()) if args.run == "all" else [args.run]

    summary_rows = []
    run_csv_paths = []
    for run_name in run_list:
        run_suffix = run_name.replace("run", "")
        csv_path = RUNS_ROOT / run_name / "val" / f"eval_results{run_suffix}.csv"
        run_csv_paths.append(csv_path)

        if not args.plot_only:
            res = evaluate_one(run_name, args, classes)
            summary_rows.append(res)
        else:
            print(f"Skipping evaluation for {run_name} (plot-only mode)")

    if not args.plot_only:
        save_summary(summary_rows, args.split)

    if args.run == "all" and len(run_csv_paths) > 1:
        averaged_csv = RUNS_ROOT / "eval_results_averaged.csv"
        average_results(run_csv_paths, averaged_csv)
        plot_map50_among_rarity(
            eval_csv=averaged_csv,
            out_pdf=RUNS_ROOT / "map50_among_cls_averaged.pdf",
            rare_thr=args.rare_thr,
            common_thr=args.common_thr,
            label_summary=Path(args.label_summary) if args.label_summary else None,
        )


if __name__ == "__main__":
    main()
