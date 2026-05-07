#!/usr/bin/env python3
"""Consolidate tracking results with species detections using hierarchical voting.

This script reconciles tracking IDs with species identification detections.
It filters short-lived tracks, matches detections using IOU, applies
hierarchical logic (Family -> Genus -> Species), and can directly render
consolidated visualization images.

Usage:
  python scripts/integrate_species.py --tracking-dir runs/analysis/csv_tracking --species-dir runs/analysis/csv_species
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics.utils.metrics import bbox_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate tracking and species consensus")
    parser.add_argument("--tracking-dir", type=str, required=True, help="Dir containing tracking CSVs")
    parser.add_argument("--species-dir", type=str, required=True, help="Dir containing species detection CSVs")
    parser.add_argument("--output-dir", type=str, default="runs/analysis/csv_consolidated", help="Output directory")
    parser.add_argument("--metadata", type=str, default="metadata/class_summary_with_JP_labeled.csv", help="Species metadata")
    parser.add_argument("--count_thre", type=int, default=5, help="Min frames to keep a track (default 5)")
    parser.add_argument("--iou_thre", type=float, default=0.5, help="IOU threshold for matching (default 0.5)")
    parser.add_argument("--conf_thre", type=float, default=0.3, help="Min species detection confidence (default 0.3)")
    parser.add_argument("--label_thre", type=float, default=0.25, help="Min fraction of frames with species labels (default 0.25)")
    parser.add_argument("--proportion_thre", type=float, default=0.1, help="Proportion threshold for refinement (default 0.1)")
    parser.add_argument("--save-images", action="store_true", help="Enable visualization (generate annotated images)")
    parser.add_argument("--img-dir", type=str, help="Path to source images (required for visualization)")
    parser.add_argument("--save-img-dir", type=str, default="runs/analysis/images_consolidated", help="Output directory for annotated images")
    parser.add_argument("--japanese", action="store_true", help="Use Japanese names when visualizing")
    return parser.parse_args()


def hex_to_bgr(hex_color):
    if pd.isna(hex_color) or not isinstance(hex_color, str):
        return (128, 128, 128)
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def hex_to_rgb(hex_color):
    bgr = hex_to_bgr(hex_color)
    return (bgr[2], bgr[1], bgr[0])


def draw_text_with_outline(draw, pos, text, font, text_color, outline_color, outline_width=2):
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)


def get_box_coords(row):
    x1 = int(row["x1_tra"]) if not pd.isna(row.get("x1_tra")) else int(row["x1"])
    y1 = int(row["y1_tra"]) if not pd.isna(row.get("y1_tra")) else int(row["y1"])
    x2 = int(row["x2_tra"]) if not pd.isna(row.get("x2_tra")) else int(row["x2"])
    y2 = int(row["y2_tra"]) if not pd.isna(row.get("y2_tra")) else int(row["y2"])
    return x1, y1, x2, y2


def get_label_text(row, japanese=False):
    if japanese:
        label = row.get("JP_name", row.get("Label", ""))
        return f"{int(row['id'])} {label}"
    label = row.get("label", row.get("Label", ""))
    return f"ID:{int(row['id'])} {label}"


def visualize_consolidated_results(csv_dir, img_root, output_dir, metadata_path, japanese=False):
    csv_root = Path(csv_dir)
    img_root = Path(img_root)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata_df = None
    metadata_path = Path(metadata_path)
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path, encoding="utf-8-sig")

    font = None
    if japanese:
        font_path = "C:/Windows/Fonts/YuGothM.ttc"
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=50)
        else:
            print("Warning: YuGothM.ttc not found. Falling back to OpenCV text rendering.")
            japanese = False

    csv_files = [f for f in os.listdir(csv_root) if f.endswith(".csv")]
    for filename in csv_files:
        clip_name = os.path.splitext(filename)[0]
        img_folder = img_root / clip_name
        if not img_folder.exists():
            print(f"Image folder not found for {clip_name}. Skipping visualization.")
            continue

        print(f"Visualizing results for {clip_name}...")
        df = pd.read_csv(csv_root / filename)

        if metadata_df is not None and "Label" in df.columns and "name" in metadata_df.columns:
            merge_cols = [col for col in ["name", "color", "JP_name", "label"] if col in metadata_df.columns]
            df = df.merge(metadata_df[merge_cols], left_on="Label", right_on="name", how="left", suffixes=("", "_meta"))
            for col in ["color", "JP_name", "label"]:
                meta_col = f"{col}_meta"
                if meta_col in df.columns:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[meta_col])
                    else:
                        df[col] = df[meta_col]
                    df = df.drop(columns=[meta_col])
            if "name" in df.columns:
                df = df.drop(columns=["name"])

        output_folder = out_root / clip_name
        output_folder.mkdir(parents=True, exist_ok=True)
        image_files = sorted([f for f in img_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frame_data = df[df["No"] == i]
            if frame_data.empty:
                cv2.imwrite(str(output_folder / img_path.name), img)
                continue

            if japanese and font is not None:
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                for _, row in frame_data.iterrows():
                    if row["id"] <= 0:
                        continue
                    x1, y1, x2, y2 = get_box_coords(row)
                    color_rgb = hex_to_rgb(row.get("color", "#00FF00"))
                    label_text = get_label_text(row, japanese=True)
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=4)
                    draw_text_with_outline(draw, (x1, max(0, y1 - 60)), label_text, font, color_rgb, (0, 0, 0), 2)

                img_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                for _, row in frame_data.iterrows():
                    if row["id"] <= 0:
                        continue
                    x1, y1, x2, y2 = get_box_coords(row)
                    color_bgr = hex_to_bgr(row.get("color", "#00FF00"))
                    label_text = get_label_text(row, japanese=False)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 4)
                    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 8)
                    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, color_bgr, 3)

                img_final = img

            cv2.imwrite(str(output_folder / img_path.name), img_final)

    print(f"Final visualization complete. Results saved to {output_dir}")


def main():
    args = parse_args()

    track_root = Path(args.tracking_dir)
    spec_root = Path(args.species_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file {args.metadata} not found.")
        return
    dflabel = pd.read_csv(args.metadata, encoding="utf-8-sig")

    csv_files = [f for f in os.listdir(track_root) if f.endswith(".csv")]

    for filename in csv_files:
        track_path = track_root / filename
        spec_path = spec_root / filename

        if not spec_path.exists():
            print(f"Species CSV not found for {filename}. Skipping.")
            continue

        print(f"Consolidating {filename}...")
        df_track = pd.read_csv(track_path)
        df_spec = pd.read_csv(spec_path)

        id_counts = df_track["id"].value_counts()
        low_freq_ids = id_counts[id_counts < args.count_thre].index
        df_track.loc[df_track["id"].isin(low_freq_ids), "id"] = -1

        unique_ids = sorted(df_track.loc[~df_track["id"].isin([0, -1]), "id"].unique())
        id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_ids)}
        df_track["id"] = df_track["id"].map(id_mapping).fillna(df_track["id"]).astype(int)

        final_ids = df_track[df_track["id"] > 0]["id"].unique()
        for idx in final_ids:
            track_subset = df_track[df_track["id"] == idx]
            label_list = []

            for _, row in track_subset.iterrows():
                b1 = torch.tensor([row["x"], row["y"], row["w"], row["h"]], dtype=torch.float).unsqueeze(0)

                spec_frame = df_spec[df_spec["No"] == row["No"]]
                if spec_frame.empty:
                    continue

                iou_list = []
                labels_in_frame = []
                for _, s_row in spec_frame.iterrows():
                    if s_row["Scores"] < args.conf_thre:
                        continue
                    b2 = torch.tensor([s_row["x"], s_row["y"], s_row["w"], s_row["h"]], dtype=torch.float).unsqueeze(0)
                    iou = bbox_iou(b1, b2, xywh=True).item()

                    if iou > args.iou_thre:
                        iou_list.append(iou)
                        labels_in_frame.append(s_row["Label"])

                if iou_list:
                    best_idx = np.argmax(iou_list)
                    label_list.append(labels_in_frame[best_idx])

            if len(label_list) >= int(len(track_subset) * args.label_thre):
                chosen_label = Counter(label_list).most_common(1)[0][0]

                if chosen_label in dflabel[dflabel["cls"] == 5]["name"].values:
                    family_num = dflabel[dflabel["name"] == chosen_label]["num"].values[0]
                    refinement_df = dflabel[(dflabel["num"] == family_num) & (dflabel["cls"] < 5)]
                    filtered_labels = [l for l in label_list if l in refinement_df["name"].values]

                    if filtered_labels:
                        counts = Counter(filtered_labels)
                        valid_labels = [l for l, c in counts.items() if c >= int(len(label_list) * args.proportion_thre)]
                        if valid_labels:
                            label_cls = np.array([int(dflabel[dflabel["name"] == l]["cls"].values[0]) for l in valid_labels])
                            if 1 in label_cls:
                                species_indices = np.where(label_cls == 1)[0]
                                if len(species_indices) == 1:
                                    chosen_label = valid_labels[species_indices[0]]
                                else:
                                    sub_counts = {l: counts[l] for l in [valid_labels[i] for i in species_indices]}
                                    chosen_label = max(sub_counts, key=sub_counts.get)
                            else:
                                min_cls = min(label_cls)
                                min_indices = np.where(label_cls == min_cls)[0]
                                sub_counts = {l: counts[l] for l in [valid_labels[i] for i in min_indices]}
                                chosen_label = max(sub_counts, key=sub_counts.get)

                elif chosen_label in dflabel[dflabel["cls"] == 4]["name"].values:
                    row_meta = dflabel[dflabel["name"] == chosen_label].iloc[0]
                    refinement_df = dflabel[
                        (dflabel["num"] == row_meta["num"]) &
                        (dflabel["genus"] == row_meta["genus"]) &
                        (dflabel["cls"] < 4)
                    ]
                    filtered_labels = [l for l in label_list if l in refinement_df["name"].values]

                    if filtered_labels:
                        counts = Counter(filtered_labels)
                        valid_labels = [l for l, c in counts.items() if c >= int(len(label_list) * args.proportion_thre)]
                        if valid_labels:
                            label_cls = np.array([int(dflabel[dflabel["name"] == l]["cls"].values[0]) for l in valid_labels])
                            if 1 in label_cls:
                                species_indices = np.where(label_cls == 1)[0]
                                if len(species_indices) == 1:
                                    chosen_label = valid_labels[species_indices[0]]
                                else:
                                    sub_counts = {l: counts[l] for l in [valid_labels[i] for i in species_indices]}
                                    chosen_label = max(sub_counts, key=sub_counts.get)
                            else:
                                min_cls = min(label_cls)
                                min_indices = np.where(label_cls == min_cls)[0]
                                sub_counts = {l: counts[l] for l in [valid_labels[i] for i in min_indices]}
                                chosen_label = max(sub_counts, key=sub_counts.get)

                elif chosen_label in dflabel[(dflabel["cls"] == 2) | (dflabel["cls"] == 3)]["name"].values:
                    sp_grp = dflabel[dflabel["name"] == chosen_label]["sp_grp"].values[0]
                    refinement_df = dflabel[dflabel["sp_grp"] == sp_grp]
                    filtered_labels = [l for l in label_list if l in refinement_df["name"].values]

                    if filtered_labels:
                        counts = Counter(filtered_labels)
                        valid_labels = [l for l, c in counts.items() if c >= int(len(label_list) * args.proportion_thre)]
                        if valid_labels:
                            label_cls = np.array([int(dflabel[dflabel["name"] == l]["cls"].values[0]) for l in valid_labels])
                            if 1 in label_cls:
                                species_indices = np.where(label_cls == 1)[0]
                                sub_counts = {l: counts[l] for l in [valid_labels[i] for i in species_indices]}
                                chosen_label = max(sub_counts, key=sub_counts.get)
            else:
                chosen_label = "unID"

            df_track.loc[df_track["id"] == idx, "Label"] = chosen_label

        df_track = df_track.merge(
            dflabel[["name", "cls", "sp_grp", "label", "JP_name"]],
            left_on="Label",
            right_on="name",
            how="left",
        )

        save_path = out_root / filename
        df_track.to_csv(save_path, index=False)

    print(f"Consolidation complete. Results saved to {args.output_dir}")

    if args.save_images:
        if not args.img_dir:
            print("Warning: --save-images enabled but --img-dir not provided. Skipping visualization.")
        else:
            visualize_consolidated_results(
                csv_dir=args.output_dir,
                img_root=args.img_dir,
                output_dir=args.save_img_dir,
                metadata_path=args.metadata,
                japanese=args.japanese,
            )


if __name__ == "__main__":
    main()
