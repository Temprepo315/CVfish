"""Run fish segmentation inference and calculate orientation (head-tail axis).

This script processes folders of images, runs a YOLO segmentation model, 
and calculates the fish's centerline, orientation angle, and key points (head, center, tail)
using skeletonization and PCA.

Usage:
  python scripts/predict_fish_segmentation.py --input data --weights runs/training/fish_segmentation/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fish segmentation and orientation prediction")
    parser.add_argument("--input", type=str, default="data", help="Input directory containing subfolders of images")
    parser.add_argument("--weights", type=str, default="runs/training/fish_segmentation/train/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--save-csv", type=str, default="runs/analysis/csv_fishseg", help="Output directory for CSV results")
    parser.add_argument("--save-img", type=str, default="runs/analysis/images_fishseg", help="Output directory for annotated images")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default 0.1 for high recall)")
    parser.add_argument("--iou", type=float, default=0.2, help="IoU threshold for NMS")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for inference")
    parser.add_argument("--device", default="0", help="Device (0, 1, cpu, etc.)")
    parser.add_argument("--save-images", action="store_true", help="Save annotated images")
    parser.add_argument("--folder-pattern", type=str, default="", help="Filter folders by suffix (e.g. 'A' or 'D')")
    return parser.parse_args()


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return (0, 255, 0)  # Default green
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def line_polygon_intersections(polygon: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> np.ndarray:
    """Find intersection points between a polygon and a line."""
    intersections = []
    line_dir = line_dir / np.linalg.norm(line_dir)

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edge = p2 - p1
        A = np.array([line_dir, -edge]).T
        b = p1 - line_point
        if np.linalg.matrix_rank(A) < 2:
            continue
        try:
            t_s = np.linalg.solve(A, b)
            t, s = t_s
            if 0 <= s <= 1:
                intersections.append(line_point + t * line_dir)
        except np.linalg.LinAlgError:
            continue
    return np.array(intersections)


def get_fish_axis_from_polygon_intersection(polygon_xy: np.ndarray) -> dict | None:
    """Compute skeleton, PCA axis through skeleton center, and find endpoints as polygon-line intersections."""
    pts = np.array(polygon_xy, dtype=np.float32)
    if pts.shape[0] < 3:
        return None

    # --- Skeletonization ---
    xmin, ymin = np.min(pts, axis=0).astype(int)
    xmax, ymax = np.max(pts, axis=0).astype(int)
    w, h = xmax - xmin + 1, ymax - ymin + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_pts = np.array([(x - xmin, y - ymin) for x, y in pts], dtype=np.int32)
    cv2.fillPoly(mask, [shifted_pts], 1)

    skel = skeletonize(mask > 0)
    skel_coords = np.column_stack(np.where(skel))
    if skel_coords.shape[0] < 3:
        return None
    skel_coords = skel_coords[:, [1, 0]] + np.array([xmin, ymin])

    # --- PCA on skeleton ---
    mean_skel, eigen_skel = cv2.PCACompute(skel_coords.astype(np.float32), mean=np.array([]))
    center_skel = mean_skel[0]
    axis_skel = eigen_skel[0]

    # --- Find polygon intersections with axis line ---
    intersections = line_polygon_intersections(pts, center_skel, axis_skel)
    if intersections.shape[0] < 2:
        proj = np.dot(pts - center_skel, axis_skel)
        head = pts[np.argmin(proj)]
        tail = pts[np.argmax(proj)]
    else:
        t_values = np.dot(intersections - center_skel, axis_skel)
        head = intersections[np.argmin(t_values)]
        tail = intersections[np.argmax(t_values)]

    return {
        "skeleton": skel_coords,
        "axis_vector": axis_skel,
        "center_skel": center_skel,
        "head": head,
        "tail": tail
    }


def main() -> None:
    args = parse_args()

    # Paths
    input_root = Path(args.input)
    csv_root = Path(args.save_csv)
    img_root = Path(args.save_img)
    weights_path = Path(args.weights)
    
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    # Metadata for colors/labels
    label_csv = METADATA_DIR / "class_summary_with_JP_labeled.csv"
    if label_csv.exists():
        df_label = pd.read_csv(label_csv)
        name_to_color = dict(zip(df_label["name"], df_label["color"]))
    else:
        print(f"Warning: Metadata not found at {label_csv}. Using default colors.")
        name_to_color = {}

    # Load Model
    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Folders to process
    folders = [f for f in input_root.iterdir() if f.is_dir()]
    if args.folder_pattern:
        folders = [f for f in folders if f.name.endswith(args.folder_pattern)]

    csv_root.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        csv_path = csv_root / f"{folder.name}.csv"
        if csv_path.exists():
            print(f"⏭️ Skipping {folder.name} (results exist)")
            continue

        print(f"Processing folder: {folder.name}")
        image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if not image_files:
            continue

        output_img_folder = img_root / folder.name
        if args.save_images:
            output_img_folder.mkdir(parents=True, exist_ok=True)

        header = ['No', 'Class', 'Label', 'Scores', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'w', 'h', 
                  'head_x', 'head_y', 'center_x', 'center_y', 'tail_x', 'tail_y', 'angle']

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(0, len(image_files), args.batch_size):
                batch_files = image_files[i:i + args.batch_size]
                batch_paths = [str(f) for f in batch_files]

                results = model.predict(
                    source=batch_paths,
                    conf=args.conf,
                    iou=args.iou,
                    agnostic_nms=True,
                    retina_masks=True,
                    batch=args.batch_size,
                    device=args.device,
                    verbose=False
                )

                for idx, (img_path, res) in enumerate(zip(batch_paths, results)):
                    frame_idx = i + idx
                    frame = None
                    if args.save_images:
                        frame = cv2.imread(img_path)

                    if not res.boxes or not res.masks:
                        continue

                    for box, mask in zip(res.boxes, res.masks):
                        cls = int(box.cls[0])
                        label = res.names[cls]
                        score = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy, w, h = box.xywh[0].cpu().numpy()

                        # Process mask for orientation
                        seg_xy = mask.xy[0]
                        axis_info = get_fish_axis_from_polygon_intersection(seg_xy)
                        if axis_info is None:
                            continue

                        head_x, head_y = axis_info["head"]
                        center_x, center_y = axis_info["center_skel"]
                        tail_x, tail_y = axis_info["tail"]

                        # Calculate angle
                        dx = head_x - tail_x
                        dy = head_y - tail_y
                        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 180

                        # Drawing
                        if args.save_images and frame is not None:
                            # Use YOLO result.plot() for premium mask visuals
                            # We draw one by one to overlay our custom orientation markers
                            # However, result.plot() plots ALL boxes in the result.
                            # So we will do it once per image after the loop or use the plotted frame.
                            pass

                        # Write to CSV
                        writer.writerow([
                            frame_idx, cls, label, score,
                            x1, y1, x2, y2, cx, cy, w, h,
                            head_x, head_y, center_x, center_y, tail_x, tail_y, angle
                        ])

                    if args.save_images and frame is not None:
                        # 1. Get standard YOLO visualization (filled masks)
                        # We use labels=True to match notebook, but boxes=False to avoid clutter if desired
                        annotated_frame = res.plot(boxes=True, labels=True, masks=True)
                        
                        # 2. Overlay our orientation markers for all fish detected in this image
                        # Re-loop through results to draw markers on the annotated_frame
                        for box, mask in zip(res.boxes, res.masks):
                            seg_xy = mask.xy[0]
                            axis_info = get_fish_axis_from_polygon_intersection(seg_xy)
                            if axis_info:
                                head_x, head_y = axis_info["head"]
                                center_x, center_y = axis_info["center_skel"]
                                tail_x, tail_y = axis_info["tail"]
                                
                                # Draw orientation line
                                cv2.line(annotated_frame, (int(head_x), int(head_y)), (int(tail_x), int(tail_y)), (255, 255, 0), 2)
                                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                        
                        out_path = output_img_folder / Path(img_path).name
                        cv2.imwrite(str(out_path), annotated_frame)

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Done. Results saved to {args.save_csv}")


if __name__ == "__main__":
    main()
