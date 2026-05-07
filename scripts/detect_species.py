#!/usr/bin/env python3
"""Run species identification inference using a trained YOLO detection model.

This script processes folders of images, runs a YOLO detection model,
and saves the results (boxes, scores, labels) to CSV files.

Usage:
  python scripts/detect_species.py --input data --weights path/to/species_best.pt
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    parser = argparse.ArgumentParser(description="Fish Species Identification")
    parser.add_argument("--input", type=str, default="data", help="Input directory containing image folders")
    parser.add_argument("--weights", type=str, default="runs/training/fish_id_bbox/run4/train/weights/best.pt", help="Path to YOLO species detection weights")
    parser.add_argument("--save-csv", type=str, default="runs/analysis/csv_species", help="Output directory for CSV results")
    parser.add_argument("--save-images", action="store_true", help="Save annotated images")
    parser.add_argument("--save-img-dir", type=str, default="runs/analysis/images_species", help="Output directory for annotated images")
    parser.add_argument("--conf", type=float, default=0.376, help="Confidence threshold (default 0.376 from notebook)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default 0.5 from notebook)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for inference")
    parser.add_argument("--device", default="0", help="Device (0, cpu, etc.)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_root = Path(args.input)
    csv_root = Path(args.save_csv)
    weights_path = Path(args.weights)
    
    # Load model
    print(f"Loading species model: {weights_path}")
    model = YOLO(str(weights_path))
    
    folders = [f for f in input_root.iterdir() if f.is_dir()]
    csv_root.mkdir(parents=True, exist_ok=True)
    
    img_root = Path(args.save_img_dir)
    if args.save_images:
        img_root.mkdir(parents=True, exist_ok=True)
    
    for folder in folders:
        csv_path = csv_root / f"{folder.name}.csv"
        if csv_path.exists():
            print(f"⏭️ Skipping {folder.name} (results exist)")
            continue
            
        print(f"Processing species detection for: {folder.name}")
        image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if not image_files:
            continue
            
        output_img_folder = img_root / folder.name
        if args.save_images:
            output_img_folder.mkdir(parents=True, exist_ok=True)
            
        header = ['No', 'Class', 'Label', 'Scores', 'x', 'y', 'w', 'h']
        
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
                    batch=args.batch_size,
                    device=args.device,
                    verbose=False
                )
                
                for idx, (img_path, res) in enumerate(zip(batch_paths, results)):
                    frame_idx = i + idx
                    if not res.boxes:
                        if args.save_images:
                            # Still save image even without detections for consistency
                            frame = cv2.imread(img_path)
                            out_path = output_img_folder / Path(img_path).name
                            cv2.imwrite(str(out_path), frame)
                        continue
                        
                    for box in res.boxes:
                        cls = int(box.cls[0])
                        label = res.names[cls]
                        score = float(box.conf[0])
                        # cx, cy, w, h format for tracking consistency
                        cx, cy, w, h = box.xywh[0].cpu().numpy()
                        
                        writer.writerow([frame_idx, cls, label, score, cx, cy, w, h])
                        
                    if args.save_images:
                        annotated_frame = res.plot(boxes=True, labels=True)
                        out_path = output_img_folder / Path(img_path).name
                        cv2.imwrite(str(out_path), annotated_frame)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Species detection complete. Results saved to {args.save_csv}")

if __name__ == "__main__":
    main()
