#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from ultralytics.trackers.byte_tracker import BYTETracker

class FakeResults:
    """Wrapper class to make a DataFrame look like Ultralytics results for BYTETracker."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.conf = self.df['Scores'].to_numpy()
        self.cls = self.df['Class'].to_numpy()

        if 'r' in self.df.columns:
            # Rotated bounding boxes (x, y, w, h, r)
            self.xywhr = self.df[['x', 'y', 'w', 'h', 'r']].to_numpy()
        else:
            # Standard bounding boxes (x, y, w, h)
            self.xywh = self.df[['x', 'y', 'w', 'h']].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return FakeResults(self.df.iloc[idx])

def hex_to_bgr(hex_color):
    if pd.isna(hex_color):
        return (128, 128, 128)
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def main():
    parser = argparse.ArgumentParser(description="Fish Tracking Script using ByteTrack")
    parser.add_argument("--input_csv_dir", type=str, required=True, help="Path to input segmentation CSVs")
    parser.add_argument("--output_csv_dir", type=str, required=True, help="Path to save tracking CSVs")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate of the video (default: 15)")
    parser.add_argument("--track_high_thresh", type=float, default=0.5, help="High confidence threshold for tracking")
    parser.add_argument("--track_low_thresh", type=float, default=0.1, help="Low confidence threshold for tracking")
    parser.add_argument("--new_track_thresh", type=float, default=0.6, help="Confidence threshold for starting a new track")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="IOU matching threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="Number of frames to keep lost tracks")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization (generate annotated images)")
    parser.add_argument("--img_dir", type=str, help="Path to source images (required for visualization)")
    parser.add_argument("--save_img_dir", type=str, default="runs/analysis/images_tracking", help="Output directory for annotated images")
    parser.add_argument("--metadata", type=str, default="metadata/class_summary_with_JP_labeled.csv", help="Path to species metadata CSV")
    
    args = parser.parse_args()

    # ByteTrack expects an object with these attributes
    tracker_args = argparse.Namespace(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        fuse_score=True
    )

    os.makedirs(args.output_csv_dir, exist_ok=True)
    
    # Load metadata for coloring unID or specific species
    if os.path.exists(args.metadata):
        df_meta = pd.read_csv(args.metadata, encoding="utf-8-sig")
        unid_color = df_meta[df_meta['name'] == "unID"]['color'].values[0] if "unID" in df_meta['name'].values else "#808080"
    else:
        df_meta = None
        unid_color = "#808080"

    csv_files = [f for f in os.listdir(args.input_csv_dir) if f.endswith(".csv")]
    
    for filename in csv_files:
        input_path = os.path.join(args.input_csv_dir, filename)
        output_path = os.path.join(args.output_csv_dir, filename)
        
        if os.path.exists(output_path):
            print(f"skipping {filename} (already exists)")
            continue
            
        print(f"Processing tracking for {filename}...")
        df = pd.read_csv(input_path)
        
        # Tracking result columns
        new_cols = ['x1_tra', 'y1_tra', 'x2_tra', 'y2_tra', 'id', 'Scores_tra', 'class_tra']
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Initialize tracker for each video/CSV
        tracker = BYTETracker(tracker_args, frame_rate=args.fps)
        
        # Sort by 'No' (frame index)
        df_sorted = df.sort_values(by='No')
        frames = df_sorted['No'].unique()
        
        results_df = pd.DataFrame()
        
        for frame_no in frames:
            subset = df_sorted[df_sorted['No'] == frame_no].copy()
            fake_results = FakeResults(subset)
            
            if len(fake_results) > 0:
                online_targets = tracker.update(fake_results)
                
                if len(online_targets) > 0:
                    for i in range(len(online_targets)):
                        target = online_targets[i]
                        # target: [x1, y1, x2, y2, track_id, score, class, original_idx]
                        original_idx = int(target[-1])
                        # The track ID is target[4]
                        subset.loc[subset.index[original_idx], new_cols] = target[:-1]
            
            # Use concat instead of append
            results_df = pd.concat([results_df, subset], ignore_index=True)
            
        # Post-processing
        results_df['id'] = results_df['id'].fillna(0).astype(int)
        results_df['class_tra'] = results_df['class_tra'].fillna(0).astype(int)
        
        results_df.to_csv(output_path, index=False)
        print(f"Saved tracking results to {output_path}")

        # Visualization
        if args.visualize:
            if not args.img_dir:
                print("Warning: --visualize enabled but --img_dir not provided. Skipping visualization.")
                continue
            
            # The folder name is usually the CSV filename without .csv
            folder_name = os.path.splitext(filename)[0]
            img_folder = os.path.join(args.img_dir, folder_name)
            
            if not os.path.isdir(img_folder):
                print(f"Warning: Image folder {img_folder} not found. Skipping visualization for {filename}.")
                continue
                
            vis_output_dir = os.path.join(args.save_img_dir, folder_name)
            os.makedirs(vis_output_dir, exist_ok=True)
            
            image_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Match images to CSV rows using the 'No' column (which is the sorted frame index)
            for i, img_name in enumerate(image_files):
                img_path = os.path.join(img_folder, img_name)
                
                # Match No = i
                track_sub = results_df[results_df['No'] == i]
                if track_sub.empty:
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                for _, row in track_sub.iterrows():
                    if row['id'] == 0: continue # Skip untracked objects
                    
                    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                    track_id = int(row['id'])
                    label = row['Label']
                    
                    # Use color from metadata
                    color_hex = unid_color
                    if df_meta is not None and label in df_meta['name'].values:
                        color_hex = df_meta[df_meta['name'] == label]['color'].values[0]
                    
                    color_bgr = hex_to_bgr(color_hex)
                    
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)
                    cv2.putText(img, f"ID:{track_id} {label}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                
                vis_save_path = os.path.join(vis_output_dir, img_name)
                cv2.imwrite(vis_save_path, img)

if __name__ == "__main__":
    main()
