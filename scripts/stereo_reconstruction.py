#!/usr/bin/env python3
"""3D Stereo Reconstruction and Length Measurement.

Reconciles two cameras (A and D) to produce 3D coordinates and body lengths.
Follows the notebook's hierarchical ID matching and epipolar constraints.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Stereo 3D Reconstruction")
    parser.add_argument("--input-dir", type=str, default="runs/analysis/csv_consolidated", help="Dir with consolidated 2D CSVs")
    parser.add_argument("--output-dir", type=str, default="runs/analysis/csv_3d", help="Output directory")
    parser.add_argument("--percentage", type=float, default=0.2, help="Y-tolerance as % of height")
    parser.add_argument("--count_thre", type=int, default=15, help="Min frames for a matched track")
    return parser.parse_args()

def undistort_points(x, y, camera_matrix, dist_coeffs, rectification_matrix, projection_matrix):
    points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, R=rectification_matrix, P=projection_matrix)
    return undistorted[0][0]

def triangulate_3D(xy_left, xy_right, proj_left, proj_right):
    xy_left = np.array([[xy_left]], dtype=np.float32).T
    xy_right = np.array([[xy_right]], dtype=np.float32).T
    point_4D = cv2.triangulatePoints(proj_left, proj_right, xy_left, xy_right)
    # Homogeneous to 3D
    point_3D = np.array([point_4D[0], point_4D[1], point_4D[2]]) / point_4D[3]
    return point_3D.flatten()

def process_dataframe(df, camera_matrix, dist_coeffs, rect_matrix, proj_matrix):
    # Required columns from original or consolidated CSV
    rect_results = []
    for _, row in df.iterrows():
        # Points to rectify
        points = {
            'center': (row['x'], row['y']),
            'x1y1': (row['x1'], row['y1']),
            'x2y2': (row['x2'], row['y2']),
            'head': (row['head_x'], row['head_y']),
            'tail': (row['tail_x'], row['tail_y'])
        }
        rectified = {}
        for k, (px, py) in points.items():
            rx, ry = undistort_points(px, py, camera_matrix, dist_coeffs, rect_matrix, proj_matrix)
            rectified[f"{k}_rect_x"] = rx
            rectified[f"{k}_rect_y"] = ry
        
        # Merge rectified into original row
        new_row = row.copy()
        for k, v in rectified.items():
            new_row[k] = v
        rect_results.append(new_row)
    return pd.DataFrame(rect_results)

def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Date ranges from notebook
    DATE_RANGE_2024 = (datetime(2024, 12, 12, 8, 0), datetime(2024, 12, 13, 17, 0))
    DATE_RANGE_2025_1 = (datetime(2025, 11, 7, 12, 0), datetime(2025, 11, 17, 11, 0))
    DATE_RANGE_2025_2 = (datetime(2025, 11, 17, 13, 0), datetime(2025, 11, 27, 10, 0))

    csv_files = sorted([f.name for f in input_path.glob("*_A.csv")])
    
    for a_filename in csv_files:
        time_prefix = a_filename.replace("_A.csv", "")
        d_filename = f"{time_prefix}_D.csv"
        d_path = input_path / d_filename
        
        if not d_path.exists():
            print(f"⚠️ Stereo pair not found for {time_prefix}. Skipping.")
            continue
            
        print(f"Processing Stereo pair: {time_prefix}...")
        
        # Parse time for calibration selection
        try:
            file_date = datetime.strptime(time_prefix, "%Y_%m%d_%H%M")
        except ValueError:
            print(f"⚠️ Unrecognized date format in {time_prefix}. Skipping.")
            continue
            
        # Select Calibration
        xml_path = None
        flip = False
        if DATE_RANGE_2024[0] <= file_date <= DATE_RANGE_2024[1]:
            xml_path = "camera_calibration/2024/stereo_calibration.xml"
            flip = False
        elif DATE_RANGE_2025_1[0] <= file_date <= DATE_RANGE_2025_1[1]:
            xml_path = "camera_calibration/2025_1/stereo_calibration.xml"
            flip = True
        elif DATE_RANGE_2025_2[0] <= file_date <= DATE_RANGE_2025_2[1]:
            xml_path = "camera_calibration/2025_2/stereo_calibration.xml"
            flip = True
            
        if not xml_path or not os.path.exists(xml_path):
            print(f"⚠️ Calibration file not found for {file_date}. Skipping.")
            continue
            
        # Load params
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        mtx_l = fs.getNode("mtx_left").mat()
        mtx_r = fs.getNode("mtx_right").mat()
        dist_l = fs.getNode("dist_left").mat()
        dist_r = fs.getNode("dist_right").mat()
        rect_l = fs.getNode("rect_left").mat()
        rect_r = fs.getNode("rect_right").mat()
        proj_l = fs.getNode("proj_left").mat()
        proj_r = fs.getNode("proj_right").mat()
        newmtx_l = fs.getNode("newcameramtx_left").mat()
        newmtx_r = fs.getNode("newcameramtx_right").mat()
        fs.release()

        # Load Data
        df_A = pd.read_csv(input_path / a_filename)
        df_D = pd.read_csv(d_path)
        
        # Rectify
        if flip:
            df_A = process_dataframe(df_A, newmtx_l, dist_l, rect_l, proj_l)
            df_D = process_dataframe(df_D, newmtx_r, dist_r, rect_r, proj_r)
        else:
            df_A = process_dataframe(df_A, newmtx_r, dist_r, rect_r, proj_r)
            df_D = process_dataframe(df_D, newmtx_l, dist_l, rect_l, proj_l)
            
        # Matching Detections
        # Note: id > 0 are tracked fish. id=0 are detections that weren't tracked.
        df_A = df_A[df_A['id'] > 0]
        combined_rows = []
        for _, row_A in df_A.iterrows():
            df_D_sub = df_D[df_D['No'] == row_A['No']]
            tol = args.percentage * row_A['h']
            
            # Notebook logic: center, edge1, edge2 must all be within tol
            # Here we use rectified centers primarily.
            matches = df_D_sub[
                (abs(df_D_sub['center_rect_y'] - row_A['center_rect_y']) <= tol) &
                (abs(df_D_sub['x1y1_rect_y'] - row_A['x1y1_rect_y']) <= tol) &
                (abs(df_D_sub['x2y2_rect_y'] - row_A['x2y2_rect_y']) <= tol)
            ]
            
            for _, match_row in matches.iterrows():
                match_renamed = match_row.add_prefix('D_')
                combined = pd.concat([row_A, match_renamed])
                combined_rows.append(combined)

        if not combined_rows:
            print(f"   No matched candidates for {time_prefix}")
            continue
            
        df_combined = pd.DataFrame(combined_rows)
        
        # ID Voting (One ID_A matches One ID_D)
        id_pair_counts = df_combined.groupby(['id', 'D_id']).size().reset_index(name='pair_count')
        total_counts = df_combined.groupby('id').size().reset_index(name='total_rows')
        pairs = pd.merge(id_pair_counts, total_counts, on='id')
        
        # Take max count pair for each ID_A
        max_idx = pairs.groupby('id')['pair_count'].idxmax()
        id_map = pairs.loc[max_idx]
        id_map = id_map[id_map['pair_count'] >= args.count_thre]
        
        # Final filter
        df_matched = df_combined.merge(id_map[['id', 'D_id']], on=['id', 'D_id'], how='inner')
        
        # Triangulation
        res_3d = []
        for _, row in df_matched.iterrows():
            pts_A = {
                'center': (row['center_rect_x'], row['center_rect_y']),
                'head': (row['head_rect_x'], row['head_rect_y']),
                'tail': (row['tail_rect_x'], row['tail_rect_y'])
            }
            pts_D = {
                'center': (row['D_center_rect_x'], row['D_center_rect_y']),
                'head': (row['D_head_rect_x'], row['D_head_rect_y']),
                'tail': (row['D_tail_rect_x'], row['D_tail_rect_y'])
            }
            
            # Check head/tail consistency
            vecA = np.array(pts_A['head']) - np.array(pts_A['center'])
            vecD = np.array(pts_D['head']) - np.array(pts_D['center'])
            if np.sign(vecA[0]) != np.sign(vecD[0]):
                pts_D['head'], pts_D['tail'] = pts_D['tail'], pts_D['head']
            
            # Triangulate
            p3d = {}
            for k in ['center', 'head', 'tail']:
                if flip:
                    p = triangulate_3D(pts_A[k], pts_D[k], proj_l, proj_r)
                else:
                    p = triangulate_3D(pts_D[k], pts_A[k], proj_l, proj_r)
                # cv2 result: x=right, y=down, z=forward
                # target: X=parallel, Y=depth from camera, Z=height (vertical) - actually notebook just uses X, Y, Z.
                # Let's keep raw X, Y, Z (meters if square_size=1 and baseline in meters)
                p3d[f"{k}_3d_x"] = p[0]
                p3d[f"{k}_3d_y"] = p[1]
                p3d[f"{k}_3d_z"] = p[2]
            
            # Length
            d_head = np.array([p3d['head_3d_x'], p3d['head_3d_y'], p3d['head_3d_z']])
            d_tail = np.array([p3d['tail_3d_x'], p3d['tail_3d_y'], p3d['tail_3d_z']])
            length = np.linalg.norm(d_head - d_tail)
            
            new_row = row.copy()
            for k, v in p3d.items(): new_row[k] = v
            new_row['fish_length'] = length
            res_3d.append(new_row)
            
        df_final = pd.DataFrame(res_3d)
        df_final.to_csv(output_path / f"{time_prefix}.csv", index=False)
        print(f"   Saved 3D results for {time_prefix}. Matched IDs: {len(id_map)}")

if __name__ == "__main__":
    main()
