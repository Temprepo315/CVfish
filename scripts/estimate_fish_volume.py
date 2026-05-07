#!/usr/bin/env python3
"""
Estimate fish volume from 3D point cloud data using the notebook's analytical framework.

Standard Analysis Sections:
1. Calibration 4 Sec 1 (Default): 
   - Start: 2024-12-12 08:00:00, End: 2024-12-13 17:00:00
   - Expected: ~50.60 m^3, 239,330 instances
2. Calibration 4 Sec 1 (Alt):
   - Start: 2025-11-07 08:00:00, End: 2025-11-09 10:00:00
   - Expected: ~26.11 m^3, 150,985 instances
3. Calibration 4 Sec 2:
   - Start: 2025-11-09 11:00:00, End: 2025-11-17 10:00:00
   - Expected: ~71.23 m^3, 742,729 instances
4. Calibration 3 Sec 1:
   - Start: 2025-11-17 12:00:00, End: 2025-11-27 17:00:00
   - Expected: ~78.17 m^3, 433,078 instances

Usage:
  python scripts/estimate_fish_volume.py --input-csv runs/analysis/csv_3d/summary_all.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import open3d as o3d
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate fish volume and visualize 3D points")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to summary_all.csv")
    parser.add_argument("--start-time", type=str, default="2025-11-09 11:00:00", help="Start time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-time", type=str, default="2025-11-17 10:00:00", help="End time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--species", type=str, help="Optional species name filter")
    parser.add_argument("--output-dir", type=str, default="runs/analysis/images_3d", help="Output directory for plots")
    # Analytical parameters from notebook
    parser.add_argument("--nb-points", type=int, default=50, help="Minimum neighbors for outlier removal")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius for outlier removal (meters)")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size for downsampling (visualization only)")
    return parser.parse_args()

def visualize_projections(df_sub, start, end, args, output_path):
    """
    Visualizes 2D projections of the 3D point cloud with strictly defined 
    styles and axis limits as per research requirements.
    """
    ticksize = 22
    xyzsize = 26
    
    # Define axis limits
    xz_y_range = 4  # Z range (-2 to 2)
    xy_y_range = 10 # Y range (0 to 10)

    # Compute height ratios to reflect real-world scale
    height_ratios = [xz_y_range, xy_y_range]

    # Create subplots with scaled height ratios
    fig, axes = plt.subplots(2, 1, figsize=(8, 14), gridspec_kw={'height_ratios': height_ratios})

    # XZ Projection (Side View)
    ax1 = axes[0]
    sc1 = ax1.scatter(
        df_sub['X_xy'],
        df_sub['Z_xy'],
        c=df_sub['Y_xy'],
        cmap='viridis', vmin=0, vmax=10,
        marker='o',
        alpha=1,
        s=0.1
    )
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel("X", fontsize=xyzsize)
    ax1.set_ylabel("Z", fontsize=xyzsize)
    ax1.set_aspect(1)
    ax1.tick_params(labelsize=ticksize)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("Y", fontsize=xyzsize)
    cbar1.ax.tick_params(labelsize=ticksize)

    # XY Projection (Top View)
    ax2 = axes[1]
    sc2 = ax2.scatter(
        df_sub['X_xy'],
        df_sub['Y_xy'],
        c=df_sub['Z_xy'],
        cmap='viridis', vmin=-2, vmax=2,
        marker='o',
        alpha=1,
        s=0.1
    )
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel("X", fontsize=xyzsize)
    ax2.set_ylabel("Y", fontsize=xyzsize)
    ax2.set_aspect(1)
    ax2.tick_params(labelsize=ticksize)
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Z", fontsize=xyzsize)
    cbar2.ax.tick_params(labelsize=ticksize)

    plt.tight_layout()
    plot_file = output_path / f"projection_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}.png"
    plt.savefig(plot_file)
    print(f"Plots saved to {plot_file}")

def main():
    args = parse_args()
    
    # 1. Load Data
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} not found.")
        return

    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # 2. Filter by Time (Primary)
    print(f"Filtering by time: {args.start_time} to {args.end_time}")
    df['date'] = pd.to_datetime(df['date'])
    start = pd.Timestamp(args.start_time)
    end = pd.Timestamp(args.end_time)
    
    df_sub = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    
    # 3. Filter by Species (Optional)
    if args.species:
        print(f"Filtering by species: {args.species}")
        if 'refined_name' in df_sub.columns:
            df_sub = df_sub[df_sub['refined_name'] == args.species]
        elif 'name' in df_sub.columns:
            df_sub = df_sub[df_sub['name'] == args.species]
        else:
            print("Warning: Species column not found. Skipping species filter.")

    if df_sub.empty:
        print("No data found for the given filters.")
        return

    print(f"Found {len(df_sub)} points.")

    # 4. Visualize 2x2D plots
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    visualize_projections(df_sub, start, end, args, output_path)

    # 5. Volume Estimation (Matching Notebook Analytical Framework)
    pts = np.vstack([
        df_sub["X_xy"].values,
        df_sub["Y_xy"].values,
        df_sub["Z_xy"].values
    ]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    print(f"Applying radius outlier removal (nb_points={args.nb_points}, radius={args.radius})...")
    # Outlier removal 
    pcd_clean, ind = pcd.remove_radius_outlier(
        nb_points=args.nb_points,
        radius=args.radius
    )

    inlier_pts = np.asarray(pcd.points)[ind]
    if len(inlier_pts) >= 4:
        pcd_inlier = o3d.geometry.PointCloud()
        pcd_inlier.points = o3d.utility.Vector3dVector(inlier_pts)

        print("Computing convex hull...")
        hull, _ = pcd_inlier.compute_convex_hull()
        hull_volume = hull.get_volume()
        hull_area = hull.get_surface_area()

        print("-" * 30)
        print(f"Volume Estimation Results (Open3D):")
        print(f"  Total points:      {len(pts)}")
        print(f"  Inlier points:     {len(inlier_pts)}")
        print(f"  Estimated Volume:  {hull_volume:.6f} m^3")
        print(f"  Surface Area:      {hull_area:.6f} m^2")
        print("-" * 30)
    else:
        print(f"Not enough inlier points to calculate volume (Found {len(inlier_pts)}, minimum 4 required).")

if __name__ == "__main__":
    main()
