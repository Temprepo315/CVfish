# Fish Analysis Framework

Pipeline for fish video analysis using python:
1. fish segmentation
2. multi-object tracking
3. species detection
4. tracking/species consolidation
5. stereo 3D reconstruction
6. volume estimation


Pipeline for plotting using R:
1. Length weight plotting
2. Time series plotting
3. Comparison plotting (Video, UVC, eDNA)

## Repository Layout

- `scripts/`: production Python/R scripts.
- `data/`: input image folders (for example `2024_1212_0800_A`, `2024_1212_0800_D`). We have 50 images to test the pipeline.
- `training_datasets/`: YOLO training datasets (`fish_segmentation`, `species_detection_run1..run4`).
- `metadata/`: class metadata and external comparison data (`class_summary_with_JP_labeled.csv`, `dfedna.csv`, `dfuvc.csv`, `dfturb.csv`).
- `camera_calibration/`: stereo calibration XML files by survey period.
- `runs/training/`: YOLO training/evaluation outputs.
- `runs/analysis/`: pipeline outputs (CSV and images). The csv folders contain all results from all images, which are not included in this repository.

## Environment

Main Python dependencies:
- `ultralytics`
- `opencv-python`
- `pandas`
- `numpy`
- `torch`
- `matplotlib`
- `open3d`
- `scikit-image`
- `Pillow`

R packages are used for final analysis/figures (`dplyr`, `ggplot2`, `tidyr`, `vegan`, `pheatmap`, `ragg`, etc.).

## Full Analysis Pipeline (A/D Stereo Pair)

All commands are run from repository root.

### 1) Fish Segmentation

```bash
python scripts/predict_fish_segmentation.py --input data --save-csv runs/analysis/csv_fishseg --save-images --save-img runs/analysis/images_fishseg
```

### 2) Tracking (ByteTrack)

```bash
python scripts/track_fish.py --input_csv_dir runs/analysis/csv_fishseg --output_csv_dir runs/analysis/csv_tracking --visualize --img_dir data --save_img_dir runs/analysis/images_tracking
```

### 3) Species Detection

```bash
python scripts/detect_species.py --input data --save-csv runs/analysis/csv_species --save-images --save-img-dir runs/analysis/images_species
```

### 4) Consolidate Tracking + Species Labels

```bash
python scripts/integrate_species.py --tracking-dir runs/analysis/csv_tracking --species-dir runs/analysis/csv_species --output-dir runs/analysis/csv_consolidated --save-images --img-dir data --save-img-dir runs/analysis/images_consolidated
```

### 5) Stereo 3D Reconstruction

```bash
python scripts/stereo_reconstruction.py --input-dir runs/analysis/csv_consolidated --output-dir runs/analysis/csv_3d
```

### 6) Build `summary_all.csv` (for volume/R_analysis)

`stereo_reconstruction.py` writes one `*.csv` per stereo pair.  
Merge them into one file before volume/R analysis.

PowerShell example:

```powershell
$files = Get-ChildItem runs/analysis/csv_3d -Filter *_3D.csv
if ($files.Count -gt 0) {
  $all = $files | ForEach-Object { Import-Csv $_.FullName }
  $all | Export-Csv runs/analysis/csv_3d/summary_all.csv -NoTypeInformation -Encoding UTF8
}
```

### 7) Volume Estimation

```bash
python scripts/estimate_fish_volume.py --input-csv runs/analysis/csv_3d/summary_all.csv --start-time "2025-11-09 11:00:00" --end-time "2025-11-17 10:00:00"
```


## R Analysis and Comparison Figures

- Main 3D-derived ecological analysis: `scripts/analyze_fish_data.R`
- Method comparison heatmaps (Video/UVC/eDNA): `scripts/compare_methods.R`

Both scripts write outputs under `runs/analysis/`.


## Training and Evaluation
Training codes we used to train our models. Note that our private raw training images are not included in this repository.


### Fish Segmentation Model

```bash
python scripts/train_fish_segmentation.py
python scripts/eval_fish_segmentation_test.py --weights runs/training/fish_segmentation/train/weights/best.pt --split test
```

### Species Detection Model

```bash
python scripts/train_species_detection.py --run run1
python scripts/eval_species_detection_test.py --run run1 --split test
python scripts/eval_species_detection_test.py --run all --split test
```

## Notes

- Paths in scripts are repository-relative.
- Stereo calibration XML is selected from filename date prefix in `stereo_reconstruction.py`.
- Metadata file `metadata/class_summary_with_JP_labeled.csv` is central for class hierarchy, labels, colors, and Japanese names.
