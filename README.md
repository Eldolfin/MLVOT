# MLVOT Project - Object Tracking

This repository contains the practical assignments (TPs) for the MLVOT course.

## Environment Setup

This project uses **Nix** for reproducible environment management.

1. Install Nix (if not already installed).
2. Enter the development environment:

   ```bash
   nix develop
   ```

   This will install Python 3.13, OpenCV, NumPy, SciPy, and other necessary tools.

## Data Setup

For TP2, TP3, and TP4, you need the **ADL-Rundle-6** dataset (MOT15 format).
Place it in the following structure (or symlink it):

```text
tp2/data/ADL-Rundle-6/
    img1/
        000001.jpg
        ...
    det/
        public-dataset/
            det.txt
```

(The scripts in TP3 and TP4 will look for the data in `tp2/data/` relative to
their execution path).

## Running the Demos

### TP 1: Single Object Tracking with Kalman Filter

Tracks a ball in a video using a Kalman Filter.

```bash
cd tp1
python main.py
```

Output: `tp1/output.avi`

### TP 2: IoU-based Tracking (Bounding-Box Tracker)

Multi-object tracking using IoU and Hungarian Algorithm.

```bash
python tp2/main.py
```

Output: `output_tp2.avi`, `ADL-Rundle-6.txt`

### TP 3: Kalman-Guided IoU Tracking

Extends TP2 with Kalman Filter predictions for better association.

```bash
python tp3/main.py
```

Output: `output_tp3.avi`

### TP 4: Appearance-Aware IoU-Kalman Tracker

Extends TP3 with a ReID model (OSNet) for appearance-based association.

```bash
python tp4/main.py
```

Output: `output_tp4.avi`

## Code Quality

Run the pre-commit hooks to ensure code quality:

```bash
prek
```
