# TP2: IoU-based Tracking (Bounding-Box Tracker)

## Overview

This module implements a Single/Multiple Object Tracker using IoU
(Intersection over Union) association and the Hungarian algorithm.

## Requirements

- `scipy` (added to `flake.nix`, requires environment reload)
- `ADL-Rundle-6` dataset (MOT15)

## Setup

1. Ensure the `ADL-Rundle-6` dataset is available. By default, the code expects:

   ```text
   tp2/data/ADL-Rundle-6/
       img1/
           000001.jpg
           ...
       det/
           det.txt
   ```

   You can link the dataset here or modify `main.py`.

2. Reload the Nix environment to pick up `scipy`:

   ```bash
   exit
   nix develop
   ```

## Running

```bash
python tp2/main.py
```

## Output

- `output_tp2.avi`: Visualization of tracking.
- `ADL-Rundle-6.txt`: Tracking results in MOT format.
