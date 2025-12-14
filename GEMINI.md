# MLVOT Project Context

## Project Overview

This repository contains practical assignments (TPs) for the MLVOT (Machine
Learning for Visual Object Tracking) course, likely part of the SCIA (Data
Science & Artificial Intelligence) curriculum.

The project currently focuses on **TP1: Object Tracking with Kalman Filters**.
It implements a basic object tracking system using computer vision techniques
to detect a moving object (a ball) and a Kalman Filter to predict and smooth
its trajectory.

## Environment & Dependencies

This project uses **Nix** for reproducible environment management.

* **Language:** Python 3.13
* **Key Libraries:** `opencv-python`, `numpy`
* **Tools:** `ruff` (linting/formatting), `ty` (type checking), `ipython`, `ipdb`.

To enter the development environment:

```bash
nix develop
```

## Directory Structure

### `tp1/` (Object Tracking)

* **`main.py`**: The entry point.
  * Reads video from `video/randomball.avi`.
  * Uses `Detector` to find the object center.
  * Uses `KalmanFilter` to predict state.
  * Draws detection (blue box), prediction (red box), and trajectory.
  * Saves result to `output.avi`.
* **`Detector.py`**: Implements object detection using Canny edge detection and
  contour finding to locate circular objects.
* **`KalmanFilter.py`**: (Presumed) Implementation of the Kalman Filter logic.
* **`objTracking.py`**: Currently empty.

### `tp2/`

* Currently empty directory.

### Root Files

* **`flake.nix` / `flake.lock`**: Nix environment configuration.
* **`reid_osnet_x025_market1501.onnx`**: A Pre-trained ReID (Person
  Re-Identification) model, likely for future use (TP2?).

## Development Workflow

### Running the Code

1. Enter the Nix shell: `nix develop`
2. Navigate to the directory or run from root (adjust paths as needed):

    ```bash
    cd tp1
    python main.py
    ```

**Note:** If you add a dependency to `flake.nix`, you must reload the environment.
For single commands without entering the shell, use `nix develop -c <command>`.

### Code Quality

The project enforces code quality via pre-commit hooks configured in `flake.nix`:

* **Formatting:** `alejandra` (for Nix), `ruff-format` (for Python).
* **Linting:** `ruff`.
* **Type Checking:** `ty` (wrapper for a type checker, checking `tp1/`).

To run the full suite of checks/formatting manually:

```bash
# Run pre-commit checks directly after each code update
prek
```

If you are missing a command within the Nix development shell, you can typically
execute it directly using `nix run <package-name> -- <command-arguments>`.

---

**Important:** Always keep this `GEMINI.md` file up to date with any changes to
the project structure, dependencies, or development workflow.
