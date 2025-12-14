import csv
import os
from typing import Dict, List, Tuple


def load_detections(det_file_path: str) -> Dict[int, List[List[float]]]:
    """
    Load detections from a MOT-challenge formatted text file.

    Args:
        det_file_path: Path to the detection file (e.g., det.txt).

    Returns:
        A dictionary where keys are frame numbers and values are lists of detections.
        Each detection is a list: [id, bb_left, bb_top, bb_width, bb_height, conf]
        The id is set to -1 for initial detections.
    """
    detections = {}
    if not os.path.exists(det_file_path):
        print(f"Warning: Detection file not found at {det_file_path}")
        return detections

    with open(det_file_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row:
                continue
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            frame_idx = int(float(row[0]))
            # id = int(float(row[1])) # Usually -1 in det.txt
            bb_left = float(row[2])
            bb_top = float(row[3])
            bb_width = float(row[4])
            bb_height = float(row[5])
            conf = float(row[6])

            # Store as [id, x, y, w, h, conf]
            # We set ID to -1 initially as per instructions
            det = [-1, bb_left, bb_top, bb_width, bb_height, conf]

            if frame_idx not in detections:
                detections[frame_idx] = []
            detections[frame_idx].append(det)

    return detections


def save_results(
    output_file_path: str,
    results: List[Tuple[int, int, float, float, float, float, float]],
):
    """
    Save tracking results to a text file in MOT format.

    Args:
        output_file_path: Path to save the results.
        results: List of tuples (frame, id, bb_left, bb_top, bb_width, bb_height, conf).
    """
    with open(output_file_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for res in results:
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
            line = list(res) + [-1, -1, -1]
            writer.writerow(line)
