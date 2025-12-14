from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class Track:
    _id_counter = 1

    def __init__(self, box: List[float], info: float = 0.0):
        self.id = Track._id_counter
        Track._id_counter += 1
        self.box = box  # [x, y, w, h]
        self.missed_frames = 0
        self.info = info  # confidence or other info
        self.color = tuple(np.random.randint(0, 255, 3).tolist())

    def update(self, box: List[float], info: float = 0.0):
        self.box = box
        self.missed_frames = 0
        self.info = info

    def mark_missed(self):
        self.missed_frames += 1


class Tracker:
    def __init__(self, max_missed_frames: int = 5, iou_threshold: float = 0.3):
        self.tracks: List[Track] = []
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold

    def update(self, detections: List[List[float]]) -> List[Track]:
        """
        Update tracks with new detections.
        detections: List of [id, x, y, w, h, conf] (id is ignored/placeholder)
        """
        # detections contains [id, x, y, w, h, conf]
        # Extract boxes for IoU
        det_boxes = [d[1:5] for d in detections]
        det_confs = [d[5] for d in detections]

        if not self.tracks:
            # If no tracks, all detections are new tracks
            for i, box in enumerate(det_boxes):
                self.tracks.append(Track(box, det_confs[i]))
            return self.tracks

        if not det_boxes:
            # If no detections, all tracks are missed
            for track in self.tracks:
                track.mark_missed()
            self._cleanup_tracks()
            return self.tracks

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(det_boxes)))
        for t, track in enumerate(self.tracks):
            for d, det_box in enumerate(det_boxes):
                iou_matrix[t, d] = calculate_iou(track.box, det_box)

        # Hungarian Algorithm (maximize IoU)
        # linear_sum_assignment minimizes cost, so we use negative IoU
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        assigned_tracks = set()
        assigned_detections = set()

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.iou_threshold:
                self.tracks[r].update(det_boxes[c], det_confs[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        # Handle unmatched tracks
        for t, track in enumerate(self.tracks):
            if t not in assigned_tracks:
                track.mark_missed()

        # Handle unmatched detections (new tracks)
        for d, det_box in enumerate(det_boxes):
            if d not in assigned_detections:
                self.tracks.append(Track(det_box, det_confs[d]))

        self._cleanup_tracks()
        return self.tracks

    def _cleanup_tracks(self):
        self.tracks = [
            t for t in self.tracks if t.missed_frames <= self.max_missed_frames
        ]
