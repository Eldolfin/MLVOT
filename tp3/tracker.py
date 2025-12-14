from typing import List, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import os

# Add tp1 to path to import KalmanFilter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tp1")))

try:
    from KalmanFilter import KalmanFilter
except ImportError:
    # Fallback if running from a different context or tp1 not found
    print("Warning: Could not import KalmanFilter from tp1. Make sure it exists.")

    # Mock for testing if import fails
    class KalmanFilter:
        def __init__(self, dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas):
            pass

        def predict(self):
            pass

        def update(self, z):
            return np.zeros(4)


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
        self.info = info  # confidence
        self.color = tuple(np.random.randint(0, 255, 3).tolist())

        # Initialize Kalman Filter for this track
        # Parameters from TP1 instructions (or tweaked for this problem)
        # dt=0.1, u_x=1, u_y=1, std_acc=1, x_dt_meas=0.1, y_dt_meas=0.1
        # Note: In TP1, state is [x, y, vx, vy].
        # Here we track centroids.
        # Box is [x, y, w, h]. Centroid is [x + w/2, y + h/2].

        self.kf: Any = KalmanFilter(
            dt=1.0,  # Assuming 1 frame per step, or use actual fps
            u_x=0,  # Assuming constant velocity mostly?
            u_y=0,
            std_acc=1,  # Process noise
            x_dt_meas=0.1,  # Measurement noise
            y_dt_meas=0.1,
        )

        # Initialize state with current centroid
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        # We need to manually set the initial state in the KF if it doesn't support it in __init__
        # TP1 KF initialized to zeros. Let's update it once with the initial measurement.
        # self.kf.state = np.array([cx, cy, 0, 0])
        setattr(self.kf, "state", np.array([cx, cy, 0, 0]))
        # Hack: directly setting internal state 'state' because TP1 KF didn't have a setter?
        # Checking TP1/KalmanFilter.py:
        # self.state = np.zeros(4)
        # It's public.

        # We also need to store width and height to reconstruct box from centroid prediction
        self.w = box[2]
        self.h = box[3]

    def predict(self):
        """
        Predict the next state using Kalman Filter.
        Updates the bounding box based on predicted centroid.
        """
        self.kf.predict()
        # kf.state is [x, y, vx, vy]
        pred_cx = self.kf.state[0]
        pred_cy = self.kf.state[1]

        # Update box based on predicted centroid and stored dimensions
        # box = [x, y, w, h] => x = cx - w/2
        self.box = [pred_cx - self.w / 2, pred_cy - self.h / 2, self.w, self.h]

    def update(self, box: List[float], info: float = 0.0):
        """
        Update track with a new measurement (detection).
        """
        # Measurement update for KF
        # Measurement z is [cx, cy]
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2

        # Call KF update
        # TP1 KF update takes 'cur_pos' which seems to be expected as np.array/list
        # In TP1 main.py: pred = kalman_filter.update(center) where center is np.array([[x], [y]])
        # Let's check TP1 KF again.
        # def update(self, cur_pos: np.ndarray): ... (cur_pos.T - self.H @ self.alexis)
        # It expects a column vector or array that can be transposed?
        # TP1 Detector returns np.array([[x], [y]]) -> shape (2, 1)
        z = np.array([[cx], [cy]])

        self.kf.update(z)

        # Update stored box dimensions (optional, could filter these too)
        self.w = box[2]
        self.h = box[3]

        # Update box with the CORRECTED state (posterior) from KF
        # After update(), kf.state is the updated state
        updated_cx = self.kf.state[0]
        updated_cy = self.kf.state[1]

        self.box = [updated_cx - self.w / 2, updated_cy - self.h / 2, self.w, self.h]

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
        detections: List of [id, x, y, w, h, conf]
        """
        det_boxes = [d[1:5] for d in detections]
        det_confs = [d[5] for d in detections]

        # 1. Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        if not self.tracks:
            # If no tracks, all detections are new tracks
            for i, box in enumerate(det_boxes):
                self.tracks.append(Track(box, det_confs[i]))
            return self.tracks

        if not det_boxes:
            for track in self.tracks:
                track.mark_missed()
            self._cleanup_tracks()
            return self.tracks

        # 2. Association (IoU between PREDICTED track boxes and DETECTED boxes)
        iou_matrix = np.zeros((len(self.tracks), len(det_boxes)))
        for t, track in enumerate(self.tracks):
            for d, det_box in enumerate(det_boxes):
                iou_matrix[t, d] = calculate_iou(track.box, det_box)

        # Hungarian Algorithm
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
