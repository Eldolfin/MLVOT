import os
import sys
from typing import Any, List

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

# Add tp1 to path to import KalmanFilter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tp1")))

try:
    from KalmanFilter import KalmanFilter
except ImportError:
    # Fallback if running from a different context or tp1 not found
    print("Warning: Could not import KalmanFilter from tp1. Make sure it exists.")

    class KalmanFilter:  # type: ignore
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


class FeatureExtractor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ReID model not found at {model_path}")
        self.net = cv.dnn.readNetFromONNX(model_path)
        self.roi_width = 64
        self.roi_height = 128
        # Standard mean/std for ReID models trained on ImageNet/Market1501
        self.roi_means = np.array([0.485, 0.456, 0.406], dtype="float32")
        self.roi_stds = np.array([0.229, 0.224, 0.225], dtype="float32")

    def preprocess(self, img_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess image crop for ReID model.
        """
        if img_crop.size == 0:
            return np.zeros((1, 3, self.roi_height, self.roi_width), dtype=np.float32)

        # Resize
        roi_input = cv.resize(img_crop, (self.roi_width, self.roi_height))
        # Convert BGR to RGB
        roi_input = cv.cvtColor(roi_input, cv.COLOR_BGR2RGB)
        # Normalize
        roi_input = roi_input.astype(np.float32) / 255.0
        roi_input = (roi_input - self.roi_means) / self.roi_stds
        # Transpose to (C, H, W)
        roi_input = np.transpose(roi_input, (2, 0, 1))
        # Add batch dimension
        roi_input = np.expand_dims(roi_input, axis=0)
        return roi_input

    def extract(self, img_crop: np.ndarray) -> np.ndarray:
        """
        Extract features from image crop.
        """
        blob = self.preprocess(img_crop)
        self.net.setInput(blob)
        feature = self.net.forward()
        # Normalize feature vector
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        return feature.flatten()


class Track:
    _id_counter = 1

    def __init__(self, box: List[float], feature: np.ndarray, info: float = 0.0):
        self.id = Track._id_counter
        Track._id_counter += 1
        self.box = box  # [x, y, w, h]
        self.feature = feature  # ReID feature vector
        self.missed_frames = 0
        self.info = info
        self.color = tuple(np.random.randint(0, 255, 3).tolist())

        self.kf: Any = KalmanFilter(
            dt=1.0,
            u_x=0,
            u_y=0,
            std_acc=1,
            x_dt_meas=0.1,
            y_dt_meas=0.1,
        )

        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        setattr(self.kf, "state", np.array([cx, cy, 0, 0]))

        self.w = box[2]
        self.h = box[3]

    def predict(self):
        self.kf.predict()
        state = getattr(self.kf, "state")
        pred_cx, pred_cy = state[0], state[1]
        self.box = [pred_cx - self.w / 2, pred_cy - self.h / 2, self.w, self.h]

    def update(self, box: List[float], feature: np.ndarray, info: float = 0.0):
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        z = np.array([[cx], [cy]])

        self.kf.update(z)

        # Update feature with moving average (optional but good practice)
        alpha = 0.9
        self.feature = alpha * self.feature + (1 - alpha) * feature
        norm = np.linalg.norm(self.feature)
        if norm > 0:
            self.feature /= norm

        self.w = box[2]
        self.h = box[3]

        state = getattr(self.kf, "state")
        updated_cx, updated_cy = state[0], state[1]

        self.box = [updated_cx - self.w / 2, updated_cy - self.h / 2, self.w, self.h]
        self.missed_frames = 0
        self.info = info

    def mark_missed(self):
        self.missed_frames += 1


class Tracker:
    def __init__(
        self,
        reid_model_path: str,
        max_missed_frames: int = 5,
        iou_threshold: float = 0.3,
        alpha: float = 0.5,  # Weight for IoU
        beta: float = 0.5,  # Weight for Appearance
    ):
        self.tracks: List[Track] = []
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.alpha = alpha
        self.beta = beta
        self.reid = FeatureExtractor(reid_model_path)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Vectors are already normalized
        return np.dot(vec1, vec2)

    def update(
        self, detections: List[List[float]], frame_img: np.ndarray
    ) -> List[Track]:
        """
        Update tracks.
        detections: List of [id, x, y, w, h, conf]
        frame_img: Current video frame (BGR)
        """
        det_boxes = [d[1:5] for d in detections]
        det_confs = [d[5] for d in detections]

        # Extract features for all detections
        det_features = []
        for box in det_boxes:
            x, y, w, h = map(int, box)
            # Clip to image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame_img.shape[1] - x)
            h = min(h, frame_img.shape[0] - y)

            if w <= 0 or h <= 0:
                det_features.append(np.zeros(512))  # Assuming 512 dim, using 0s
                continue

            crop = frame_img[y : y + h, x : x + w]
            feat = self.reid.extract(crop)
            det_features.append(feat)

        # 1. Predict
        for track in self.tracks:
            track.predict()

        if not self.tracks:
            for i, box in enumerate(det_boxes):
                if i < len(det_features):
                    self.tracks.append(Track(box, det_features[i], det_confs[i]))
            return self.tracks

        if not det_boxes:
            for track in self.tracks:
                track.mark_missed()
            self._cleanup_tracks()
            return self.tracks

        # 2. Association (IoU + ReID)
        cost_matrix = np.zeros((len(self.tracks), len(det_boxes)))

        for t, track in enumerate(self.tracks):
            for d, det_box in enumerate(det_boxes):
                iou = calculate_iou(track.box, det_box)

                sim = self._cosine_similarity(track.feature, det_features[d])
                # Normalized Similarity: Cosine similarity is [-1, 1], usually [0, 1] for ReID if ReLU used.
                # If vectors are non-negative, dot product is [0, 1].
                # Let's assume standard cosine similarity.
                # Subject says: "normalized Similarity is obtained by normalizing the feature similarity score"
                # "For cosine similarity, this could be directly used"

                # Combined score
                score = self.alpha * iou + self.beta * sim

                cost_matrix[t, d] = score

        # Hungarian Algorithm (maximize score)
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)

        assigned_tracks = set()
        assigned_detections = set()

        for r, c in zip(row_indices, col_indices):
            # Thresholding on combined score? Or just IoU?
            # Typically verify IoU > thresh to avoid spatially impossible matches even if appearance is similar
            iou = calculate_iou(self.tracks[r].box, det_boxes[c])

            # Using combined threshold or IoU threshold?
            # Let's stick to a basic consistency check.
            if iou >= 0.1:  # Relaxed IoU threshold because ReID helps
                self.tracks[r].update(det_boxes[c], det_features[c], det_confs[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        for t, track in enumerate(self.tracks):
            if t not in assigned_tracks:
                track.mark_missed()

        for d, det_box in enumerate(det_boxes):
            if d not in assigned_detections:
                self.tracks.append(Track(det_box, det_features[d], det_confs[d]))

        self._cleanup_tracks()
        return self.tracks

    def _cleanup_tracks(self):
        self.tracks = [
            t for t in self.tracks if t.missed_frames <= self.max_missed_frames
        ]
