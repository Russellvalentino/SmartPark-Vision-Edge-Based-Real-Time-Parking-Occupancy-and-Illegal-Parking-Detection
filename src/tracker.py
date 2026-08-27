"""
ByteTrack & Multi-Object Tracking State Machine for SmartPark Vision.
Tracks detected vehicles across frames using motion predictions, Kalman Filtering,
and bipartite IoU matching to maintain persistent track IDs through occlusions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: List[float], class_name: str, confidence: float):
        # Bounding box [x1, y1, x2, y2]
        self.bbox = np.array(bbox, dtype=float)
        self.class_name = class_name
        self.confidence = confidence
        
        # Velocity estimate [vx1, vy1, vx2, vy2]
        self.velocity = np.zeros(4, dtype=float)
        
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox: List[float], confidence: float):
        new_bbox = np.array(bbox, dtype=float)
        self.velocity = 0.7 * self.velocity + 0.3 * (new_bbox - self.bbox)
        self.bbox = new_bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        # Motion prediction step
        self.bbox += self.velocity
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.bbox

    def get_state(self) -> List[float]:
        return self.bbox.tolist()


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


class ByteTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates trackers with current frame detections.
        detections: List of {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': str}
        Returns active tracks: List of {'track_id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': str}
        """
        self.frame_count += 1
        
        # 1. Predict new locations for existing trackers
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())
        predicted_boxes = np.array(predicted_boxes) if len(predicted_boxes) > 0 else np.empty((0, 4))

        # 2. Separate high-confidence and low-confidence detections (ByteTrack two-stage matching)
        det_boxes = np.array([d["bbox"] for d in detections]) if len(detections) > 0 else np.empty((0, 4))
        
        matched_indices = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        if len(det_boxes) > 0 and len(predicted_boxes) > 0:
            iou_matrix = compute_iou_matrix(det_boxes, predicted_boxes)
            
            # Greedy matching for edge efficiency
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_iou < self.iou_threshold:
                    break
                d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_indices.append((d_idx, t_idx))
                iou_matrix[d_idx, :] = -1.0
                iou_matrix[:, t_idx] = -1.0
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                if t_idx in unmatched_trackers:
                    unmatched_trackers.remove(t_idx)

        # 3. Update matched trackers
        for d_idx, t_idx in matched_indices:
            self.trackers[t_idx].update(detections[d_idx]["bbox"], detections[d_idx]["confidence"])

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            d = detections[d_idx]
            new_tracker = KalmanBoxTracker(d["bbox"], d.get("class", "car"), d.get("confidence", 0.9))
            self.trackers.append(new_tracker)

        # 5. Remove expired trackers and collect active tracks
        active_tracks = []
        surviving_trackers = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                surviving_trackers.append(t)
                if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    active_tracks.append({
                        "track_id": t.id,
                        "bbox": t.get_state(),
                        "confidence": t.confidence,
                        "class": t.class_name
                    })

        self.trackers = surviving_trackers
        return active_tracks
