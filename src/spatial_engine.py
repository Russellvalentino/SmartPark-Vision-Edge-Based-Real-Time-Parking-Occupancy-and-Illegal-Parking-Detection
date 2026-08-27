"""
Spatial Polygon & Occupancy Debounce Engine for SmartPark Vision.
Calculates spatial inclusion and IoU between vehicle bounding boxes and slot polygons.
Implements an N-frame debounce state machine to prevent flicker.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from shapely.geometry import Polygon, box

class ParkingSlot:
    def __init__(self, spot_id: str, name: str, polygon_coords: List[List[int]], debounce_n: int = 5):
        self.spot_id = spot_id
        self.name = name
        self.polygon_coords = np.array(polygon_coords, dtype=np.int32)
        self.shapely_polygon = Polygon(polygon_coords)
        self.debounce_n = debounce_n
        
        # State tracking
        self.status = "vacant"  # "vacant" or "occupied"
        self.candidate_status = "vacant"
        self.candidate_frame_count = 0
        self.vehicle_track_id: Optional[int] = None
        self.candidate_track_id: Optional[int] = None

    def calculate_intersection_ratio(self, bbox: List[float]) -> float:
        """
        Calculates the ratio of intersection area over the slot area or vehicle area.
        bbox: [x1, y1, x2, y2]
        """
        try:
            bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
            if not self.shapely_polygon.intersects(bbox_poly):
                return 0.0
            
            intersection_area = self.shapely_polygon.intersection(bbox_poly).area
            # Use intersection over slot area for reliable occupancy detection
            slot_area = self.shapely_polygon.area
            if slot_area <= 0:
                return 0.0
            return intersection_area / slot_area
        except Exception:
            return 0.0

    def update(self, detected_vehicles: List[Dict[str, Any]], iou_threshold: float = 0.25) -> Optional[Dict[str, Any]]:
        """
        Updates the slot state with current frame vehicle detections.
        detected_vehicles: List of {'track_id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': str}
        Returns state transition event if a debounced state change occurred, otherwise None.
        """
        best_match_id: Optional[int] = None
        max_ratio = 0.0

        for veh in detected_vehicles:
            ratio = self.calculate_intersection_ratio(veh["bbox"])
            if ratio > iou_threshold and ratio > max_ratio:
                max_ratio = ratio
                best_match_id = veh.get("track_id")

        raw_state = "occupied" if best_match_id is not None else "vacant"
        transition_event = None

        if raw_state == self.candidate_status:
            self.candidate_frame_count += 1
            if self.candidate_frame_count >= self.debounce_n and self.status != self.candidate_status:
                old_status = self.status
                self.status = self.candidate_status
                self.vehicle_track_id = self.candidate_track_id if self.status == "occupied" else None
                transition_event = {
                    "spot_id": self.spot_id,
                    "previous_status": old_status,
                    "new_status": self.status,
                    "vehicle_track_id": self.vehicle_track_id
                }
        else:
            self.candidate_status = raw_state
            self.candidate_track_id = best_match_id
            self.candidate_frame_count = 1

        # Keep current track_id refreshed if still occupied
        if self.status == "occupied" and best_match_id is not None:
            self.vehicle_track_id = best_match_id

        return transition_event

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spot_id": self.spot_id,
            "status": self.status,
            "vehicle_track_id": self.vehicle_track_id
        }


class SpatialEngine:
    def __init__(self, config: Dict[str, Any]):
        self.debounce_n = config.get("spatial_logic", {}).get("debounce_frames", 5)
        self.iou_threshold = config.get("spatial_logic", {}).get("occupancy_iou_threshold", 0.25)
        self.slots: Dict[str, ParkingSlot] = {}

    def load_zones_and_slots(self, zones_data: Dict[str, Any]):
        self.slots.clear()
        for s in zones_data.get("parking_slots", []):
            slot = ParkingSlot(
                spot_id=s["spot_id"],
                name=s.get("name", s["spot_id"]),
                polygon_coords=s["polygon"],
                debounce_n=self.debounce_n
            )
            self.slots[s["spot_id"]] = slot

    def process_frame(self, detected_vehicles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process frame detections, update all slots, and return (current_slot_statuses, list_of_state_transitions).
        """
        transitions = []
        statuses = []
        for spot_id, slot in self.slots.items():
            evt = slot.update(detected_vehicles, iou_threshold=self.iou_threshold)
            if evt:
                transitions.append(evt)
            statuses.append(slot.to_dict())

        return statuses, transitions
