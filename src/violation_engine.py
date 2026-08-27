"""
Illegal Parking & Restricted Zone Violation Detection Engine for SmartPark Vision.
Tracks vehicle dwell time in designated zones (e.g., Fire Lanes, Loading Docks, No-Standing).
Generates evidence snapshots and alert event payloads when dwell time exceeds T_max.
"""

import os
import time
import uuid
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from shapely.geometry import Polygon, box, Point

class RestrictedZone:
    def __init__(self, zone_id: str, name: str, zone_type: str, polygon_coords: List[List[int]], dwell_time_threshold_seconds: float = 60.0):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type
        self.polygon_coords = np.array(polygon_coords, dtype=np.int32)
        self.shapely_polygon = Polygon(polygon_coords)
        self.threshold_seconds = dwell_time_threshold_seconds
        
        # Tracking per track_id: {track_id: {"first_seen_time": float, "last_seen_time": float, "alert_triggered": bool, "last_bbox": list, "class": str, "confidence": float}}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def is_vehicle_in_zone(self, bbox: List[float]) -> bool:
        """
        Determines if vehicle is inside the restricted zone using centroid and bbox intersection.
        """
        try:
            # Check bottom-center of bounding box (wheel base / contact point)
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3] # Base of car
            point = Point(cx, cy)
            if self.shapely_polygon.contains(point):
                return True
            
            # Fallback: check significant intersection
            bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
            if self.shapely_polygon.intersects(bbox_poly):
                intersection_ratio = self.shapely_polygon.intersection(bbox_poly).area / bbox_poly.area
                return intersection_ratio > 0.35
            return False
        except Exception:
            return False

    def update(self, detected_vehicles: List[Dict[str, Any]], current_time: float, frame: np.ndarray, snapshots_dir: str) -> List[Dict[str, Any]]:
        """
        Updates zone dwell times for detected vehicles.
        Returns list of new violation events.
        """
        current_frame_track_ids = set()
        violations = []

        for veh in detected_vehicles:
            track_id = veh.get("track_id")
            if track_id is None:
                continue

            bbox = veh["bbox"]
            if self.is_vehicle_in_zone(bbox):
                current_frame_track_ids.add(track_id)
                
                if track_id not in self.active_tracks:
                    # Vehicle newly entered zone
                    self.active_tracks[track_id] = {
                        "first_seen_time": current_time,
                        "last_seen_time": current_time,
                        "alert_triggered": False,
                        "last_bbox": bbox,
                        "class": veh.get("class", "car"),
                        "confidence": veh.get("confidence", 0.90)
                    }
                else:
                    entry = self.active_tracks[track_id]
                    entry["last_seen_time"] = current_time
                    entry["last_bbox"] = bbox
                    entry["confidence"] = veh.get("confidence", entry["confidence"])
                    
                    dwell_time = current_time - entry["first_seen_time"]
                    if dwell_time >= self.threshold_seconds and not entry["alert_triggered"]:
                        entry["alert_triggered"] = True
                        
                        # Generate Evidence Snapshot
                        event_id = f"evt_viol_{uuid.uuid4().hex[:6]}"
                        snapshot_rel_path = self._save_evidence_snapshot(
                            frame=frame,
                            bbox=bbox,
                            track_id=track_id,
                            event_id=event_id,
                            snapshots_dir=snapshots_dir
                        )

                        violation_payload = {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
                            "event_id": event_id,
                            "violation_type": "illegal_parking_dwell_exceeded",
                            "zone_id": self.zone_id,
                            "vehicle_details": {
                                "track_id": track_id,
                                "class": entry["class"],
                                "confidence": round(float(entry["confidence"]), 2),
                                "dwell_time_seconds": int(dwell_time),
                                "threshold_seconds": int(self.threshold_seconds)
                            },
                            "evidence_snapshot_url": snapshot_rel_path
                        }
                        violations.append(violation_payload)

        # Clean up vehicles that have left the zone for > 3.0 seconds
        disappeared_tracks = []
        for track_id, entry in self.active_tracks.items():
            if track_id not in current_frame_track_ids:
                if current_time - entry["last_seen_time"] > 3.0:
                    disappeared_tracks.append(track_id)

        for track_id in disappeared_tracks:
            del self.active_tracks[track_id]

        return violations

    def _save_evidence_snapshot(self, frame: np.ndarray, bbox: List[float], track_id: int, event_id: str, snapshots_dir: str) -> str:
        os.makedirs(snapshots_dir, exist_ok=True)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add slight margin around bounding box for context
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        
        annotated_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        date_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{date_str}_{event_id}_track_{track_id}.jpg"
        filepath = os.path.join(snapshots_dir, filename)
        
        cv2.imwrite(filepath, annotated_crop)
        return filepath.replace("\\", "/")


class ViolationEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.snapshots_dir = config.get("storage", {}).get("snapshots_dir", "data/snapshots")
        self.default_dwell_threshold = config.get("spatial_logic", {}).get("dwell_time_threshold_seconds", 60.0)
        self.zones: Dict[str, RestrictedZone] = {}

    def load_zones(self, zones_data: Dict[str, Any]):
        self.zones.clear()
        for z in zones_data.get("restricted_zones", []):
            threshold = z.get("dwell_time_threshold_seconds", self.default_dwell_threshold)
            zone = RestrictedZone(
                zone_id=z["zone_id"],
                name=z.get("name", z["zone_id"]),
                zone_type=z.get("type", "restricted"),
                polygon_coords=z["polygon"],
                dwell_time_threshold_seconds=threshold
            )
            self.zones[z["zone_id"]] = zone

    def process_frame(self, detected_vehicles: List[Dict[str, Any]], current_time: float, frame: np.ndarray) -> List[Dict[str, Any]]:
        all_violations = []
        for zone in self.zones.values():
            zone_violations = zone.update(detected_vehicles, current_time, frame, self.snapshots_dir)
            all_violations.extend(zone_violations)
        return all_violations
