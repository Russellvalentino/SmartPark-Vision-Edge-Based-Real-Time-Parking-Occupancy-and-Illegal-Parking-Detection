"""
Interactive Polygon Calibration Module for SmartPark Vision.
Allows defining, labeling, and saving N-sided polygons for parking slots (P1..Pn)
and restricted violation zones (Z1..Zm) to config/zones_and_slots.json.
"""

import sys
import json
import cv2
import numpy as np

class PolygonCalibrator:
    def __init__(self, image_or_video_path: str, output_config_path: str = "config/zones_and_slots.json"):
        self.output_config_path = output_config_path
        self.current_points = []
        self.parking_slots = []
        self.restricted_zones = []
        self.current_mode = "slot"  # "slot" or "zone"
        self.window_name = "SmartPark Vision - Polygon Calibrator (Press 'h' for help)"

        # Load existing config if available
        try:
            with open(output_config_path, "r") as f:
                data = json.load(f)
                self.parking_slots = data.get("parking_slots", [])
                self.restricted_zones = data.get("restricted_zones", [])
        except Exception:
            pass

        # Load first frame
        cap = cv2.VideoCapture(image_or_video_path)
        ret, self.base_frame = cap.read()
        cap.release()

        if not ret or self.base_frame is None:
            self.base_frame = np.full((1080, 1920, 3), 45, dtype=np.uint8)

        self.height, self.width = self.base_frame.shape[:2]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([int(x), int(y)])
            print(f"[Calibrator] Added point: ({x}, {y})")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== SmartPark Vision Polygon Calibrator ===")
        print(" [Left Click] : Add polygon vertex")
        print(" [Enter/Space]: Finish current polygon & save spot/zone")
        print(" [c]          : Clear in-progress polygon points")
        print(" [z]          : Switch mode (Current: PARKING SLOT vs RESTRICTED ZONE)")
        print(" [s]          : Save calibration to JSON")
        print(" [r]          : Reset all polygons")
        print(" [q/Esc]      : Quit\n")

        while True:
            display = self.base_frame.copy()

            # Draw existing parking slots
            for s in self.parking_slots:
                pts = np.array(s["polygon"], np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(display, s["spot_id"], (center[0] - 15, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw existing restricted zones
            for z in self.restricted_zones:
                pts = np.array(z["polygon"], np.int32)
                cv2.polylines(display, [pts], True, (0, 0, 255), 2)
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(display, z["zone_id"], (center[0] - 30, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw current points in progress
            if len(self.current_points) > 0:
                pts = np.array(self.current_points, np.int32)
                color = (0, 255, 255) if self.current_mode == "slot" else (0, 165, 255)
                for p in self.current_points:
                    cv2.circle(display, tuple(p), 5, color, -1)
                if len(self.current_points) > 1:
                    cv2.polylines(display, [pts], False, color, 2)

            # Instructions HUD
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (20, 20, 20), -1)
            hud = f"MODE: {self.current_mode.upper()} | Slots: {len(self.parking_slots)} | Zones: {len(self.restricted_zones)} | Current Points: {len(self.current_points)} | [z] Toggle Mode | [Space] Finalize | [s] Save"
            cv2.putText(display, hud, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in [27, ord('q')]: # Quit
                break
            elif key == ord('z'): # Toggle mode
                self.current_mode = "zone" if self.current_mode == "slot" else "slot"
                print(f"[Calibrator] Switched mode to: {self.current_mode.upper()}")
            elif key in [13, 32]: # Enter or Space
                if len(self.current_points) >= 3:
                    if self.current_mode == "slot":
                        spot_num = len(self.parking_slots) + 1
                        spot_id = f"A{spot_num}"
                        self.parking_slots.append({
                            "spot_id": spot_id,
                            "name": f"Bay {spot_id}",
                            "polygon": list(self.current_points)
                        })
                        print(f"[Calibrator] Saved Parking Bay: {spot_id}")
                    else:
                        zone_num = len(self.restricted_zones) + 1
                        zone_id = f"zone_{zone_num}"
                        self.restricted_zones.append({
                            "zone_id": zone_id,
                            "name": f"Restricted Zone {zone_num}",
                            "type": "fire_lane",
                            "dwell_time_threshold_seconds": 60,
                            "polygon": list(self.current_points)
                        })
                        print(f"[Calibrator] Saved Restricted Zone: {zone_id}")
                    self.current_points = []
                    self.save_json()
                else:
                    print("[Calibrator] Error: Polygon requires at least 3 points.")
            elif key == ord('c'):
                self.current_points = []
                print("[Calibrator] Cleared in-progress points.")
            elif key == ord('s'):
                self.save_json()
            elif key == ord('r'):
                self.parking_slots = []
                self.restricted_zones = []
                self.current_points = []
                print("[Calibrator] Reset all slots and zones.")

        cv2.destroyAllWindows()

    def save_json(self):
        data = {
            "camera_id": "cam_zone_A",
            "resolution": [self.width, self.height],
            "parking_slots": self.parking_slots,
            "restricted_zones": self.restricted_zones
        }
        with open(self.output_config_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Calibrator] Saved calibration to {self.output_config_path}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/datasets/sample_parking_feed.mp4"
    calibrator = PolygonCalibrator(src)
    calibrator.run()
