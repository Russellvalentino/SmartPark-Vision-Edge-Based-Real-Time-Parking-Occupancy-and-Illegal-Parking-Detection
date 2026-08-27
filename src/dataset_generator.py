"""
Dataset Generator & Simulation Feed for SmartPark Vision.
Generates realistic parking lot test footage and ground-truth metadata in JSON format (strictly NO CSV).
Includes parking bays, moving vehicles, bay occupancy transitions, and restricted fire lane dwell violations.
"""

import os
import json
import math
import numpy as np
import cv2

def create_parking_lot_dataset(
    output_video_path: str = "data/datasets/sample_parking_feed.mp4",
    output_metadata_path: str = "data/datasets/parking_scene_metadata.json",
    num_frames: int = 360,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30
):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Base background (asphalt pavement)
    base_bg = np.full((height, width, 3), (45, 48, 52), dtype=np.uint8)

    # Road driving lane top
    cv2.rectangle(base_bg, (0, 200), (width, 550), (55, 58, 62), -1)
    
    # Yellow dashed divider line
    for x in range(0, width, 80):
        cv2.line(base_bg, (x, 380), (x + 40, 380), (0, 215, 255), 4)

    # Define Parking Bays (A1, A2, A3, A4)
    slots = [
        {"id": "A1", "poly": np.array([[150, 600], [450, 600], [400, 950], [80, 950]], np.int32)},
        {"id": "A2", "poly": np.array([[480, 600], [780, 600], [740, 950], [420, 950]], np.int32)},
        {"id": "A3", "poly": np.array([[810, 600], [1110, 600], [1080, 950], [760, 950]], np.int32)},
        {"id": "A4", "poly": np.array([[1140, 600], [1440, 600], [1420, 950], [1100, 950]], np.int32)}
    ]

    # Draw Slot outlines and markings on base background
    for s in slots:
        cv2.polylines(base_bg, [s["poly"]], True, (240, 240, 240), 4)
        center = np.mean(s["poly"], axis=0).astype(int)
        cv2.putText(base_bg, s["id"], (center[0] - 25, center[1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 200), 2)

    # Define Restricted Fire Lane Zone (Z1)
    fire_lane_poly = np.array([[1480, 450], [1850, 450], [1850, 950], [1480, 950]], np.int32)
    # Red hatched markings for fire lane
    for y in range(460, 940, 40):
        cv2.line(base_bg, (1490, y), (1840, y), (40, 40, 200), 3)
    cv2.polylines(base_bg, [fire_lane_poly], True, (0, 0, 255), 5)
    cv2.putText(base_bg, "NO PARKING - FIRE LANE", (1500, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    dataset_metadata = {
        "dataset_name": "SmartPark_Synthetic_Benchmark_v1",
        "description": "Ground truth parking lot CCTV dataset with vehicle movement, bay occupancy transitions, and fire lane dwell violation.",
        "video_file": output_video_path,
        "resolution": [width, height],
        "fps": fps,
        "total_frames": num_frames,
        "ground_truth_events": [
            {
                "frame": 0,
                "event": "initial_state",
                "occupied_spots": ["A1", "A3"],
                "vacant_spots": ["A2", "A4"]
            },
            {
                "frame": 60,
                "event": "vehicle_entry_bay_A2",
                "description": "Car 102 parks in Bay A2"
            },
            {
                "frame": 120,
                "event": "fire_lane_unauthorized_stop",
                "description": "Vehicle 87 enters Fire Lane and begins continuous dwell"
            }
        ]
    }

    # Helper function to render a realistic vehicle sprite
    def draw_vehicle(canvas, center_x, center_y, width_v, height_v, angle_deg, color, text="CAR"):
        rect = ((center_x, center_y), (width_v, height_v), angle_deg)
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        
        # Shadow
        shadow_rect = ((center_x + 10, center_y + 10), (width_v + 10, height_v + 10), angle_deg)
        shadow_pts = cv2.boxPoints(shadow_rect).astype(np.int32)
        cv2.fillPoly(canvas, [shadow_pts], (25, 25, 25))

        # Car Body
        cv2.fillPoly(canvas, [box_pts], color)
        cv2.polylines(canvas, [box_pts], True, (20, 20, 20), 2)
        
        # Windshield / Roof
        roof_rect = ((center_x, center_y), (width_v * 0.65, height_v * 0.5), angle_deg)
        roof_pts = cv2.boxPoints(roof_rect).astype(np.int32)
        cv2.fillPoly(canvas, [roof_pts], (30, 30, 30))

        # Headlights / Taillights
        cv2.putText(canvas, text, (int(center_x - 20), int(center_y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"[Dataset] Generating synthetic benchmark video ({num_frames} frames)...")

    for f in range(num_frames):
        frame = base_bg.copy()

        # 1. Parked Car in Bay A1 (Static)
        draw_vehicle(frame, 270, 780, 160, 260, 10, (180, 80, 50), "A1-CAR")

        # 2. Parked Car in Bay A3 (Static)
        draw_vehicle(frame, 930, 780, 160, 260, 8, (60, 140, 200), "A3-CAR")

        # 3. Car 102 driving into Bay A2 (frames 20 to 120)
        if f < 30:
            car_x = 200 + f * 12
            car_y = 350
            draw_vehicle(frame, car_x, car_y, 250, 140, 0, (40, 180, 90), "V-102")
        elif f < 100:
            prog = (f - 30) / 70.0
            car_x = 560 + prog * 40
            car_y = 350 + prog * 420
            angle = prog * 85
            w_cur = 250 - prog * 90
            h_cur = 140 + prog * 120
            draw_vehicle(frame, car_x, car_y, w_cur, h_cur, angle, (40, 180, 90), "V-102")
        else:
            # Parked in A2
            draw_vehicle(frame, 600, 770, 160, 260, 8, (40, 180, 90), "A2-CAR")

        # 4. Car 87 entering Fire Lane and stopping (Dwell violation)
        if f < 60:
            # Not entered yet or driving on road
            fx = 100 + f * 20
            draw_vehicle(frame, fx, 320, 260, 140, 0, (30, 30, 190), "V-87")
        elif f < 120:
            # Turning into Fire Lane
            prog = (f - 60) / 60.0
            fx = 1300 + prog * 360
            fy = 320 + prog * 380
            angle = prog * 90
            w_cur = 260 - prog * 100
            h_cur = 140 + prog * 120
            draw_vehicle(frame, fx, fy, w_cur, h_cur, angle, (30, 30, 190), "V-87")
        else:
            # Continuously parked in Fire Lane (triggers dwell time alert)
            draw_vehicle(frame, 1660, 700, 160, 260, 90, (30, 30, 190), "V-87")

        # Add subtle noise/lighting variation
        noise = np.random.normal(0, 1.2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()

    with open(output_metadata_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print(f"[Dataset] Generated {output_video_path} and {output_metadata_path}")
    return output_video_path, output_metadata_path

if __name__ == "__main__":
    create_parking_lot_dataset()
