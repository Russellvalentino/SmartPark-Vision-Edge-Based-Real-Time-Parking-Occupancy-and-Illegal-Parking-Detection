"""
SmartPark Vision - Edge AI Pipeline Execution Entry Point.

Usage:
  python run_smartpark.py --generate-dataset
  python run_smartpark.py --source data/datasets/sample_parking_feed.mp4 --record
  python run_smartpark.py --calibrate data/datasets/sample_parking_feed.mp4
  python run_smartpark.py --source rtsp://your_camera_ip:554/stream
"""

import argparse
import sys
import os

from src.dataset_generator import create_parking_lot_dataset
from src.pipeline import SmartParkPipeline

def main():
    parser = argparse.ArgumentParser(description="SmartPark Vision - Edge-Based Parking & Violation Detection")
    parser.add_argument("--source", type=str, default="data/datasets/sample_parking_feed.mp4", help="Path to video file, RTSP URL, or webcam index")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate synthetic parking lot test dataset and JSON metadata")
    parser.add_argument("--calibrate", action="store_true", help="Launch interactive polygon calibration UI")
    parser.add_argument("--display", action="store_true", help="Show live visual HUD window")
    parser.add_argument("--record", action="store_true", help="Save annotated output video to data/annotated_output.mp4")
    parser.add_argument("--output", type=str, default="data/annotated_output.mp4", help="Output path for recorded video")
    parser.add_argument("--config", type=str, default="config/settings.json", help="System configuration path")
    parser.add_argument("--zones", type=str, default="config/zones_and_slots.json", help="Zones and slots definition path")

    args = parser.parse_args()

    if args.generate_dataset or not os.path.exists(args.source):
        print("\n[SmartPark Vision] Generating synthetic parking dataset...")
        create_parking_lot_dataset()

    if args.calibrate:
        from calibration_tool import PolygonCalibrator
        print(f"\n[SmartPark Vision] Launching Polygon Calibration Tool on: {args.source}")
        calibrator = PolygonCalibrator(args.source, output_config_path=args.zones)
        calibrator.run()
        return

    # Run Edge Pipeline
    pipeline = SmartParkPipeline(config_path=args.config, zones_path=args.zones)
    output_path = args.output if args.record else None
    
    print(f"\n[SmartPark Vision] Starting Edge Processing on source: {args.source}")
    pipeline.run(source_path_or_rtsp=args.source, output_display=args.display, output_video_path=output_path)

if __name__ == "__main__":
    main()
