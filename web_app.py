"""
SmartPark Vision - Web Application Dashboard Server.
Streams real-time YOLOv11 + ByteTrack AI detections and live parking telemetry in your browser.
"""

import os
import json
import time
import sqlite3
import cv2
from flask import Flask, render_template, Response, jsonify

from src.pipeline import SmartParkPipeline

app = Flask(__name__, template_folder="templates")

# Global pipeline instance
pipeline = SmartParkPipeline()
video_source = "data/datasets/sample_parking_feed.mp4"
latest_telemetry = {
    "fps": 30.0,
    "cpu_percent": 15,
    "memory_percent": 45,
    "total_spots": 4,
    "occupied_spots": 2,
    "spots": [
        {"spot_id": "A1", "status": "occupied", "vehicle_track_id": 2},
        {"spot_id": "A2", "status": "vacant", "vehicle_track_id": None},
        {"spot_id": "A3", "status": "occupied", "vehicle_track_id": 1},
        {"spot_id": "A4", "status": "vacant", "vehicle_track_id": None}
    ],
    "alerts": []
}

def generate_frames():
    global latest_telemetry
    while True:
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        start_wall_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            simulated_timestamp = start_wall_time + (frame_idx / fps)

            # AI processing
            health_stats = pipeline.health_monitor.update_frame(frame)
            raw_detections = pipeline.detector.detect(frame)
            active_tracks = pipeline.tracker.update(raw_detections)
            spot_statuses, transitions = pipeline.spatial_engine.process_frame(active_tracks)
            violations = pipeline.violation_engine.process_frame(active_tracks, simulated_timestamp, frame)

            for v in violations:
                pipeline.telemetry.emit_violation(v)
                latest_telemetry["alerts"].insert(0, v)
                latest_telemetry["alerts"] = latest_telemetry["alerts"][:5]

            # Update latest telemetry cache
            occupied_count = sum(1 for s in spot_statuses if s.get("status") == "occupied")
            latest_telemetry.update({
                "fps": health_stats["fps"],
                "cpu_percent": health_stats["cpu_percent"],
                "memory_percent": health_stats["memory_percent"],
                "total_spots": len(spot_statuses),
                "occupied_spots": occupied_count,
                "spots": spot_statuses
            })

            # Render HUD overlay
            annotated_frame = pipeline._render_hud(frame, spot_statuses, active_tracks, violations, health_stats)
            
            # Encode frame to JPEG
            ret_encode, buffer = cv2.imencode('.jpg', cv2.resize(annotated_frame, (1280, 720)))
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1.0 / fps)

        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    return jsonify(latest_telemetry)

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" 🚀 SMARTPARK VISION WEB DASHBOARD RUNNING")
    print(" Open your browser at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
