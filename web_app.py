"""
SmartPark Vision - Web Application Dashboard Server.
Streams real-time YOLOv11 + ByteTrack AI detections and live parking telemetry in your browser.
"""

import os
import json
import time
import sqlite3
import threading
import cv2
from flask import Flask, render_template, Response, jsonify

from src.pipeline import SmartParkPipeline

app = Flask(__name__, template_folder="templates")

video_source = "data/datasets/sample_parking_feed.mp4"
latest_frame = None

def background_pipeline():
    """Runs the AI pipeline continuously in a background thread."""
    global latest_frame
    print("[Web App] Starting background AI pipeline...")
    pipeline = SmartParkPipeline()
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_wall_time = time.time()
    frame_idx = 0

    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_source)
            
        ret, frame = cap.read()
        if not ret:
            # End of video, loop it
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        simulated_timestamp = start_wall_time + (frame_idx / fps)
        current_time = time.time()

        # AI processing
        health_stats = pipeline.health_monitor.update_frame(frame)
        raw_detections = pipeline.detector.detect(frame)
        active_tracks = pipeline.tracker.update(raw_detections)
        spot_statuses, transitions = pipeline.spatial_engine.process_frame(active_tracks)
        violations = pipeline.violation_engine.process_frame(active_tracks, simulated_timestamp, frame)

        # Emit telemetry to SQLite/MQTT
        if current_time - pipeline.last_occupancy_emit >= pipeline.occupancy_emit_interval or len(transitions) > 0:
            pipeline.telemetry.emit_occupancy(spot_statuses)
            pipeline.last_occupancy_emit = current_time

        for v in violations:
            pipeline.telemetry.emit_violation(v)

        # Render HUD overlay
        annotated_frame = pipeline._render_hud(frame, spot_statuses, active_tracks, violations, health_stats)
        
        # Encode frame to JPEG
        ret_encode, buffer = cv2.imencode('.jpg', cv2.resize(annotated_frame, (1280, 720)))
        if ret_encode:
            latest_frame = buffer.tobytes()

        # Match processing speed roughly to video FPS to avoid running away
        time.sleep(1.0 / fps)

# Start the background thread
threading.Thread(target=background_pipeline, daemon=True).start()

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        # Stream at roughly 30 FPS
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    db_path = "data/events.db"
    response_data = {
        "fps": 0, "cpu_percent": 0, "memory_percent": 0,
        "total_spots": 0, "occupied_spots": 0, "spots": [],
        "alerts": []
    }
    
    if not os.path.exists(db_path):
        return jsonify(response_data)
        
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get latest occupancy
            cursor.execute("SELECT payload FROM event_buffer WHERE topic LIKE '%occupancy' ORDER BY id DESC LIMIT 1")
            occ_row = cursor.fetchone()
            if occ_row:
                occ_data = json.loads(occ_row[0])
                response_data["total_spots"] = occ_data.get("total_spots", 0)
                response_data["occupied_spots"] = occ_data.get("occupied_spots", 0)
                response_data["spots"] = occ_data.get("spots", [])
            
            # Get latest alerts (last 5)
            cursor.execute("SELECT payload FROM event_buffer WHERE topic LIKE '%alerts' ORDER BY id DESC LIMIT 5")
            alerts_rows = cursor.fetchall()
            for row in alerts_rows:
                response_data["alerts"].append(json.loads(row[0]))
                
    except Exception as e:
        print(f"[Web App] Error reading telemetry from DB: {e}")
        
    return jsonify(response_data)

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" 🚀 SMARTPARK VISION WEB DASHBOARD RUNNING")
    print(" Open your browser at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
