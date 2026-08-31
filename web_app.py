"""
SmartPark Vision - Web Application Dashboard Server.
Streams real-time YOLOv11 + ByteTrack AI detections, provides dynamic in-browser polygon calibration,
and live system configuration management.
"""

import os
import json
import time
import sqlite3
import threading
import base64
import cv2
from flask import Flask, render_template, Response, jsonify, request

from src.pipeline import SmartParkPipeline

app = Flask(__name__, template_folder="templates")

video_source = "data/datasets/sample_parking_feed.mp4"
latest_frame = None
latest_raw_frame = None
pipeline_instance = None
pipeline_lock = threading.Lock()

def background_pipeline():
    """Runs the AI pipeline continuously in a background thread."""
    global latest_frame, latest_raw_frame, pipeline_instance
    print("[Web App] Starting background AI pipeline...")
    with pipeline_lock:
        pipeline_instance = SmartParkPipeline()
        
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

        # Cache raw frame for in-browser canvas calibration
        latest_raw_frame = frame.copy()

        with pipeline_lock:
            # AI processing
            health_stats = pipeline_instance.health_monitor.update_frame(frame)
            raw_detections = pipeline_instance.detector.detect(frame)
            active_tracks = pipeline_instance.tracker.update(raw_detections)
            spot_statuses, transitions = pipeline_instance.spatial_engine.process_frame(active_tracks)
            violations = pipeline_instance.violation_engine.process_frame(active_tracks, simulated_timestamp, frame)

            # Emit telemetry to SQLite/MQTT
            if current_time - pipeline_instance.last_occupancy_emit >= pipeline_instance.occupancy_emit_interval or len(transitions) > 0:
                pipeline_instance.telemetry.emit_occupancy(spot_statuses)
                pipeline_instance.last_occupancy_emit = current_time

            for v in violations:
                pipeline_instance.telemetry.emit_violation(v)

            # Render HUD overlay
            annotated_frame = pipeline_instance._render_hud(frame, spot_statuses, active_tracks, violations, health_stats)
        
        # Encode frame to JPEG
        ret_encode, buffer = cv2.imencode('.jpg', cv2.resize(annotated_frame, (1280, 720)))
        if ret_encode:
            latest_frame = buffer.tobytes()

        # Match processing speed roughly to video FPS
        time.sleep(1.0 / fps)

# Start the background thread
threading.Thread(target=background_pipeline, daemon=True).start()

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
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
        "fps": 30.0, "cpu_percent": 0, "memory_percent": 0,
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

@app.route('/api/raw_frame')
def api_raw_frame():
    """Returns a snapshot of the current raw camera frame for in-browser canvas calibration."""
    global latest_raw_frame
    if latest_raw_frame is None:
        frame = cv2.imread("data/snapshots/20260827_235707_evt_viol_64153a_track_4.jpg")
    else:
        frame = latest_raw_frame

    if frame is None:
        return jsonify({"error": "No frame available"}), 404

    _, buffer = cv2.imencode('.jpg', frame)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image_data": f"data:image/jpeg;base64,{b64_str}", "width": frame.shape[1], "height": frame.shape[0]})

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    config_path = "config/settings.json"
    if request.method == 'GET':
        with open(config_path, 'r') as f:
            return jsonify(json.load(f))
    
    # POST: Update settings
    try:
        new_config = request.json
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)

        # Hot-reload in running pipeline
        global pipeline_instance
        with pipeline_lock:
            if pipeline_instance:
                pipeline_instance.config = new_config
                pipeline_instance.detector.confidence_thresh = new_config.get("ai_pipeline", {}).get("confidence_threshold", 0.35)
                pipeline_instance.detector.iou_thresh = new_config.get("ai_pipeline", {}).get("iou_threshold", 0.45)
                pipeline_instance.spatial_engine.debounce_n = new_config.get("spatial_logic", {}).get("debounce_frames", 5)
                pipeline_instance.spatial_engine.iou_threshold = new_config.get("spatial_logic", {}).get("occupancy_iou_threshold", 0.25)
                pipeline_instance.violation_engine.default_dwell_threshold = new_config.get("spatial_logic", {}).get("dwell_time_threshold_seconds", 60.0)

        return jsonify({"status": "success", "message": "Configuration updated and applied to pipeline."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/zones', methods=['GET', 'POST'])
def api_zones():
    zones_path = "config/zones_and_slots.json"
    if request.method == 'GET':
        with open(zones_path, 'r') as f:
            return jsonify(json.load(f))

    # POST: Update zones and slots
    try:
        new_zones = request.json
        with open(zones_path, 'w') as f:
            json.dump(new_zones, f, indent=2)

        # Hot-reload in running pipeline
        global pipeline_instance
        with pipeline_lock:
            if pipeline_instance:
                pipeline_instance.zones_data = new_zones
                pipeline_instance.spatial_engine.load_zones_and_slots(new_zones)
                pipeline_instance.violation_engine.load_zones(new_zones)

        return jsonify({"status": "success", "message": "Zones & parking slots updated and reloaded in pipeline."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" >>> SMARTPARK VISION WEB DASHBOARD RUNNING")
    print(" Open your browser at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
