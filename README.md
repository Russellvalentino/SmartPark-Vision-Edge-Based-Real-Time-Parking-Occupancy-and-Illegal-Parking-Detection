# SmartPark Vision (Edge-Native AI Pipeline)

Edge-Based Real-Time Parking Occupancy & Violation Detection using **YOLOv11**, **ByteTrack**, and **Spatial Polygon Analytics**.

---

## 📁 Project Structure (Strictly JSON & SQLite - No CSV)

```
ML/
├── config/
│   ├── settings.json              # Edge node & AI thresholds configuration
│   └── zones_and_slots.json       # N-sided polygon calibrations (P1..Pn, Z1..Zm)
├── data/
│   ├── datasets/                  # Video datasets and JSON benchmark metadata
│   │   ├── parking_scene_metadata.json
│   │   └── sample_parking_feed.mp4
│   ├── snapshots/                 # Low-res cropped evidence snapshots on violation
│   ├── annotated_output.mp4       # Rendered video stream with live HUD overlay
│   └── events.db                  # Local SQLite offline buffer & event log
├── src/
│   ├── detector.py                # YOLOv11 vehicle object detector
│   ├── tracker.py                 # ByteTrack Kalman filter tracker & track ID state machine
│   ├── spatial_engine.py          # Polygon IoU & N-frame debounce state machine
│   ├── violation_engine.py        # Restricted zone dwell-time counter & alert engine
│   ├── telemetry.py               # MQTT telemetry dispatcher & SQLite buffer
│   ├── health_monitor.py          # FPS, system health & camera obstruction detection
│   ├── dataset_generator.py       # Parking benchmark generator (JSON + MP4)
│   └── pipeline.py                # Main edge processing pipeline
├── calibration_tool.py            # Interactive visual polygon calibration UI
├── run_smartpark.py               # CLI entry point
└── requirements.txt               # Dependencies
```

---

## 📊 Datasets

While this repository uses a synthetic sample MP4 for quickstart testing, for robust evaluation we recommend these public datasets:
- **PKLot Dataset:** [Kaggle Link](https://www.kaggle.com/code/blatalia/pklot/notebook)
- **CNRPark-EXT Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/ddsshubham/cnrpark-ext)

*Note:* Due to GitHub's strict file size limits, do not upload raw gigabyte-scale image sequences to this repository. Use them locally for testing or generating your own MP4 feeds.

---

## 🚀 Quickstart & Commands

### 1. Run the Pipeline on Benchmark Dataset
```bash
.venv\Scripts\python.exe run_smartpark.py --source data/datasets/sample_parking_feed.mp4 --record
```

### 2. Generate a New Benchmark Dataset
```bash
.venv\Scripts\python.exe run_smartpark.py --generate-dataset
```

### 3. Launch Interactive Polygon Calibration Tool
```bash
.venv\Scripts\python.exe calibration_tool.py data/datasets/sample_parking_feed.mp4
```
* **Left Click**: Add polygon vertex
* **Space / Enter**: Save polygon bay / zone
* **`z`**: Toggle between **Parking Slot** mode and **Restricted Zone** mode
* **`s`**: Save calibration to `config/zones_and_slots.json`

### 4. Connect to a Live RTSP Camera Feed
```bash
.venv\Scripts\python.exe run_smartpark.py --source "rtsp://username:password@camera_ip:554/live" --display
```

---

## 📡 Telemetry Payloads (PRD Section 6)

### Occupancy Telemetry (`smartpark/edge/{device_id}/occupancy`)
```json
{
  "timestamp": "2026-08-27T18:27:31Z",
  "device_id": "edge_node_north_garage_01",
  "camera_id": "cam_zone_A",
  "total_spots": 4,
  "occupied_spots": 3,
  "spots": [
    { "spot_id": "A1", "status": "occupied", "vehicle_track_id": 2 },
    { "spot_id": "A2", "status": "occupied", "vehicle_track_id": 3 },
    { "spot_id": "A3", "status": "occupied", "vehicle_track_id": 1 },
    { "spot_id": "A4", "status": "vacant", "vehicle_track_id": null }
  ]
}
```

### Illegal Parking Alert (`smartpark/edge/{device_id}/alerts`)
```json
{
  "timestamp": "2026-08-27T18:27:07Z",
  "event_id": "evt_viol_64153a",
  "violation_type": "illegal_parking_dwell_exceeded",
  "zone_id": "fire_lane_01",
  "vehicle_details": {
    "track_id": 4,
    "class": "car",
    "confidence": 0.94,
    "dwell_time_seconds": 5,
    "threshold_seconds": 5
  },
  "evidence_snapshot_url": "data/snapshots/20260827_235707_evt_viol_64153a_track_4.jpg"
}
```
