# SmartPark-Vision

Edge-deployed computer vision system for real-time parking occupancy detection
and illegal-parking violation alerts, with a live web dashboard.

## Features

- YOLOv8/ONNX-based vehicle detection optimized for edge devices
- Stationary-vehicle tracking (ByteTrack/SORT) for dwell-time analysis
- Polygon-based occupancy analysis for legal parking bays and no-parking zones
- Timer-based illegal parking violation detection with snapshot capture
- FastAPI backend with WebSocket telemetry streaming
- React + Tailwind live dashboard (occupancy grid, violation alerts, stats)

## Project Structure

See the repository tree for a full breakdown of `src/edge_vision`,
`src/backend`, `src/utils`, `tools/`, and `web_dashboard/`.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-org>/SmartPark-Vision.git
cd SmartPark-Vision

# 2. Python environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# On edge hardware with GPU acceleration:
pip install -r requirements-edge.txt

# 3. Mark parking zones (one-time setup per camera)
python tools/mark_roi.py --video data/sample_demo.mp4 --output config/parking_slots.json

# 4. Run the edge vision pipeline
python -m src.edge_vision.detector --config config/config.yaml

# 5. Run the backend API + WebSocket server
uvicorn src.backend.main:app --reload --port 8000

# 6. Run the dashboard
cd web_dashboard
npm install
npm run dev
```

## Configuration

- `config/config.yaml` — detection thresholds, inference interval, video source(s)
- `config/parking_slots.json` — polygon coordinates for legal parking bays
- `config/no_parking_zones.json` — polygon coordinates for restricted zones

## Testing & CI

GitHub Actions (`.github/workflows/python-lint-test.yml`) runs Flake8 and
Pytest on every push/PR.

```bash
pip install flake8 pytest
flake8 src/
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
