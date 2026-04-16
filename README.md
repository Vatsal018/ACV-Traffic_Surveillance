# Traffic Surveillance System
### YOLOv11 · Centroid Tracker · ROI Line Counter · Industrial Grade Python

---

## Overview

A production-grade, fully OOP Python traffic surveillance system that:

- **Detects** vehicles in real-time using **YOLOv11** (auto-downloaded on first run)
- **Tracks** every vehicle across frames using a centroid-based multi-object tracker
- **Counts** vehicles crossing configurable **Green (IN)** and **Red (OUT)** counting lines
- **Classifies** vehicles into three categories:
  - 🔵 **Two Wheeler** – Bicycle, Motorbike, Scooter, Cycle
  - 🟢 **Passenger Vehicle** – Car, Auto-Rickshaw, Taxi
  - 🔴 **Heavy Vehicle** – Bus, Truck, Ambulance, Police Van
- **Exports** annotated MP4 video + CSV per-track log + JSON summary report
- **Visualises** bounding boxes, trajectory trails, track IDs, and a live dashboard

---

## Project Structure

```
traffic_surveillance/
│
├── main.py                  ← Primary entry-point
├── pipeline.py              ← Orchestration pipeline
├── batch_process.py         ← Batch folder processing
├── setup_roi.py             ← Interactive ROI/line drawing tool
├── requirements.txt
│
├── config/
│   ├── config.yaml          ← Master configuration (edit this!)
│   └── config_loader.py
│
├── core/
│   ├── vehicle.py           ← Vehicle data models + classifier
│   ├── detector.py          ← YOLOv11 inference wrapper
│   ├── tracker.py           ← Centroid tracker + Detection DTO
│   └── roi_counter.py       ← ROI polygon + line crossing counter
│
├── utils/
│   ├── geometry.py          ← IoU, crossing detection, coordinate scaling
│   └── logger.py            ← Coloured + file logging factory
│
├── visualization/
│   ├── overlay.py           ← OpenCV annotation renderer
│   └── reporter.py          ← CSV + JSON report writer
│
├── input/                   ← ← ← DROP YOUR VIDEOS HERE
├── output/                  ← Annotated videos + reports written here
├── logs/                    ← Rotating log files
└── models/                  ← (Optional) place custom YOLO weights here
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** required. GPU (CUDA/MPS) is auto-detected; CPU fallback is automatic.

### 2. Drop your video into `input/`

```
traffic_surveillance/
└── input/
    └── your_traffic_video.mp4   ← drag and drop here
```

### 3. Run

```bash
python main.py
```

That's it. Processed output appears in `output/`.

---

## Usage Options

```bash
# Process all videos in input/ folder (default)
python main.py

# Process a single video
python main.py --video input/my_video.mp4

# Use a custom config
python main.py --config config/my_config.yaml

# Batch process a specific folder
python main.py --folder /path/to/videos

# Batch with summary table
python batch_process.py
```

---

## Interactive ROI Setup

Before processing, you can visually configure the counting lines to match
your camera angle:

```bash
python setup_roi.py --video input/your_video.mp4
```

| Control | Action |
|---------|--------|
| **Left-click** | Set Green (IN) line – click 2 points |
| **Right-click** | Set Red (OUT) line – click 2 points |
| **[P]** | Toggle ROI polygon drawing mode |
| **[R]** | Reset all drawings |
| **[S]** | Save coordinates to config.yaml and exit |
| **[ESC]** | Exit without saving |

---

## Configuration (`config/config.yaml`)

Key settings to customise:

```yaml
model:
  weights: "yolo11n.pt"          # yolo11n / yolo11s / yolo11m / yolo11l / yolo11x
  confidence_threshold: 0.40
  device: "auto"                 # "cpu", "cuda", "mps", or "auto"

roi:
  green_line:                    # IN count line (normalised 0-1 coords)
    start: [0.0, 0.45]
    end:   [1.0, 0.45]
  red_line:                      # OUT count line
    start: [0.0, 0.55]
    end:   [1.0, 0.55]

tracker:
  max_disappeared: 30            # Frames before a track is dropped
  max_distance: 80               # Max centroid shift between frames (pixels)

output:
  save_video: true
  save_csv_report: true
  save_summary_json: true
```

---

## Output Files

For each processed video `input/my_video.mp4`, the system creates:

| File | Description |
|------|-------------|
| `output/my_video_annotated.mp4` | Full annotated video with overlays |
| `output/my_video_report.csv` | Per-track log (ID, class, IN/OUT, confidence) |
| `output/my_video_summary.json` | Aggregate count summary with metadata |
| `logs/traffic_surveillance_YYYYMMDD.log` | Rotating log file |

---

## Architecture

```
Video Frame
    │
    ▼
YOLODetector (YOLOv11)
    │  List[Detection]
    ▼
VehicleTracker (Centroid Matching)
    │  Dict[track_id, Vehicle]
    ▼
ROICounter (Line Crossing Detection)
    │  Updates CountStore
    ▼
OverlayRenderer (OpenCV Drawing)
    │  Annotated Frame
    ▼
VideoWriter ──► output/*.mp4
Reporter    ──► output/*.csv + *.json
```

---

## Vehicle Classification

| Category | COCO Classes | Colour |
|----------|-------------|--------|
| Two Wheeler | bicycle (1), motorcycle (3) | 🔵 Cyan |
| Passenger Vehicle | car (2) | 🟢 Green |
| Heavy Vehicle | bus (5), train (6), truck (7) | 🔴 Red |

---

## YOLO Model Sizes

| Model | Speed | Accuracy | File Size |
|-------|-------|----------|-----------|
| `yolo11n.pt` | ⚡ Fastest | Good | ~6 MB |
| `yolo11s.pt` | Fast | Better | ~20 MB |
| `yolo11m.pt` | Medium | Best for CPU | ~40 MB |
| `yolo11l.pt` | Slow | Excellent | ~50 MB |
| `yolo11x.pt` | Slowest | Maximum | ~110 MB |

Change `model.weights` in `config/config.yaml`.

---

## Requirements

```
ultralytics >= 8.3.0   (YOLOv11)
opencv-python >= 4.9.0
numpy >= 1.26.0
scipy >= 1.12.0
pandas >= 2.2.0
matplotlib >= 3.8.0
PyYAML >= 6.0.1
colorlog >= 6.8.0
tqdm >= 4.66.0
Pillow >= 10.2.0
```

---

## License

For educational and research purposes.
