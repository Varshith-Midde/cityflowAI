# CityFlow AI - Real-Time Vehicle Detection & Tracking System 🚗 🏍️ 🚌 🚚

A high-performance, modular Python application for real-time vehicle detection, classification, and multi-object tracking using state-of-the-art YOLO models (YOLOv8 / YOLO11) with laptop camera and video stream support.

---

## ✨ Features

- **Real-Time Detection & Multi-Object Tracking**: Detects and tracks vehicles across frames with persistent IDs and trajectory trails (powered by ByteTrack).
- **Vehicle Filtering**: Targeted filtering for Cars, Motorcycles, Buses, Trucks, and Bicycles.
- **Modern Cyber HUD**: Real-time FPS counter, hardware monitor, and per-class vehicle count breakdown banner.
- **Optimized for Laptop Cameras**: Built-in Windows DirectShow (`CAP_DSHOW`) optimization for instant webcam connection and ultra-low latency.
- **Interactive Controls**: Pause/resume feeds, take high-resolution snapshots, and toggle tracking/counters dynamically on the fly.
- **Flexible Input Sources**: Works with built-in laptop webcams, USB cameras, local video files (`.mp4`, `.avi`), and RTSP streams.
- **Video Recording**: Ability to record and save annotated video output.

---

## 📁 Project Structure

```
cityflow AI/
├── config.py             # Centralized configuration (classes, colors, thresholds)
├── requirements.txt      # Dependencies (ultralytics, opencv, torch)
├── main.py               # Main CLI application runner
├── test_system.py        # Automated diagnostics & verification script
├── README.md             # Documentation & usage guide
├── outputs/              # Saved snapshots & recordings
└── src/
    ├── __init__.py
    ├── camera.py         # Camera & video stream capture engine
    ├── detector.py       # YOLO inference & vehicle tracking pipeline
    └── visualizer.py     # Modern HUD dashboard & bounding box visualizer
```

---

## 🚀 Quick Start & Launch Modes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Modern Tkinter Desktop UI (Recommended for PC Testing) 💻
Launch the rich graphical desktop interface with real-time video viewport, video scrubber, sidebar telemetry, class filters, instant snapshots, and one-click video recording:
```bash
python ui_app.py
```

### 3. Launch CLI / OpenCV Window Mode (Direct Laptop Camera) 📹
```bash
python main.py
```

### 4. Run System Diagnostics & Verification Suite
Run the automated test suite to check model loading, synthetic frame inference, and camera accessibility:
```bash
python test_system.py
```

---

## 🎮 Interactive Keyboard Controls

While the camera detection window is focused, you can use the following keyboard shortcuts:

| Key | Action |
|---|---|
| `Q` or `ESC` | Exit application cleanly |
| `P` or `SPACE` | Pause / Resume video feed |
| `S` | Save a high-resolution snapshot to `outputs/` |
| `T` | Toggle vehicle tracking & trajectory trails ON/OFF |
| `C` | Toggle vehicle counter breakdown overlay ON/OFF |
| `H` | Toggle the entire top HUD dashboard banner |

---

## 🛠️ CLI Options & Usage Examples

### Run on a video file:
```bash
python main.py --source "path/to/traffic_video.mp4"
```

### Use a higher accuracy YOLO model (e.g. YOLOv8 Small or Medium):
```bash
python main.py --model yolov8s.pt
```

### Adjust confidence threshold:
```bash
python main.py --conf 0.50
```

### Record annotated video to file:
```bash
python main.py --save "outputs/recorded_feed.mp4"
```

### Force CPU execution:
```bash
python main.py --device cpu
```

---

## ⚙️ Configuration

You can customize vehicle classes, colors, resolution, and tracking settings in [config.py](file:///c:/Users/varsh/Desktop/cityflow%20AI/config.py):
- `target_class_ids`: Select which COCO classes to detect.
- `class_colors`: Custom BGR colors for each vehicle type.
- `trail_length`: Length of motion history trails.
- `fps_target` & `frame_width`/`frame_height`: Default camera capture properties.
