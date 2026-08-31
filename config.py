"""
Configuration settings for the Real-Time Vehicle Detection and Tracking System.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class VehicleDetectionConfig:
    # Model configuration
    model_name: str = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolo11n.pt, etc.
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = ""  # '' = auto (CUDA if available else CPU), 'cpu', 'cuda:0'
    imgsz: int = 640

    # COCO Class IDs for vehicles:
    # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    target_class_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])

    # Class mappings with human-readable names
    class_names: Dict[int, str] = field(default_factory=lambda: {
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    })

    # Color scheme (BGR format)
    class_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        1: (0, 215, 255),    # Gold / Yellow for Bicycle
        2: (255, 140, 0),    # Cyan / Dodger Blue for Car
        3: (180, 105, 255),  # Hot Pink / Purple for Motorcycle
        5: (0, 165, 255),    # Orange for Bus
        7: (0, 0, 255)       # Red for Truck
    })

    # Fallback default color
    default_color: Tuple[int, int, int] = (0, 255, 0)

    # Video stream configuration
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps_target: int = 30

    # Tracking configuration
    enable_tracking: bool = True
    tracker_type: str = "bytetrack.yaml"  # bytetrack.yaml or botsort.yaml

    # UI / HUD display options
    show_hud: bool = True
    show_fps: bool = True
    show_counts: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_ids: bool = True
    show_trails: bool = True
    trail_length: int = 25

    # Multi-Vehicle and Traffic Analytics Configuration
    enable_counting_line: bool = False
    counting_line_y: float = 0.50
    show_direction: bool = True
    show_density: bool = True
    color_by_track_id: bool = True

    # Snapshot and Recording output directory
    output_dir: str = "outputs"
    snapshots_dir: str = "outputs/snapshots"
    recordings_dir: str = "outputs/recordings"
    exports_dir: str = "outputs/exports"
