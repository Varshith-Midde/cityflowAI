"""
Vehicle Detection and Tracking engine utilizing Ultralytics YOLO models.
"""
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from config import VehicleDetectionConfig

logger = logging.getLogger(__name__)


# Distinct color palette for multi-vehicle track identification (BGR)
TRACK_PALETTE: List[Tuple[int, int, int]] = [
    (0, 242, 254),    # Neon Cyan
    (255, 140, 0),    # Dodger Blue
    (56, 239, 125),   # Emerald Green
    (255, 75, 43),    # Fiery Red
    (246, 211, 101),  # Gold / Amber
    (255, 105, 180),  # Hot Pink
    (180, 105, 255),  # Violet
    (0, 255, 255),    # Electric Yellow
    (0, 165, 255),    # Vibrant Orange
    (50, 205, 50),    # Lime Green
]


@dataclass
class DetectedVehicle:
    """Represents a single detected vehicle instance."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    center: Tuple[int, int] = (0, 0)
    color: Tuple[int, int, int] = (0, 255, 0)
    direction: str = "Stationary"
    speed_px: float = 0.0
    lane_pos: str = "Center"


class VehicleDetector:
    """
    Manages YOLO model loading, inference, multi-vehicle detection & tracking,
    direction estimation, counting line crossings, and traffic density metrics.
    """

    def __init__(self, config: Optional[VehicleDetectionConfig] = None):
        self.config = config or VehicleDetectionConfig()
        self.model: Optional[YOLO] = None
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.trail_length)
        )
        
        # Multi-vehicle tracking history & unique track counter
        self.all_seen_track_ids = set()
        self.last_latency_ms: float = 0.0

        # Virtual Counting Line Crossings
        self.inbound_count: int = 0
        self.outbound_count: int = 0
        self._track_prev_positions: Dict[int, Tuple[int, int]] = {}
        self._counted_tracks = set()
        
        # FPS calculation state
        self._prev_timestamp: float = 0.0
        self.current_fps: float = 0.0
        self._fps_smoothing: float = 0.85  # Exponential moving average factor

        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO model on specified device."""
        logger.info(f"Loading YOLO model: {self.config.model_name}...")
        try:
            self.model = YOLO(self.config.model_name)
            # If a specific device is set (e.g., 'cuda:0' or 'cpu'), configure it
            if self.config.device:
                self.model.to(self.config.device)
            logger.info(f"Successfully loaded {self.config.model_name}")
        except Exception as e:
            logger.error(f"Error loading model '{self.config.model_name}': {e}")
            raise

    def change_model(self, model_name: str) -> bool:
        """Dynamically load and switch to a different YOLO model weights."""
        try:
            old_name = self.config.model_name
            self.config.model_name = model_name
            self._load_model()
            self.reset_tracking()
            logger.info(f"Switched model from {old_name} to {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change model to {model_name}: {e}")
            return False

    def reset_tracking(self) -> None:
        """Resets trajectory trails, crossing counts, and unique track ID counter."""
        self.track_history.clear()
        self.all_seen_track_ids.clear()
        self._track_prev_positions.clear()
        self._counted_tracks.clear()
        self.inbound_count = 0
        self.outbound_count = 0

    def get_total_unique_vehicles(self) -> int:
        """Returns total count of distinct vehicle tracking IDs recorded."""
        return len(self.all_seen_track_ids)

    def get_traffic_density(self, num_vehicles: int) -> Tuple[str, Tuple[int, int, int]]:
        """Calculates traffic density level and associated BGR color."""
        if num_vehicles == 0:
            return "NO VEHICLES", (180, 190, 200)
        elif num_vehicles <= 2:
            return "LOW DENSITY (SMOOTH)", (56, 239, 125)
        elif num_vehicles <= 5:
            return "MODERATE TRAFFIC", (0, 215, 255)
        else:
            return "HEAVY CONGESTION", (0, 0, 255)

    def _estimate_direction(self, track_id: int, current_pt: Tuple[int, int]) -> Tuple[str, float]:
        """Calculates movement direction and speed in pixels/frame based on history."""
        pts = self.track_history.get(track_id, [])
        if len(pts) < 3:
            return "Tracking...", 0.0

        # Compare current position to position 3 frames ago
        prev_x, prev_y = pts[-3]
        curr_x, curr_y = current_pt
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        dist = float(np.sqrt(dx**2 + dy**2))

        if dist < 4.0:
            return "Stationary", dist

        if abs(dy) > abs(dx):
            direction = "Southbound ⬇" if dy > 0 else "Northbound ⬆"
        else:
            direction = "Eastbound ➡" if dx > 0 else "Westbound ⬅"

        return direction, dist

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[DetectedVehicle], Dict[str, int], float]:
        """
        Executes multi-vehicle detection and tracking on a single frame.

        Returns:
            Tuple containing:
            - List of DetectedVehicle instances
            - Dictionary of vehicle counts by class
            - Current FPS estimate
        """
        if self.model is None or frame is None:
            return [], {}, 0.0

        start_time = time.perf_counter()
        h, w = frame.shape[:2]
        line_y = int(h * self.config.counting_line_y)

        # Run multi-vehicle detection / tracking
        if self.config.enable_tracking:
            results = self.model.track(
                source=frame,
                persist=True,
                classes=self.config.target_class_ids,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                tracker=self.config.tracker_type,
                verbose=False
            )
        else:
            results = self.model.predict(
                source=frame,
                classes=self.config.target_class_ids,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                verbose=False
            )

        detections: List[DetectedVehicle] = []
        counts: Dict[str, int] = {name: 0 for name in self.config.class_names.values()}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
                classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else boxes.cls.numpy().astype(int)
                
                track_ids = None
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes.id, "cpu") else boxes.id.numpy().astype(int)

                for i, box in enumerate(xyxy):
                    cls_id = int(classes[i])
                    conf = float(confs[i])
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    cls_name = self.config.class_names.get(cls_id, result.names.get(cls_id, f"Class {cls_id}"))
                    
                    track_id = int(track_ids[i]) if track_ids is not None else None

                    # Distinct vehicle color assignment
                    if track_id is not None and self.config.color_by_track_id:
                        color = TRACK_PALETTE[track_id % len(TRACK_PALETTE)]
                    else:
                        color = self.config.class_colors.get(cls_id, self.config.default_color)

                    # Direction and speed calculation
                    direction = "Detected"
                    speed_px = 0.0
                    if track_id is not None:
                        self.track_history[track_id].append((cx, cy))
                        self.all_seen_track_ids.add(track_id)
                        direction, speed_px = self._estimate_direction(track_id, (cx, cy))

                        # Virtual Counting Line Check
                        if self.config.enable_counting_line and track_id in self._track_prev_positions:
                            prev_y = self._track_prev_positions[track_id][1]
                            if track_id not in self._counted_tracks:
                                if prev_y < line_y <= cy:
                                    self.inbound_count += 1
                                    self._counted_tracks.add(track_id)
                                elif prev_y > line_y >= cy:
                                    self.outbound_count += 1
                                    self._counted_tracks.add(track_id)

                        self._track_prev_positions[track_id] = (cx, cy)

                    # Spatial Lane Estimation
                    if cx < w * 0.33:
                        lane_pos = "Left Lane"
                    elif cx < w * 0.66:
                        lane_pos = "Center Lane"
                    else:
                        lane_pos = "Right Lane"

                    # Increment class counters
                    counts[cls_name] = counts.get(cls_name, 0) + 1

                    detection = DetectedVehicle(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        track_id=track_id,
                        center=(cx, cy),
                        color=color,
                        direction=direction,
                        speed_px=speed_px,
                        lane_pos=lane_pos
                    )
                    detections.append(detection)

        # Update FPS & latency calculation
        duration = time.perf_counter() - start_time
        self.last_latency_ms = duration * 1000.0
        instant_fps = 1.0 / duration if duration > 0 else 0.0

        if self._prev_timestamp == 0.0:
            self.current_fps = instant_fps
        else:
            self.current_fps = (self._fps_smoothing * self.current_fps) + ((1.0 - self._fps_smoothing) * instant_fps)
        self._prev_timestamp = start_time

        return detections, counts, self.current_fps

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Returns the list of recent centroid coordinates for a given track ID."""
        return list(self.track_history.get(track_id, []))
