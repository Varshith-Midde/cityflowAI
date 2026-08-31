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


class VehicleDetector:
    """
    Manages YOLO model loading, inference, filtering by vehicle classes,
    multi-object tracking (ByteTrack), and FPS benchmarking.
    """

    def __init__(self, config: Optional[VehicleDetectionConfig] = None):
        self.config = config or VehicleDetectionConfig()
        self.model: Optional[YOLO] = None
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.trail_length)
        )
        
        # Tracking history & unique track counter
        self.all_seen_track_ids = set()
        self.last_latency_ms: float = 0.0
        
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
        """Resets trajectory trails and unique track ID counter."""
        self.track_history.clear()
        self.all_seen_track_ids.clear()

    def get_total_unique_vehicles(self) -> int:
        """Returns total count of distinct vehicle tracking IDs recorded."""
        return len(self.all_seen_track_ids)

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[DetectedVehicle], Dict[str, int], float]:
        """
        Executes detection or tracking on a single frame.

        Returns:
            Tuple containing:
            - List of DetectedVehicle instances
            - Dictionary of vehicle counts by class
            - Current FPS estimate
        """
        if self.model is None or frame is None:
            return [], {}, 0.0

        start_time = time.perf_counter()

        # Run detection / tracking
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
                    color = self.config.class_colors.get(cls_id, self.config.default_color)

                    track_id = int(track_ids[i]) if track_ids is not None else None

                    # Record trajectory trail point & unique tracking ID
                    if track_id is not None:
                        self.track_history[track_id].append((cx, cy))
                        self.all_seen_track_ids.add(track_id)

                    # Increment vehicle counts
                    counts[cls_name] = counts.get(cls_name, 0) + 1

                    detection = DetectedVehicle(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        track_id=track_id,
                        center=(cx, cy),
                        color=color
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
