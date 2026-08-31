"""
HUD Visualizer for rendering modern, cyber-styled overlays, bounding boxes,
vehicle counters, and motion trails on video frames.
"""
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from config import VehicleDetectionConfig
from src.detector import DetectedVehicle


class HUDVisualizer:
    """
    Renders bounding boxes, tracking trails, and real-time statistics HUD.
    """

    def __init__(self, config: Optional[VehicleDetectionConfig] = None):
        self.config = config or VehicleDetectionConfig()

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: List[DetectedVehicle],
        counts: Dict[str, int],
        fps: float,
        track_histories: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        is_paused: bool = False,
        device_info: str = "CPU"
    ) -> np.ndarray:
        """
        Draws all visual components onto the frame and returns the annotated frame.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Draw Tracking Trails
        if self.config.show_trails and track_histories:
            self._draw_trails(output, track_histories, detections)

        # 2. Draw Bounding Boxes & Badges
        for det in detections:
            self._draw_detection(output, det)

        # 3. Draw Top Information Dashboard
        if self.config.show_hud:
            self._draw_top_dashboard(output, counts, fps, device_info, w, h)

        # 4. Draw Bottom Control Help Bar
        self._draw_bottom_bar(output, is_paused, w, h)

        return output

    def _draw_detection(self, frame: np.ndarray, det: DetectedVehicle) -> None:
        """Draws bounding box, corner accents, and label badge for a vehicle."""
        x1, y1, x2, y2 = det.bbox
        color = det.color

        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw stylish corner accents
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 3
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # Build label text
        parts = []
        if self.config.show_ids and det.track_id is not None:
            parts.append(f"#{det.track_id}")
        if self.config.show_labels:
            parts.append(det.class_name)
        if self.config.show_conf:
            parts.append(f"{int(det.confidence * 100)}%")

        label = " | ".join(parts)
        if not label:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Label background coordinates
        badge_y1 = max(0, y1 - label_h - 10)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(frame.shape[1], x1 + label_w + 12)

        # Draw filled background badge
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
        
        # High-contrast text (dark text on bright colors)
        text_color = (20, 20, 20) if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 150 else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (badge_x1 + 6, badge_y2 - 6),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    def _draw_trails(
        self,
        frame: np.ndarray,
        histories: Dict[int, List[Tuple[int, int]]],
        detections: List[DetectedVehicle]
    ) -> None:
        """Renders smooth trailing motion lines behind tracked vehicles."""
        id_to_color = {d.track_id: d.color for d in detections if d.track_id is not None}

        for track_id, points in histories.items():
            if len(points) < 2:
                continue
            color = id_to_color.get(track_id, (0, 255, 0))
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    def _draw_top_dashboard(
        self,
        frame: np.ndarray,
        counts: Dict[str, int],
        fps: float,
        device_info: str,
        w: int,
        h: int
    ) -> None:
        """Draws semi-transparent HUD banner at the top of the video."""
        # Top banner dimensions
        banner_h = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 24, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Bottom accent border
        cv2.line(frame, (0, banner_h), (w, banner_h), (50, 180, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Left Telemetry: Model & FPS
        fps_text = f"FPS: {fps:.1f}" if self.config.show_fps else ""
        title_text = f"CITYFLOW AI | {self.config.model_name.upper()} ({device_info})"
        
        cv2.putText(frame, title_text, (16, 22), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        
        # FPS with color threshold
        fps_color = (0, 255, 120) if fps >= 25 else ((0, 215, 255) if fps >= 15 else (0, 80, 255))
        if fps_text:
            cv2.putText(frame, fps_text, (16, 44), font, 0.55, fps_color, 2, cv2.LINE_AA)

        # 2. Right Stats: Vehicle Counts
        if self.config.show_counts:
            total_vehicles = sum(counts.values())
            
            # Start position from right
            x_offset = w - 20
            
            # Draw Total Count badge
            total_str = f"TOTAL: {total_vehicles}"
            (tw, th), _ = cv2.getTextSize(total_str, font, 0.65, 2)
            x_offset -= tw
            cv2.putText(frame, total_str, (x_offset, 34), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            x_offset -= 30

            # Draw breakdown for each class (only if > 0 or standard set)
            for cls_id, cls_name in reversed(list(self.config.class_names.items())):
                cnt = counts.get(cls_name, 0)
                color = self.config.class_colors.get(cls_id, (255, 255, 255))
                badge_str = f"{cls_name}: {cnt}"
                (bw, bh), _ = cv2.getTextSize(badge_str, font, 0.50, 1)
                x_offset -= (bw + 20)
                if x_offset < 320:  # Prevent overlapping left telemetry on narrow frames
                    break
                
                # Tiny colored bullet
                cv2.circle(frame, (x_offset, 30), 4, color, cv2.FILLED)
                cv2.putText(frame, badge_str, (x_offset + 10, 34), font, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

    def _draw_bottom_bar(self, frame: np.ndarray, is_paused: bool, w: int, h: int) -> None:
        """Draws subtle bottom keyboard control guide."""
        bar_h = 28
        y1 = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (w, h), (15, 18, 22), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        track_status = "ON" if self.config.enable_tracking else "OFF"
        controls = f"[Q] Quit   [P] {'RESUME' if is_paused else 'PAUSE'}   [S] Snapshot   [T] Tracking: {track_status}   [C] Toggle Counts"
        
        if is_paused:
            # Add glowing PAUSED indicator
            pause_label = "|| PAUSED ||"
            (pw, _), _ = cv2.getTextSize(pause_label, font, 0.55, 2)
            cv2.putText(frame, pause_label, ((w - pw) // 2, h - 8), font, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, controls, (20, h - 8), font, 0.45, (180, 190, 200), 1, cv2.LINE_AA)
