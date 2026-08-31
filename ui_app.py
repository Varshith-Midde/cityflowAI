"""
CityFlow AI - Real-Time Vehicle Detection & Traffic Analytics Desktop UI
Built with Tkinter, OpenCV, Pillow, and Ultralytics YOLO.
"""
import os
import sys
import time
import queue
import csv
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Set console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import VehicleDetectionConfig
from src.camera import VideoStream
from src.detector import VehicleDetector, DetectedVehicle
from src.visualizer import HUDVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CityFlowUI")


# ==============================================================================
# UI STYLING CONSTANTS (Obsidian Cyber Theme)
# ==============================================================================
BG_DARK = "#0d1117"        # Main window dark background
BG_PANEL = "#161b22"       # Card / container background
BG_PANEL_ALT = "#1f242c"   # Slightly lighter panel
BG_INPUT = "#21262d"       # Input fields / dropdowns
BORDER_COLOR = "#30363d"   # Clean subtle border
ACCENT_CYAN = "#00f2fe"    # Primary highlight cyan
ACCENT_BLUE = "#4facfe"    # Secondary blue
ACCENT_GREEN = "#38ef7d"   # Success / online green
ACCENT_YELLOW = "#f6d365"  # Warning / paused amber
ACCENT_RED = "#ff4b2b"     # Danger / record red
TEXT_LIGHT = "#f0f6fc"     # Primary text
TEXT_MUTED = "#8b949e"     # Secondary text / labels

FONT_FAMILY = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"


class VideoWorker(threading.Thread):
    """
    Background worker thread for high-FPS video capture, YOLO inference,
    multi-object tracking, and recording.
    """

    def __init__(
        self,
        source: str,
        config: VehicleDetectionConfig,
        detector: VehicleDetector,
        visualizer: HUDVisualizer,
        frame_queue: queue.Queue,
        event_queue: queue.Queue,
        loop_video: bool = True
    ):
        super().__init__(daemon=True)
        self.source = source
        self.config = config
        self.detector = detector
        self.visualizer = visualizer
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.loop_video = loop_video

        self._stop_event = threading.Event()
        self._is_paused = False
        self._seek_req = None

        # Video recording state
        self._record_writer: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[str] = None
        self._record_start_time: float = 0.0
        self.is_recording = False

        # Snapshot request flag
        self._snapshot_req = False
        self._snapshot_path = ""

        # Hardware info
        if self.config.device.lower().startswith("cuda") or (not self.config.device and torch.cuda.is_available()):
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
            self.device_info = f"GPU: {gpu_name.split()[0]}"
        else:
            self.device_info = "CPU"

    def run(self):
        logger.info(f"Starting VideoWorker for source: {self.source}")
        
        # Check if source is a static image
        if isinstance(self.source, str) and self.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            self._run_image_mode()
            return

        # Initialize stream
        source_val = int(self.source) if str(self.source).isdigit() else self.source
        stream = VideoStream(
            source=source_val,
            width=self.config.frame_width,
            height=self.config.frame_height,
            fps=self.config.fps_target
        )

        if not stream.is_opened:
            logger.error(f"Failed to open stream for source: {self.source}")
            self.event_queue.put(("error", f"Could not open source: {self.source}"))
            return

        total_frames = stream.get_total_frames()
        is_live = stream.is_webcam

        self.event_queue.put(("stream_started", {
            "source": self.source,
            "is_live": is_live,
            "total_frames": total_frames,
            "fps": stream.get_fps(),
            "res": stream.get_resolution()
        }))

        frame_idx = 0
        last_frame_annotated = None
        last_frame_raw = None

        while not self._stop_event.is_set():
            # Handle seek request
            if self._seek_req is not None and not is_live:
                target_frame = int(self._seek_req * total_frames)
                stream.set_frame_index(target_frame)
                self._seek_req = None

            # Handle Pause
            if self._is_paused:
                if last_frame_raw is not None:
                    time.sleep(0.03)
                    self._push_frame(last_frame_annotated, last_frame_raw, [], {}, 0.0, 0.0, frame_idx, total_frames, is_paused=True)
                else:
                    time.sleep(0.05)
                continue

            ret, frame = stream.read()
            if not ret or frame is None:
                if not is_live and self.loop_video and total_frames > 0:
                    stream.restart()
                    continue
                elif is_live:
                    time.sleep(0.05)
                    continue
                else:
                    logger.info("End of video file reached.")
                    self.event_queue.put(("stream_ended", None))
                    break

            frame_idx = stream.get_current_frame_index()
            last_frame_raw = frame.copy()

            # Execute YOLO detection & tracking
            detections, counts, fps = self.detector.process_frame(frame)
            latency = self.detector.last_latency_ms

            # Compute Traffic Density
            density_label, density_color = self.detector.get_traffic_density(len(detections))

            # Gather track trajectories
            track_histories = {}
            if self.config.enable_tracking:
                for det in detections:
                    if det.track_id is not None:
                        track_histories[det.track_id] = self.detector.get_track_history(det.track_id)

            # Draw modern visualizer overlay
            annotated_frame = self.visualizer.draw_hud(
                frame=frame,
                detections=detections,
                counts=counts,
                fps=fps,
                track_histories=track_histories,
                is_paused=False,
                device_info=self.device_info,
                inbound_count=self.detector.inbound_count,
                outbound_count=self.detector.outbound_count,
                density_label=density_label,
                density_color=density_color
            )
            last_frame_annotated = annotated_frame

            # Handle snapshot request
            if self._snapshot_req:
                self._save_snapshot(annotated_frame)
                self._snapshot_req = False

            # Handle active video recording
            if self.is_recording and self._record_writer is not None:
                self._record_writer.write(annotated_frame)

            # Push frame to UI queue
            self._push_frame(
                annotated_frame=annotated_frame,
                raw_frame=frame,
                detections=detections,
                counts=counts,
                fps=fps,
                latency=latency,
                frame_idx=frame_idx,
                total_frames=total_frames,
                is_paused=False,
                density_label=density_label,
                density_color=density_color,
                inbound_count=self.detector.inbound_count,
                outbound_count=self.detector.outbound_count
            )

            # Sleep slightly to match native FPS on video playback if not live
            if not is_live:
                time.sleep(0.015)

        # Cleanup
        stream.release()
        self._stop_recording()
        logger.info("VideoWorker terminated cleanly.")

    def _run_image_mode(self):
        """Processes a static image source."""
        img = cv2.imread(self.source)
        if img is None:
            self.event_queue.put(("error", f"Could not read image: {self.source}"))
            return

        self.event_queue.put(("stream_started", {
            "source": self.source,
            "is_live": False,
            "total_frames": 1,
            "fps": 0,
            "res": (img.shape[1], img.shape[0])
        }))

        detections, counts, fps = self.detector.process_frame(img)
        annotated = self.visualizer.draw_hud(
            frame=img,
            detections=detections,
            counts=counts,
            fps=fps,
            device_info=self.device_info
        )

        while not self._stop_event.is_set():
            if self._snapshot_req:
                self._save_snapshot(annotated)
                self._snapshot_req = False

            self._push_frame(
                annotated_frame=annotated,
                raw_frame=img,
                detections=detections,
                counts=counts,
                fps=fps,
                latency=self.detector.last_latency_ms,
                frame_idx=1,
                total_frames=1,
                is_paused=False
            )
            time.sleep(0.05)

    def _push_frame(
        self, annotated_frame, raw_frame, detections, counts, fps, latency, frame_idx, total_frames, is_paused,
        density_label: str = "SMOOTH", density_color: Tuple[int, int, int] = (56, 239, 125), inbound_count: int = 0, outbound_count: int = 0
    ):
        """Pushes data into the frame queue, dropping older frames if queue is full."""
        data = {
            "annotated": annotated_frame,
            "raw": raw_frame,
            "detections": detections,
            "counts": counts,
            "fps": fps,
            "latency": latency,
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "is_paused": is_paused,
            "unique_tracks": self.detector.get_total_unique_vehicles(),
            "density_label": density_label,
            "density_color": density_color,
            "inbound_count": inbound_count,
            "outbound_count": outbound_count
        }
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(data)
        except Exception:
            pass

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def toggle_pause(self):
        self._is_paused = not self._is_paused
        return self._is_paused

    def seek(self, progress: float):
        self._seek_req = max(0.0, min(1.0, progress))

    def request_snapshot(self, path: str):
        self._snapshot_path = path
        self._snapshot_req = True

    def _save_snapshot(self, frame: np.ndarray):
        try:
            os.makedirs(os.path.dirname(self._snapshot_path), exist_ok=True)
            cv2.imwrite(self._snapshot_path, frame)
            logger.info(f"Snapshot saved: {self._snapshot_path}")
            self.event_queue.put(("snapshot_saved", self._snapshot_path))
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
            self.event_queue.put(("error", f"Snapshot failed: {e}"))

    def start_recording(self, path: str, frame_size: Tuple[int, int], fps: float = 30.0):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._record_writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
            self._record_path = path
            self._record_start_time = time.time()
            self.is_recording = True
            logger.info(f"Started recording to: {path}")
            self.event_queue.put(("recording_started", path))
        except Exception as e:
            logger.error(f"Recording start failed: {e}")
            self.event_queue.put(("error", f"Could not start recording: {e}"))

    def stop_recording(self):
        self._stop_recording()

    def _stop_recording(self):
        if self._record_writer is not None:
            self._record_writer.release()
            self._record_writer = None
            duration = time.time() - self._record_start_time
            logger.info(f"Recording stopped. Saved to {self._record_path} ({duration:.1f}s)")
            self.event_queue.put(("recording_stopped", {"path": self._record_path, "duration": duration}))
        self.is_recording = False

    def stop(self):
        self._stop_event.set()


# ==============================================================================
# MAIN TKINTER GUI APPLICATION
# ==============================================================================
class CityFlowApp(tk.Tk):
    """
    Main Tkinter Desktop UI for CityFlow AI Vehicle Detection System.
    """

    def __init__(self):
        super().__init__()
        self.title("CityFlow AI - Real-Time Vehicle Detection & Tracking Suite")
        self.geometry("1400x880")
        self.minsize(1100, 720)
        self.configure(bg=BG_DARK)

        # Core Engines & Configurations
        self.config = VehicleDetectionConfig()
        self.detector = VehicleDetector(self.config)
        self.visualizer = HUDVisualizer(self.config)

        # Thread Queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.event_queue = queue.Queue()
        self.worker: Optional[VideoWorker] = None

        # State Variables
        self.current_source_type = tk.StringVar(value="sample")
        self.camera_idx_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="outputs/sample_traffic.mp4")
        self.image_path_var = tk.StringVar(value="outputs/sample_bus.jpg")
        self.rtsp_url_var = tk.StringVar(value="rtsp://192.168.1.100:554/stream")
        
        self.model_name_var = tk.StringVar(value="yolov8n.pt")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.tracking_var = tk.BooleanVar(value=True)
        self.trails_var = tk.BooleanVar(value=True)
        self.hud_var = tk.BooleanVar(value=True)
        self.loop_var = tk.BooleanVar(value=True)
        self.counting_line_var = tk.BooleanVar(value=False)
        self.direction_var = tk.BooleanVar(value=True)
        self.color_by_track_var = tk.BooleanVar(value=True)

        # Vehicle class filter variables
        self.class_filters = {
            "Car": tk.BooleanVar(value=True),
            "Motorcycle": tk.BooleanVar(value=True),
            "Bus": tk.BooleanVar(value=True),
            "Truck": tk.BooleanVar(value=True),
            "Bicycle": tk.BooleanVar(value=True),
        }

        # Telemetry storage
        self.last_frame_raw: Optional[np.ndarray] = None
        self.last_frame_annotated: Optional[np.ndarray] = None
        self.current_counts: Dict[str, int] = {}
        self.session_events: List[Dict] = []
        self.is_video_scrubbing = False

        # Build UI Components
        self._setup_styles()
        self._build_header()
        self._build_body()
        self._build_status_bar()

        # Start Event Polling & Animation Loop
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(20, self._process_frame_queue)
        self.after(50, self._process_event_queue)

        # Automatically start laptop camera (Index 0) on startup
        self.after(400, self._auto_start_camera)

        logger.info("CityFlow AI Tkinter UI initialized successfully.")

    def _setup_styles(self):
        """Configure ttk styles for dark modern cyber appearance."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Notebook tabs
        style.configure(
            "TNotebook",
            background=BG_PANEL,
            borderwidth=0
        )
        style.configure(
            "TNotebook.Tab",
            background=BG_INPUT,
            foreground=TEXT_MUTED,
            font=(FONT_FAMILY, 10, "bold"),
            padding=[14, 8],
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", BG_PANEL_ALT)],
            foreground=[("selected", ACCENT_CYAN)]
        )

        # Scale slider
        style.configure(
            "Horizontal.TScale",
            background=BG_PANEL,
            troughcolor=BG_INPUT,
            sliderrelief="flat"
        )

        # Combobox
        style.configure(
            "TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_INPUT,
            foreground=TEXT_LIGHT,
            arrowcolor=ACCENT_CYAN,
            bordercolor=BORDER_COLOR
        )

    # --------------------------------------------------------------------------
    # HEADER BAR
    # --------------------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self, bg=BG_PANEL_ALT, height=64, highlightthickness=1, highlightbackground=BORDER_COLOR)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        # Left: Logo and Brand
        brand_frame = tk.Frame(header, bg=BG_PANEL_ALT)
        brand_frame.pack(side=tk.LEFT, padx=18, pady=8)

        logo_lbl = tk.Label(
            brand_frame,
            text="🚘 CITYFLOW AI",
            font=(FONT_FAMILY, 15, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_PANEL_ALT
        )
        logo_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            brand_frame,
            text="Real-Time Vehicle Detection & Tracking Suite",
            font=(FONT_FAMILY, 9),
            fg=TEXT_MUTED,
            bg=BG_PANEL_ALT
        )
        sub_lbl.pack(anchor="w")

        # Center: Telemetry Badges
        center_frame = tk.Frame(header, bg=BG_PANEL_ALT)
        center_frame.pack(side=tk.LEFT, expand=True, padx=20)

        self.status_pill = tk.Label(
            center_frame,
            text="● IDLE",
            font=(FONT_FAMILY, 10, "bold"),
            fg=TEXT_MUTED,
            bg=BG_INPUT,
            padx=12,
            pady=4,
            relief="flat"
        )
        self.status_pill.pack(side=tk.LEFT, padx=6)

        # Device badge
        dev_text = "⚡ GPU (CUDA)" if torch.cuda.is_available() else "⚡ CPU"
        dev_pill = tk.Label(
            center_frame,
            text=dev_text,
            font=(FONT_FAMILY, 9, "bold"),
            fg=ACCENT_BLUE,
            bg=BG_INPUT,
            padx=10,
            pady=4
        )
        dev_pill.pack(side=tk.LEFT, padx=6)

        # FPS Badge
        self.fps_badge = tk.Label(
            center_frame,
            text="0.0 FPS",
            font=(FONT_FAMILY, 10, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_INPUT,
            padx=10,
            pady=4
        )
        self.fps_badge.pack(side=tk.LEFT, padx=6)

        # Latency Badge
        self.latency_badge = tk.Label(
            center_frame,
            text="0.0 ms",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_INPUT,
            padx=10,
            pady=4
        )
        self.latency_badge.pack(side=tk.LEFT, padx=6)

        # Right: Action Buttons (Snapshot, Record, Open Outputs)
        right_frame = tk.Frame(header, bg=BG_PANEL_ALT)
        right_frame.pack(side=tk.RIGHT, padx=18, pady=10)

        self.btn_snapshot = tk.Button(
            right_frame,
            text="📸 Snapshot",
            font=(FONT_FAMILY, 9, "bold"),
            bg="#238636",
            fg="white",
            activebackground="#2ea043",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_take_snapshot
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=4)

        self.btn_record = tk.Button(
            right_frame,
            text="🎥 Record",
            font=(FONT_FAMILY, 9, "bold"),
            bg="#b62324",
            fg="white",
            activebackground="#da3633",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_toggle_record
        )
        self.btn_record.pack(side=tk.LEFT, padx=4)

        btn_folder = tk.Button(
            right_frame,
            text="📂 Outputs",
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            activebackground=BORDER_COLOR,
            activeforeground=TEXT_LIGHT,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_open_outputs
        )
        btn_folder.pack(side=tk.LEFT, padx=4)

    # --------------------------------------------------------------------------
    # MAIN BODY (SPLIT PANES: VIEWPORT & SIDEBAR)
    # --------------------------------------------------------------------------
    def _build_body(self):
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left: Video Viewport & Playback Controls
        left_pane = tk.Frame(body, bg=BG_PANEL, highlightthickness=1, highlightbackground=BORDER_COLOR)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self._build_viewport(left_pane)

        # Right: Sidebar (Analytics, Controls, Settings, Source)
        right_pane = tk.Frame(body, bg=BG_PANEL, width=420, highlightthickness=1, highlightbackground=BORDER_COLOR)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 0))
        right_pane.pack_propagate(False)

        self._build_sidebar(right_pane)

    # --------------------------------------------------------------------------
    # VIDEO VIEWPORT & TIMELINE CONTROLS
    # --------------------------------------------------------------------------
    def _build_viewport(self, parent: tk.Frame):
        # 1. Canvas for high-FPS video rendering
        self.canvas_frame = tk.Frame(parent, bg="#05070a")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg="#05070a",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder message when idle
        self._render_placeholder()

        # 2. Video Scrubbing Bar (for video files)
        self.timeline_frame = tk.Frame(parent, bg=BG_PANEL, height=36)
        self.timeline_frame.pack(fill=tk.X, padx=12, pady=(6, 2))

        self.time_lbl_current = tk.Label(
            self.timeline_frame,
            text="00:00",
            font=(FONT_FAMILY, 9),
            fg=TEXT_MUTED,
            bg=BG_PANEL
        )
        self.time_lbl_current.pack(side=tk.LEFT, padx=(0, 8))

        self.timeline_slider = ttk.Scale(
            self.timeline_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            style="Horizontal.TScale",
            command=self._on_scrub
        )
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.timeline_slider.bind("<ButtonPress-1>", lambda e: setattr(self, "is_video_scrubbing", True))
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_scrub_release)

        self.time_lbl_total = tk.Label(
            self.timeline_frame,
            text="00:00",
            font=(FONT_FAMILY, 9),
            fg=TEXT_MUTED,
            bg=BG_PANEL
        )
        self.time_lbl_total.pack(side=tk.RIGHT, padx=(8, 0))

        # 3. Master Playback Controls Bar
        control_bar = tk.Frame(parent, bg=BG_PANEL, height=48)
        control_bar.pack(fill=tk.X, padx=12, pady=(2, 10))

        self.btn_play_pause = tk.Button(
            control_bar,
            text="▶ Play Source",
            font=(FONT_FAMILY, 10, "bold"),
            bg=ACCENT_BLUE,
            fg="white",
            activebackground=ACCENT_CYAN,
            activeforeground="black",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._on_toggle_play
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_cam_quick = tk.Button(
            control_bar,
            text="📹 Laptop Camera",
            font=(FONT_FAMILY, 10, "bold"),
            bg="#238636",
            fg="white",
            activebackground="#2ea043",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_start_camera
        )
        self.btn_cam_quick.pack(side=tk.LEFT, padx=4)

        self.btn_stop = tk.Button(
            control_bar,
            text="■ Stop",
            font=(FONT_FAMILY, 10),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            activebackground=BORDER_COLOR,
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_stop
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        # Loop checkbox
        chk_loop = tk.Checkbutton(
            control_bar,
            text="🔁 Loop Video",
            variable=self.loop_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            activeforeground=ACCENT_CYAN,
            command=self._on_toggle_loop
        )
        chk_loop.pack(side=tk.LEFT, padx=12)

        # Reset Tracking Counters
        btn_reset_track = tk.Button(
            control_bar,
            text="🔄 Reset Analytics",
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            activebackground=BORDER_COLOR,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_reset_tracking
        )
        btn_reset_track.pack(side=tk.RIGHT, padx=(6, 0))

    def _render_placeholder(self):
        """Draws clean cyber placeholder graphic onto the video canvas."""
        self.canvas.delete("all")
        w = max(400, self.canvas.winfo_width())
        h = max(300, self.canvas.winfo_height())
        cx, cy = w // 2, h // 2

        self.canvas.create_text(
            cx, cy - 30,
            text="📹",
            font=(FONT_FAMILY, 44),
            fill=TEXT_MUTED
        )
        self.canvas.create_text(
            cx, cy + 24,
            text="CityFlow AI Vehicle Detection Engine",
            font=(FONT_FAMILY, 14, "bold"),
            fill=TEXT_LIGHT
        )
        self.canvas.create_text(
            cx, cy + 50,
            text="Select an Input Source from the right panel and click 'Play Source'",
            font=(FONT_FAMILY, 10),
            fill=TEXT_MUTED
        )

    # --------------------------------------------------------------------------
    # SIDEBAR TABS (ANALYTICS, SOURCE, SETTINGS, EXPORT)
    # --------------------------------------------------------------------------
    def _build_sidebar(self, parent: tk.Frame):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1: Analytics & Counts
        tab_analytics = tk.Frame(notebook, bg=BG_PANEL)
        notebook.add(tab_analytics, text="📊 Analytics")
        self._build_tab_analytics(tab_analytics)

        # Tab 2: Video Sources
        tab_source = tk.Frame(notebook, bg=BG_PANEL)
        notebook.add(tab_source, text="📹 Source")
        self._build_tab_source(tab_source)

        # Tab 3: Model & Detection Settings
        tab_settings = tk.Frame(notebook, bg=BG_PANEL)
        notebook.add(tab_settings, text="🎛️ Settings")
        self._build_tab_settings(tab_settings)

        # Tab 4: Export & System
        tab_export = tk.Frame(notebook, bg=BG_PANEL)
        notebook.add(tab_export, text="💾 Export")
        self._build_tab_export(tab_export)

    # --------------------------------------------------------------------------
    # TAB 1: ANALYTICS & REAL-TIME VEHICLE COUNTS
    # --------------------------------------------------------------------------
    def _build_tab_analytics(self, parent: tk.Frame):
        scroll_canvas = tk.Canvas(parent, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas, bg=BG_PANEL)

        scroll_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=380)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        scrollbar.pack(side="right", fill="y")

        # 1. Total Live Vehicle Count Card & Traffic Density
        top_stats = tk.Frame(scroll_frame, bg=BG_PANEL)
        top_stats.pack(fill=tk.X, pady=(4, 8))

        card_total = tk.Frame(top_stats, bg=BG_PANEL_ALT, padx=12, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        card_total.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        tk.Label(card_total, text="LIVE VEHICLES", font=(FONT_FAMILY, 8, "bold"), fg=TEXT_MUTED, bg=BG_PANEL_ALT).pack(anchor="w")
        self.lbl_live_total = tk.Label(card_total, text="0", font=(FONT_FAMILY, 24, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL_ALT)
        self.lbl_live_total.pack(anchor="w")

        card_density = tk.Frame(top_stats, bg=BG_PANEL_ALT, padx=12, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        card_density.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        tk.Label(card_density, text="TRAFFIC DENSITY", font=(FONT_FAMILY, 8, "bold"), fg=TEXT_MUTED, bg=BG_PANEL_ALT).pack(anchor="w")
        self.lbl_density = tk.Label(card_density, text="SMOOTH", font=(FONT_FAMILY, 12, "bold"), fg=ACCENT_GREEN, bg=BG_PANEL_ALT)
        self.lbl_density.pack(anchor="w", pady=(6, 0))

        # 2. Cumulative Unique Tracked Vehicles & Tripwire Flow
        flow_frame = tk.Frame(scroll_frame, bg=BG_PANEL_ALT, padx=12, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        flow_frame.pack(fill=tk.X, pady=(0, 8))

        f_row1 = tk.Frame(flow_frame, bg=BG_PANEL_ALT)
        f_row1.pack(fill=tk.X)
        tk.Label(f_row1, text="TOTAL UNIQUE TRACKED:", font=(FONT_FAMILY, 9, "bold"), fg=TEXT_MUTED, bg=BG_PANEL_ALT).pack(side=tk.LEFT)
        self.lbl_unique_total = tk.Label(f_row1, text="0", font=(FONT_FAMILY, 12, "bold"), fg=ACCENT_GREEN, bg=BG_PANEL_ALT)
        self.lbl_unique_total.pack(side=tk.RIGHT)

        f_row2 = tk.Frame(flow_frame, bg=BG_PANEL_ALT)
        f_row2.pack(fill=tk.X, pady=(4, 0))
        self.lbl_inbound = tk.Label(f_row2, text="⬇ INBOUND: 0", font=(FONT_FAMILY, 9, "bold"), fg=ACCENT_BLUE, bg=BG_PANEL_ALT)
        self.lbl_inbound.pack(side=tk.LEFT)
        self.lbl_outbound = tk.Label(f_row2, text="⬆ OUTBOUND: 0", font=(FONT_FAMILY, 9, "bold"), fg=ACCENT_YELLOW, bg=BG_PANEL_ALT)
        self.lbl_outbound.pack(side=tk.RIGHT)

        # 3. Class Breakdown Badges
        breakdown_frame = tk.LabelFrame(
            scroll_frame,
            text=" Vehicle Breakdown by Class ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=6,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        breakdown_frame.pack(fill=tk.X, pady=(0, 8))

        self.class_count_labels = {}
        class_emojis = {
            "Car": "🚗",
            "Motorcycle": "🏍️",
            "Bus": "🚌",
            "Truck": "🚚",
            "Bicycle": "🚲"
        }
        class_colors_hex = {
            "Car": "#008cff",
            "Motorcycle": "#ff69b4",
            "Bus": "#ffaa00",
            "Truck": "#ff3333",
            "Bicycle": "#ffd700"
        }

        for cls_name, emoji in class_emojis.items():
            row = tk.Frame(breakdown_frame, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=2)

            bullet = tk.Label(
                row,
                text=f"{emoji} {cls_name}",
                font=(FONT_FAMILY, 9, "bold"),
                fg=class_colors_hex.get(cls_name, TEXT_LIGHT),
                bg=BG_PANEL
            )
            bullet.pack(side=tk.LEFT)

            cnt_lbl = tk.Label(
                row,
                text="0",
                font=(FONT_FAMILY, 9, "bold"),
                fg=TEXT_LIGHT,
                bg=BG_INPUT,
                padx=8,
                pady=1,
                relief="flat"
            )
            cnt_lbl.pack(side=tk.RIGHT)
            self.class_count_labels[cls_name] = cnt_lbl

        # 4. Multi-Vehicle Active Roster Table
        roster_frame = tk.LabelFrame(
            scroll_frame,
            text=" 🚘 Active Multi-Vehicle Inspector ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_PANEL,
            padx=6,
            pady=6,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        roster_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Table Treeview
        cols = ("id", "class", "conf", "dir", "lane")
        self.tree_vehicles = ttk.Treeview(
            roster_frame,
            columns=cols,
            show="headings",
            height=5,
            selectmode="browse"
        )
        self.tree_vehicles.heading("id", text="ID")
        self.tree_vehicles.heading("class", text="Type")
        self.tree_vehicles.heading("conf", text="Conf")
        self.tree_vehicles.heading("dir", text="Direction")
        self.tree_vehicles.heading("lane", text="Lane")

        self.tree_vehicles.column("id", width=45, anchor="center")
        self.tree_vehicles.column("class", width=65, anchor="w")
        self.tree_vehicles.column("conf", width=45, anchor="center")
        self.tree_vehicles.column("dir", width=100, anchor="w")
        self.tree_vehicles.column("lane", width=85, anchor="center")
        self.tree_vehicles.pack(fill=tk.BOTH, expand=True)

        # 5. Real-Time Detection Events Feed
        events_frame = tk.LabelFrame(
            scroll_frame,
            text=" Live Activity Stream ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=8,
            pady=6,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        events_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.event_listbox = tk.Listbox(
            events_frame,
            height=5,
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            selectbackground=BG_PANEL_ALT,
            selectforeground=ACCENT_CYAN,
            font=("Consolas", 8),
            borderwidth=0,
            highlightthickness=0
        )
        self.event_listbox.pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------------------
    # TAB 2: INPUT SOURCE SELECTION
    # --------------------------------------------------------------------------
    def _build_tab_source(self, parent: tk.Frame):
        scroll_canvas = tk.Canvas(parent, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas, bg=BG_PANEL)

        scroll_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=380)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        scrollbar.pack(side="right", fill="y")

        # 1. Built-in Demo Traffic Clips (Instant Zero-Setup PC Testing)
        demo_card = tk.LabelFrame(
            scroll_frame,
            text=" 🧪 Built-in Test Samples (Instant PC Test) ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_PANEL,
            padx=10,
            pady=10,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        demo_card.pack(fill=tk.X, pady=(4, 10))

        tk.Label(
            demo_card,
            text="Test vehicle detection instantly without extra setup:",
            font=(FONT_FAMILY, 8),
            fg=TEXT_MUTED,
            bg=BG_PANEL,
            wraplength=340,
            justify="left"
        ).pack(anchor="w", pady=(0, 8))

        btn_sample_vid = tk.Button(
            demo_card,
            text="🎬 Load Sample Traffic Video",
            font=(FONT_FAMILY, 9, "bold"),
            bg=BG_INPUT,
            fg=ACCENT_CYAN,
            activebackground=BORDER_COLOR,
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2",
            command=self._on_load_sample_video
        )
        btn_sample_vid.pack(fill=tk.X, pady=3)

        btn_sample_img = tk.Button(
            demo_card,
            text="🖼️ Load Sample Bus Image",
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            activebackground=BORDER_COLOR,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_load_sample_image
        )
        btn_sample_img.pack(fill=tk.X, pady=3)

        # 2. Video File Source
        video_card = tk.LabelFrame(
            scroll_frame,
            text=" 📁 Video File ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        video_card.pack(fill=tk.X, pady=(0, 10))

        v_row = tk.Frame(video_card, bg=BG_PANEL)
        v_row.pack(fill=tk.X, pady=3)

        self.entry_video = tk.Entry(
            v_row,
            textvariable=self.video_path_var,
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            insertbackground=TEXT_LIGHT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        self.entry_video.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_browse_vid = tk.Button(
            v_row,
            text="Browse...",
            font=(FONT_FAMILY, 8),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._on_browse_video
        )
        btn_browse_vid.pack(side=tk.RIGHT)

        btn_play_vid = tk.Button(
            video_card,
            text="▶ Play Video File",
            font=(FONT_FAMILY, 9, "bold"),
            bg=BG_INPUT,
            fg=ACCENT_BLUE,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_start_video_file
        )
        btn_play_vid.pack(fill=tk.X, pady=(4, 0))

        # 3. Webcam / Camera Index
        cam_card = tk.LabelFrame(
            scroll_frame,
            text=" 📹 Live Webcam / USB Camera ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        cam_card.pack(fill=tk.X, pady=(0, 10))

        c_row = tk.Frame(cam_card, bg=BG_PANEL)
        c_row.pack(fill=tk.X, pady=3)

        tk.Label(c_row, text="Camera Index:", font=(FONT_FAMILY, 9), fg=TEXT_MUTED, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 8))
        self.combo_cam = ttk.Combobox(
            c_row,
            textvariable=self.camera_idx_var,
            values=["0", "1", "2", "3"],
            width=6,
            state="readonly"
        )
        self.combo_cam.pack(side=tk.LEFT)

        btn_connect_cam = tk.Button(
            cam_card,
            text="⚡ Connect Live Camera",
            font=(FONT_FAMILY, 9, "bold"),
            bg=BG_INPUT,
            fg=ACCENT_GREEN,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_start_camera
        )
        btn_connect_cam.pack(fill=tk.X, pady=(6, 0))

        # 4. Image File
        img_card = tk.LabelFrame(
            scroll_frame,
            text=" 🖼️ Single Image File ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        img_card.pack(fill=tk.X, pady=(0, 10))

        i_row = tk.Frame(img_card, bg=BG_PANEL)
        i_row.pack(fill=tk.X, pady=3)

        self.entry_img = tk.Entry(
            i_row,
            textvariable=self.image_path_var,
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            insertbackground=TEXT_LIGHT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        self.entry_img.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_browse_img = tk.Button(
            i_row,
            text="Browse...",
            font=(FONT_FAMILY, 8),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._on_browse_image
        )
        btn_browse_img.pack(side=tk.RIGHT)

        btn_analyze_img = tk.Button(
            img_card,
            text="🔍 Analyze Single Image",
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_start_image
        )
        btn_analyze_img.pack(fill=tk.X, pady=(4, 0))

        # 5. RTSP / IP Camera Stream
        rtsp_card = tk.LabelFrame(
            scroll_frame,
            text=" 🌐 RTSP / Network Stream URL ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        rtsp_card.pack(fill=tk.X, pady=(0, 8))

        self.entry_rtsp = tk.Entry(
            rtsp_card,
            textvariable=self.rtsp_url_var,
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            insertbackground=TEXT_LIGHT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        self.entry_rtsp.pack(fill=tk.X, pady=3)

        btn_connect_rtsp = tk.Button(
            rtsp_card,
            text="Connect RTSP Stream",
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_start_rtsp
        )
        btn_connect_rtsp.pack(fill=tk.X, pady=(4, 0))

    # --------------------------------------------------------------------------
    # TAB 3: MODEL & DETECTION SETTINGS
    # --------------------------------------------------------------------------
    def _build_tab_settings(self, parent: tk.Frame):
        scroll_canvas = tk.Canvas(parent, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas, bg=BG_PANEL)

        scroll_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=380)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        scrollbar.pack(side="right", fill="y")

        # 1. Model Weights Selector
        m_frame = tk.LabelFrame(
            scroll_frame,
            text=" YOLO Model Architecture ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        m_frame.pack(fill=tk.X, pady=(4, 10))

        self.combo_model = ttk.Combobox(
            m_frame,
            textvariable=self.model_name_var,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt"],
            state="readonly"
        )
        self.combo_model.pack(fill=tk.X, pady=3)
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_change)

        # 2. Confidence & IoU Sliders
        thresh_frame = tk.LabelFrame(
            scroll_frame,
            text=" Detection Thresholds ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        thresh_frame.pack(fill=tk.X, pady=(0, 10))

        # Confidence Slider
        c_head = tk.Frame(thresh_frame, bg=BG_PANEL)
        c_head.pack(fill=tk.X)
        tk.Label(c_head, text="Confidence Threshold:", font=(FONT_FAMILY, 9), fg=TEXT_LIGHT, bg=BG_PANEL).pack(side=tk.LEFT)
        self.lbl_conf_val = tk.Label(c_head, text="40%", font=(FONT_FAMILY, 9, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL)
        self.lbl_conf_val.pack(side=tk.RIGHT)

        self.scale_conf = ttk.Scale(
            thresh_frame,
            from_=0.10,
            to=0.95,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            style="Horizontal.TScale",
            command=self._on_conf_change
        )
        self.scale_conf.pack(fill=tk.X, pady=(2, 8))

        # IoU Slider
        i_head = tk.Frame(thresh_frame, bg=BG_PANEL)
        i_head.pack(fill=tk.X)
        tk.Label(i_head, text="NMS IoU Threshold:", font=(FONT_FAMILY, 9), fg=TEXT_LIGHT, bg=BG_PANEL).pack(side=tk.LEFT)
        self.lbl_iou_val = tk.Label(i_head, text="45%", font=(FONT_FAMILY, 9, "bold"), fg=ACCENT_BLUE, bg=BG_PANEL)
        self.lbl_iou_val.pack(side=tk.RIGHT)

        self.scale_iou = ttk.Scale(
            thresh_frame,
            from_=0.10,
            to=0.95,
            orient=tk.HORIZONTAL,
            variable=self.iou_var,
            style="Horizontal.TScale",
            command=self._on_iou_change
        )
        self.scale_iou.pack(fill=tk.X, pady=(2, 4))

        # 3. Vehicle Class Filter Checkboxes
        cls_frame = tk.LabelFrame(
            scroll_frame,
            text=" Active Vehicle Classes to Detect ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        cls_frame.pack(fill=tk.X, pady=(0, 10))

        for name, var in self.class_filters.items():
            chk = tk.Checkbutton(
                cls_frame,
                text=f"Detect {name}",
                variable=var,
                font=(FONT_FAMILY, 9),
                fg=TEXT_LIGHT,
                bg=BG_PANEL,
                selectcolor=BG_INPUT,
                activebackground=BG_PANEL,
                activeforeground=ACCENT_CYAN,
                command=self._on_class_filter_change
            )
            chk.pack(anchor="w", pady=1)

        # 4. Tracking & Visual Overlays
        overlay_frame = tk.LabelFrame(
            scroll_frame,
            text=" Multi-Vehicle Tracking & Analytics ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        overlay_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Checkbutton(
            overlay_frame,
            text="Enable ByteTrack Multi-Object Tracking",
            variable=self.tracking_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            command=self._on_tracking_toggle
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            overlay_frame,
            text="Distinct Colors for Each Vehicle ID",
            variable=self.color_by_track_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            command=self._on_color_by_track_toggle
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            overlay_frame,
            text="Show Trajectory Motion Trails",
            variable=self.trails_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            command=self._on_trails_toggle
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            overlay_frame,
            text="Show Vehicle Movement Directions (⬆ ⬇ ➡ ⬅)",
            variable=self.direction_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            command=self._on_direction_toggle
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            overlay_frame,
            text="Enable Virtual Tripwire / Flow Counting Line",
            variable=self.counting_line_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            command=self._on_counting_line_toggle
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            overlay_frame,
            text="Show Top Telemetry HUD Banner",
            variable=self.hud_var,
            font=(FONT_FAMILY, 9),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            selectcolor=BG_INPUT,
            activebackground=BG_PANEL,
            command=self._on_hud_toggle
        ).pack(anchor="w", pady=1)

    # --------------------------------------------------------------------------
    # TAB 4: EXPORT & RECORDS
    # --------------------------------------------------------------------------
    def _build_tab_export(self, parent: tk.Frame):
        scroll_frame = tk.Frame(parent, bg=BG_PANEL, padx=10, pady=10)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Output folder status
        stat_card = tk.LabelFrame(
            scroll_frame,
            text=" Session Data Export ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=10,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        stat_card.pack(fill=tk.X, pady=(0, 10))

        btn_csv = tk.Button(
            stat_card,
            text="📊 Export Session Report to CSV",
            font=(FONT_FAMILY, 9, "bold"),
            bg=BG_INPUT,
            fg=ACCENT_CYAN,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            command=self._on_export_csv
        )
        btn_csv.pack(fill=tk.X, pady=4)

        btn_open = tk.Button(
            stat_card,
            text="📂 Open Outputs Directory in Explorer",
            font=(FONT_FAMILY, 9),
            bg=BG_INPUT,
            fg=TEXT_LIGHT,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            command=self._on_open_outputs
        )
        btn_open.pack(fill=tk.X, pady=4)

        # Diagnostic info
        diag_card = tk.LabelFrame(
            scroll_frame,
            text=" System Diagnostics ",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_PANEL,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        diag_card.pack(fill=tk.X, pady=(0, 10))

        cuda_avail = torch.cuda.is_available()
        diag_info = [
            f"PyTorch: {torch.__version__}",
            f"CUDA Available: {'Yes' if cuda_avail else 'No'}",
            f"GPU Device: {torch.cuda.get_device_name(0) if cuda_avail else 'N/A'}",
            f"OpenCV: {cv2.__version__}",
            f"Python: {sys.version.split()[0]}"
        ]
        for line in diag_info:
            tk.Label(
                diag_card,
                text=line,
                font=("Consolas", 8),
                fg=TEXT_MUTED,
                bg=BG_PANEL,
                anchor="w"
            ).pack(fill=tk.X, pady=1)

    # --------------------------------------------------------------------------
    # STATUS BAR
    # --------------------------------------------------------------------------
    def _build_status_bar(self):
        status_bar = tk.Frame(self, bg=BG_PANEL_ALT, height=26, highlightthickness=1, highlightbackground=BORDER_COLOR)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar.pack_propagate(False)

        self.status_msg_lbl = tk.Label(
            status_bar,
            text="Ready. Select an input source to begin vehicle detection.",
            font=(FONT_FAMILY, 8),
            fg=TEXT_MUTED,
            bg=BG_PANEL_ALT
        )
        self.status_msg_lbl.pack(side=tk.LEFT, padx=12)

        self.res_lbl = tk.Label(
            status_bar,
            text="1280x720",
            font=(FONT_FAMILY, 8),
            fg=TEXT_MUTED,
            bg=BG_PANEL_ALT
        )
        self.res_lbl.pack(side=tk.RIGHT, padx=12)

    # --------------------------------------------------------------------------
    # CONTROL HANDLERS
    # --------------------------------------------------------------------------
    def _on_canvas_resize(self, event):
        """Redraw placeholder if no active video stream is rendering."""
        if self.worker is None or not self.worker.is_alive():
            self._render_placeholder()

    def _start_source(self, source_path_or_idx: str):
        """Stops any existing worker and starts a new worker thread for the given source."""
        self._on_stop()

        # Update class filters in config
        self._sync_class_filters()

        self.worker = VideoWorker(
            source=source_path_or_idx,
            config=self.config,
            detector=self.detector,
            visualizer=self.visualizer,
            frame_queue=self.frame_queue,
            event_queue=self.event_queue,
            loop_video=self.loop_var.get()
        )
        self.worker.start()

        self.btn_play_pause.configure(text="❚❚ Pause", bg=ACCENT_YELLOW, fg="black")
        self.status_pill.configure(text="● RUNNING", fg=ACCENT_GREEN)
        self._set_status(f"Streaming from: {source_path_or_idx}")

    def _auto_start_camera(self):
        """Attempts to auto-connect laptop camera (index 0) on startup."""
        logger.info("Auto-starting laptop camera (Index 0)...")
        try:
            self._start_source("0")
        except Exception as e:
            logger.warning(f"Could not auto-start camera: {e}")

    def _on_toggle_play(self):
        """Toggles between Play / Pause / Start."""
        if self.worker is not None and self.worker.is_alive():
            is_paused = self.worker.toggle_pause()
            if is_paused:
                self.btn_play_pause.configure(text="▶ Resume", bg=ACCENT_GREEN, fg="black")
                self.status_pill.configure(text="❚❚ PAUSED", fg=ACCENT_YELLOW)
                self._set_status("Video stream paused.")
            else:
                self.btn_play_pause.configure(text="❚❚ Pause", bg=ACCENT_YELLOW, fg="black")
                self.status_pill.configure(text="● RUNNING", fg=ACCENT_GREEN)
                self._set_status("Video stream resumed.")
        else:
            # Start laptop camera by default
            self._on_start_camera()

    def _on_stop(self):
        """Stops active stream worker."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=1.0)
            self.worker = None

        self.btn_play_pause.configure(text="▶ Play Source", bg=ACCENT_BLUE, fg="white")
        self.btn_record.configure(text="🎥 Record", bg="#b62324")
        self.status_pill.configure(text="■ STOPPED", fg=TEXT_MUTED)
        self.fps_badge.configure(text="0.0 FPS")
        self.latency_badge.configure(text="0.0 ms")
        self._set_status("Stream stopped.")

    def _on_load_sample_video(self):
        sample_path = "outputs/sample_traffic.mp4"
        if not os.path.exists(sample_path):
            sample_path = "outputs/sample_traffic_real.mp4"
        self.video_path_var.set(sample_path)
        self._start_source(sample_path)

    def _on_load_sample_image(self):
        sample_img = "outputs/sample_bus.jpg"
        self.image_path_var.set(sample_img)
        self._start_source(sample_img)

    def _on_start_video_file(self):
        path = self.video_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("File Not Found", f"Please select a valid video file:\n{path}")
            return
        self._start_source(path)

    def _on_start_camera(self):
        cam_idx = self.camera_idx_var.get().strip()
        self._start_source(cam_idx)

    def _on_start_image(self):
        path = self.image_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("File Not Found", f"Please select a valid image file:\n{path}")
            return
        self._start_source(path)

    def _on_start_rtsp(self):
        url = self.rtsp_url_var.get().strip()
        if not url:
            messagebox.showwarning("Empty URL", "Please enter a valid RTSP stream URL.")
            return
        self._start_source(url)

    def _on_browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv"), ("All Files", "*.*")]
        )
        if path:
            self.video_path_var.set(path)
            self._start_source(path)

    def _on_browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if path:
            self.image_path_var.set(path)
            self._start_source(path)

    def _on_scrub(self, val):
        if self.worker and self.is_video_scrubbing:
            self.worker.seek(float(val))

    def _on_scrub_release(self, event):
        self.is_video_scrubbing = False
        if self.worker:
            self.worker.seek(self.timeline_slider.get())

    def _on_toggle_loop(self):
        if self.worker:
            self.worker.loop_video = self.loop_var.get()

    def _on_reset_tracking(self):
        self.detector.reset_tracking()
        self.lbl_unique_total.configure(text="0")
        self.event_listbox.delete(0, tk.END)
        self.session_events.clear()
        self._set_status("Tracking counters and trajectory history reset.")

    def _on_take_snapshot(self):
        """Saves high-res snapshot of the current frame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(self.config.snapshots_dir, f"snapshot_{timestamp}.jpg")
        if self.worker and self.worker.is_alive():
            self.worker.request_snapshot(snap_path)
            self._set_status(f"Capturing snapshot -> {snap_path}")
        elif self.last_frame_annotated is not None:
            os.makedirs(self.config.snapshots_dir, exist_ok=True)
            cv2.imwrite(snap_path, self.last_frame_annotated)
            self._set_status(f"Snapshot saved to: {snap_path}")
            messagebox.showinfo("Snapshot Saved", f"Saved high-res snapshot:\n{snap_path}")
        else:
            messagebox.showinfo("No Active Stream", "Start a video feed or load an image first to take a snapshot.")

    def _on_toggle_record(self):
        """Starts or stops live video recording."""
        if self.worker is None or not self.worker.is_alive():
            messagebox.showinfo("No Active Stream", "Start a video source first to record.")
            return

        if not self.worker.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_path = os.path.join(self.config.recordings_dir, f"recording_{timestamp}.mp4")
            
            w = self.config.frame_width
            h = self.config.frame_height
            if self.last_frame_annotated is not None:
                h, w = self.last_frame_annotated.shape[:2]

            self.worker.start_recording(rec_path, (w, h), fps=30.0)
            self.btn_record.configure(text="⏹ Stop Rec", bg="#ff416c")
            self._set_status(f"Recording video to: {rec_path}")
        else:
            self.worker.stop_recording()
            self.btn_record.configure(text="🎥 Record", bg="#b62324")
            self._set_status("Recording stopped.")

    def _on_export_csv(self):
        """Exports session detection counts and events to CSV."""
        os.makedirs(self.config.exports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.config.exports_dir, f"traffic_report_{timestamp}.csv")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["CityFlow AI - Traffic Detection Session Report"])
                writer.writerow(["Export Timestamp", datetime.now().isoformat()])
                writer.writerow(["Model", self.config.model_name])
                writer.writerow(["Confidence Threshold", f"{self.config.confidence_threshold:.2f}"])
                writer.writerow(["Total Unique Tracked Vehicles", self.detector.get_total_unique_vehicles()])
                writer.writerow([])
                writer.writerow(["Vehicle Class Breakdown"])
                for cls_name, cnt in self.current_counts.items():
                    writer.writerow([cls_name, cnt])
                writer.writerow([])
                writer.writerow(["Detection Event Logs"])
                writer.writerow(["Timestamp", "Class", "Confidence", "Track ID"])
                for evt in self.session_events:
                    writer.writerow([evt.get("time"), evt.get("class"), evt.get("conf"), evt.get("track_id")])

            self._set_status(f"Session report exported to: {csv_path}")
            messagebox.showinfo("Export Complete", f"Traffic analytics report exported to:\n{csv_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export CSV: {e}")

    def _on_open_outputs(self):
        """Opens the outputs directory in Windows Explorer."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(self.config.output_dir))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", os.path.abspath(self.config.output_dir)])
        except Exception as e:
            logger.error(f"Could not open folder: {e}")

    # --------------------------------------------------------------------------
    # PARAMETER CHANGE LISTENERS
    # --------------------------------------------------------------------------
    def _on_model_change(self, event=None):
        model_name = self.model_name_var.get()
        self._set_status(f"Loading YOLO model weights: {model_name}...")
        self.update_idletasks()
        try:
            success = self.detector.change_model(model_name)
            if success:
                self._set_status(f"Switched model to {model_name}")
            else:
                messagebox.showerror("Model Load Error", f"Failed to load model {model_name}")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))

    def _on_conf_change(self, val):
        conf = float(val)
        self.config.confidence_threshold = conf
        self.detector.config.confidence_threshold = conf
        self.lbl_conf_val.configure(text=f"{int(conf * 100)}%")

    def _on_iou_change(self, val):
        iou = float(val)
        self.config.iou_threshold = iou
        self.detector.config.iou_threshold = iou
        self.lbl_iou_val.configure(text=f"{int(iou * 100)}%")

    def _on_tracking_toggle(self):
        enabled = self.tracking_var.get()
        self.config.enable_tracking = enabled
        self.detector.config.enable_tracking = enabled

    def _on_trails_toggle(self):
        self.config.show_trails = self.trails_var.get()
        self.visualizer.config.show_trails = self.trails_var.get()

    def _on_hud_toggle(self):
        self.config.show_hud = self.hud_var.get()
        self.visualizer.config.show_hud = self.hud_var.get()

    def _on_counting_line_toggle(self):
        enabled = self.counting_line_var.get()
        self.config.enable_counting_line = enabled
        self.visualizer.config.enable_counting_line = enabled

    def _on_direction_toggle(self):
        enabled = self.direction_var.get()
        self.config.show_direction = enabled
        self.visualizer.config.show_direction = enabled

    def _on_color_by_track_toggle(self):
        enabled = self.color_by_track_var.get()
        self.config.color_by_track_id = enabled
        self.detector.config.color_by_track_id = enabled

    def _on_class_filter_change(self):
        self._sync_class_filters()

    def _sync_class_filters(self):
        """Updates the list of target COCO class IDs based on checkboxes."""
        name_to_id = {"Bicycle": 1, "Car": 2, "Motorcycle": 3, "Bus": 5, "Truck": 7}
        active_ids = [name_to_id[name] for name, var in self.class_filters.items() if var.get() and name in name_to_id]
        if not active_ids:
            active_ids = [1, 2, 3, 5, 7]  # fallback
        self.config.target_class_ids = active_ids
        self.detector.config.target_class_ids = active_ids

    def _set_status(self, msg: str):
        self.status_msg_lbl.configure(text=msg)

    # --------------------------------------------------------------------------
    # PERIODIC QUEUE POLLING & CANVAS RENDERING
    # --------------------------------------------------------------------------
    def _process_frame_queue(self):
        """Checks for newly processed video frames and renders them on the canvas."""
        try:
            while not self.frame_queue.empty():
                data = self.frame_queue.get_nowait()
                self._render_frame(data)
        except Exception as e:
            logger.debug(f"Frame queue error: {e}")

        self.after(15, self._process_frame_queue)

    def _render_frame(self, data: dict):
        """Renders annotated frame onto the Tkinter canvas with smooth scaling."""
        annotated_bgr = data["annotated"]
        self.last_frame_annotated = annotated_bgr
        self.last_frame_raw = data["raw"]

        # Convert OpenCV BGR to RGB PIL Image
        rgb_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Get Canvas dimensions
        c_w = max(100, self.canvas.winfo_width())
        c_h = max(100, self.canvas.winfo_height())

        # Aspect-Ratio Preserving Scaling
        img_w, img_h = pil_img.size
        scale = min(c_w / img_w, c_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        photo_img = ImageTk.PhotoImage(resized_img)

        # Draw centered on canvas
        self.canvas.delete("all")
        offset_x = (c_w - new_w) // 2
        offset_y = (c_h - new_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=photo_img)
        self.canvas.image = photo_img  # Keep reference to avoid garbage collection

        # Update Telemetry & Badges
        fps = data["fps"]
        latency = data["latency"]
        self.fps_badge.configure(text=f"{fps:.1f} FPS")
        self.latency_badge.configure(text=f"{latency:.1f} ms")
        self.res_lbl.configure(text=f"{img_w}x{img_h}")

        # Update Multi-Vehicle Counts & Density
        counts = data["counts"]
        self.current_counts = counts
        live_total = sum(counts.values())
        self.lbl_live_total.configure(text=str(live_total))
        self.lbl_unique_total.configure(text=str(data["unique_tracks"]))

        dens_lbl = data.get("density_label", "SMOOTH")
        self.lbl_density.configure(text=dens_lbl)
        self.lbl_inbound.configure(text=f"⬇ INBOUND: {data.get('inbound_count', 0)}")
        self.lbl_outbound.configure(text=f"⬆ OUTBOUND: {data.get('outbound_count', 0)}")

        for cls_name, lbl in self.class_count_labels.items():
            cnt = counts.get(cls_name, 0)
            lbl.configure(text=str(cnt))

        # Update Multi-Vehicle Active Roster Table
        detections: List[DetectedVehicle] = data.get("detections", [])
        for item in self.tree_vehicles.get_children():
            self.tree_vehicles.delete(item)
        for det in detections:
            tid = f"#{det.track_id}" if det.track_id is not None else "-"
            conf_str = f"{int(det.confidence * 100)}%"
            self.tree_vehicles.insert("", tk.END, values=(tid, det.class_name, conf_str, det.direction, det.lane_pos))

        # Update Event Activity Stream for new detections
        for det in detections:
            if det.track_id is not None and not any(e.get("track_id") == det.track_id for e in self.session_events):
                now_str = datetime.now().strftime("%H:%M:%S")
                evt_text = f"[{now_str}] #{det.track_id} {det.class_name} ({int(det.confidence*100)}%) {det.direction}"
                self.event_listbox.insert(0, evt_text)
                if self.event_listbox.size() > 50:
                    self.event_listbox.delete(50, tk.END)
                self.session_events.append({
                    "time": now_str,
                    "class": det.class_name,
                    "conf": f"{det.confidence:.2f}",
                    "track_id": det.track_id,
                    "dir": det.direction
                })

        # Update Video Scrubber Position (if not scrubbing)
        total_f = data["total_frames"]
        curr_f = data["frame_idx"]
        if total_f > 0 and not self.is_video_scrubbing:
            prog = curr_f / total_f
            self.timeline_slider.set(prog)
            cur_sec = int(curr_f / 30.0)
            tot_sec = int(total_f / 30.0)
            self.time_lbl_current.configure(text=f"{cur_sec//60:02d}:{cur_sec%60:02d}")
            self.time_lbl_total.configure(text=f"{tot_sec//60:02d}:{tot_sec%60:02d}")

    def _process_event_queue(self):
        """Processes async worker events (status, errors, completion)."""
        try:
            while not self.event_queue.empty():
                evt_type, payload = self.event_queue.get_nowait()
                if evt_type == "snapshot_saved":
                    self._set_status(f"Snapshot saved: {payload}")
                    messagebox.showinfo("Snapshot Saved", f"High-res snapshot saved to:\n{payload}")
                elif evt_type == "recording_started":
                    self._set_status(f"Recording video: {payload}")
                elif evt_type == "recording_stopped":
                    path = payload.get("path")
                    dur = payload.get("duration", 0.0)
                    self._set_status(f"Recording saved ({dur:.1f}s): {path}")
                    messagebox.showinfo("Recording Saved", f"Video recording saved ({dur:.1f}s):\n{path}")
                elif evt_type == "stream_started":
                    info = payload or {}
                    w, h = info.get("res", (1280, 720))
                    self.res_lbl.configure(text=f"{w}x{h}")
                elif evt_type == "stream_ended":
                    self._on_stop()
                    self._set_status("Playback finished.")
                elif evt_type == "error":
                    self._on_stop()
                    messagebox.showerror("Stream Error", str(payload))
        except Exception as e:
            logger.debug(f"Event queue error: {e}")

        self.after(50, self._process_event_queue)

    def _on_close(self):
        """Clean shutdown handler."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=1.0)
        self.destroy()


def main():
    app = CityFlowApp()
    app.mainloop()


if __name__ == "__main__":
    main()
