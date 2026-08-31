"""
Video stream capture module supporting webcam feeds, video files, and RTSP streams.
Optimized for Windows low latency with DirectShow backend support.
"""
import sys
import logging
from typing import Union, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Robust Video Stream reader with automatic backend selection,
    reconnect handling, and dimension control.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        fps: Optional[int] = 30
    ):
        self.source = source
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

        if isinstance(source, str) and source.isdigit():
            self.source = int(source)

        self._initialize_stream()

    def _initialize_stream(self) -> None:
        """Initializes the VideoCapture instance with optimized backend flags."""
        if self.is_webcam:
            idx = int(self.source)
            logger.info(f"Opening camera index {idx}...")

            # On Windows, DirectShow (CAP_DSHOW) offers fast startup and low latency
            if sys.platform.startswith("win"):
                logger.debug("Attempting cv2.CAP_DSHOW backend on Windows")
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    logger.warning("CAP_DSHOW failed, falling back to default backend")
                    self.cap = cv2.VideoCapture(idx)
            else:
                self.cap = cv2.VideoCapture(idx)

            if self.cap and self.cap.isOpened():
                # Try setting MJPG format for better webcam FPS throughput
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if self.desired_width and self.desired_height:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
                if self.desired_fps:
                    self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Real-time minimal latency
        else:
            logger.info(f"Opening video source: {self.source}")
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            self.is_opened = False
        else:
            self.is_opened = True
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Video stream active: {w}x{h} @ {fps:.1f} FPS")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next frame from the stream."""
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        return True, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Returns current (width, height) of the video feed."""
        if self.cap and self.is_opened:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (0, 0)

    def get_fps(self) -> float:
        """Returns reported FPS of the video source."""
        if self.cap and self.is_opened:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def get_total_frames(self) -> int:
        """Returns total frame count for video files, or 0 for live cameras."""
        if self.cap and self.is_opened and not self.is_webcam:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def get_current_frame_index(self) -> int:
        """Returns current frame position index."""
        if self.cap and self.is_opened:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    def set_frame_index(self, frame_index: int) -> bool:
        """Seek to a specific frame index in video files."""
        if self.cap and self.is_opened and not self.is_webcam:
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        return False

    def restart(self) -> bool:
        """Restarts the stream from frame 0 if it's a video file."""
        if not self.is_webcam:
            return self.set_frame_index(0)
        return False

    def release(self) -> None:
        """Releases the video stream resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info("Video stream released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
