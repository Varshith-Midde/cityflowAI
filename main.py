"""
Main application runner for Real-Time Vehicle Detection and Tracking using YOLO.

Run with laptop webcam:
    python main.py

Run with a specific video file:
    python main.py --source traffic_sample.mp4

Run with a larger YOLO model:
    python main.py --model yolov8s.pt
"""
import os
import sys
import time
import argparse
import logging
from datetime import datetime

# Ensure UTF-8 console output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import torch

from config import VehicleDetectionConfig
from src.camera import VideoStream
from src.detector import VehicleDetector
from src.visualizer import HUDVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CityFlowAI")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CityFlow AI - Real-Time Vehicle Detection & Tracking with YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for default laptop camera, camera index, or path to video file/RTSP stream."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or name (e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt, yolo11n.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence detection threshold (0.0 to 1.0)."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU threshold (0.0 to 1.0)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Processing device: 'cuda', 'cuda:0', 'cpu', or '' (auto-detect)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Desired camera frame capture width."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Desired camera frame capture height."
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable multi-object tracking and persistent vehicle IDs."
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Hide the top telemetry & statistics HUD banner."
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the output video stream (e.g. output.mp4)."
    )
    return parser.parse_args()


def get_device_name(device_setting: str) -> str:
    """Returns a short description of the active computing device."""
    if device_setting.lower().startswith("cuda") or (not device_setting and torch.cuda.is_available()):
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        return f"GPU: {gpu_name.split()[0]}"
    return "CPU"


def main():
    args = parse_arguments()

    # Build configuration
    config = VehicleDetectionConfig(
        model_name=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        enable_tracking=not args.no_track,
        show_hud=not args.no_hud,
        frame_width=args.width,
        frame_height=args.height
    )

    # Ensure output directory exists for snapshots
    os.makedirs(config.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CITYFLOW AI - REAL-TIME VEHICLE DETECTION SYSTEM")
    print("=" * 60)
    print(f" Source:     {args.source}")
    print(f" Model:      {config.model_name}")
    print(f" Confidence: {config.confidence_threshold:.2f}")
    print(f" Tracking:   {'Enabled' if config.enable_tracking else 'Disabled'}")
    print("=" * 60)
    print(" Controls:")
    print("   [Q] / [ESC]   - Quit Application")
    print("   [P] / [SPACE] - Pause / Resume Video Feed")
    print("   [S]           - Save High-Res Snapshot")
    print("   [T]           - Toggle Tracking On/Off")
    print("   [C]           - Toggle Vehicle Counters")
    print("   [H]           - Toggle HUD Banner")
    print("=" * 60 + "\n")

    # Determine device information
    device_info = get_device_name(config.device)
    logger.info(f"Target execution hardware: {device_info}")

    # Initialize YOLO Vehicle Detector
    detector = VehicleDetector(config)

    # Initialize Video Capture
    source = int(args.source) if args.source.isdigit() else args.source
    stream = VideoStream(
        source=source,
        width=config.frame_width,
        height=config.frame_height
    )

    if not stream.is_opened:
        logger.error(f"Cannot proceed: Video source '{args.source}' could not be opened.")
        return

    # Initialize Video Writer if recording is requested
    writer = None
    if args.save:
        res_w, res_h = stream.get_resolution()
        fps = stream.get_fps()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (res_w, res_h))
        logger.info(f"Recording video output to: {args.save}")

    window_name = "CityFlow AI - Vehicle Detection & Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    visualizer = HUDVisualizer(config)
    is_paused = False
    last_frame = None

    try:
        while True:
            if not is_paused:
                ret, frame = stream.read()
                if not ret or frame is None:
                    if stream.is_webcam:
                        logger.warning("Camera stream interrupted. Attempting to continue...")
                        time.sleep(0.1)
                        continue
                    else:
                        logger.info("End of video stream reached.")
                        break

                # Execute detection & tracking
                detections, counts, fps = detector.process_frame(frame)

                # Collect track histories
                track_histories = {}
                if config.enable_tracking:
                    for det in detections:
                        if det.track_id is not None:
                            track_histories[det.track_id] = detector.get_track_history(det.track_id)

                # Render modern visualizer overlay
                annotated_frame = visualizer.draw_hud(
                    frame=frame,
                    detections=detections,
                    counts=counts,
                    fps=fps,
                    track_histories=track_histories,
                    is_paused=False,
                    device_info=device_info
                )
                last_frame = (frame, annotated_frame)
            else:
                # When paused, keep rendering paused state
                if last_frame is not None:
                    raw_frame, _ = last_frame
                    annotated_frame = visualizer.draw_hud(
                        frame=raw_frame,
                        detections=detections,
                        counts=counts,
                        fps=0.0,
                        track_histories=track_histories,
                        is_paused=True,
                        device_info=device_info
                    )

            # Write to output video file if enabled
            if writer is not None and not is_paused:
                writer.write(annotated_frame)

            # Display on screen
            cv2.imshow(window_name, annotated_frame)

            # Process key events (1ms delay)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
                logger.info("Exit requested by user.")
                break
            elif key in [ord("p"), ord("P"), 32]:  # 'p' or SPACE
                is_paused = not is_paused
                logger.info(f"Video feed {'PAUSED' if is_paused else 'RESUMED'}")
            elif key in [ord("s"), ord("S")]:  # 's' snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(config.output_dir, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, annotated_frame)
                logger.info(f"Snapshot saved to: {snap_path}")
            elif key in [ord("t"), ord("T")]:  # 't' toggle tracking
                config.enable_tracking = not config.enable_tracking
                detector.config.enable_tracking = config.enable_tracking
                logger.info(f"Vehicle Tracking toggled: {'ENABLED' if config.enable_tracking else 'DISABLED'}")
            elif key in [ord("c"), ord("C")]:  # 'c' toggle counts
                config.show_counts = not config.show_counts
            elif key in [ord("h"), ord("H")]:  # 'h' toggle top HUD
                config.show_hud = not config.show_hud

    except KeyboardInterrupt:
        logger.info("Program interrupted by user (Ctrl+C).")
    finally:
        # Cleanup
        stream.release()
        if writer is not None:
            writer.release()
            logger.info(f"Saved recorded video to {args.save}")
        cv2.destroyAllWindows()
        logger.info("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
