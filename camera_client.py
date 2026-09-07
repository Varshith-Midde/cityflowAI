"""
CityFlow Camera Node Edge Client.
Runs real-time YOLO vehicle detection on a camera feed or video file,
and streams live traffic density, flow metrics, and accident signatures
to the centralized CityFlow FastAPI server.
"""
import argparse
import logging
import sys
import time
import httpx
import cv2
import numpy as np

from config import VehicleDetectionConfig
from src.camera import CameraStream
from src.detector import VehicleDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CameraClient")


def parse_args():
    parser = argparse.ArgumentParser(description="CityFlow Edge Camera Node Client")
    parser.add_argument("--zone", type=str, default="zone_hitech_junction", help="Assigned City Zone ID")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Centralized server base URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Push interval in seconds")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without opening OpenCV GUI window")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cfg = VehicleDetectionConfig()
    cfg.model_name = args.model
    cfg.confidence_threshold = args.conf
    cfg.camera_index = source if isinstance(source, int) else 0

    logger.info(f"Initializing Camera Client for Zone: {args.zone}")
    logger.info(f"Source: {source}, Server: {args.server}")

    # Initialize YOLO Detector
    try:
        detector = VehicleDetector(cfg)
    except Exception as e:
        logger.error(f"Failed to initialize YOLO detector: {e}")
        sys.exit(1)

    # Initialize Video Capture
    camera = CameraStream(source, cfg.frame_width, cfg.frame_height, cfg.fps_target)
    if not camera.start():
        logger.error(f"Could not open camera / video source: {source}")
        sys.exit(1)

    client = httpx.Client(timeout=4.0)
    last_push_time = 0.0

    print("\n" + "=" * 60)
    print(f"  CityFlow Edge Node Active | Zone: {args.zone}")
    print(f"  Streaming telemetry to: {args.server}/api/zones/{args.zone}/update")
    print(f"  Press 'q' in OpenCV window to quit")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                # If video file reached EOF, loop it back
                if isinstance(source, str):
                    camera.stop()
                    camera = CameraStream(source, cfg.frame_width, cfg.frame_height, cfg.fps_target)
                    camera.start()
                    continue
                else:
                    time.sleep(0.01)
                    continue

            # Run YOLO Detection and Tracking
            detections, counts, fps = detector.process_frame(frame)
            total_vehicles = len(detections)
            density_label, density_color = detector.get_traffic_density(total_vehicles)

            # Periodic Push to Central Server
            current_time = time.time()
            if current_time - last_push_time >= args.interval:
                last_push_time = current_time

                # Convert BGR density color to RGB for web dashboard
                rgb_color = (density_color[2], density_color[1], density_color[0])

                payload = {
                    "vehicle_count": total_vehicles,
                    "density_label": density_label,
                    "density_color": list(rgb_color),
                    "class_counts": counts,
                    "inbound_flow": detector.inbound_count,
                    "outbound_flow": detector.outbound_count,
                    "fps": round(fps, 1),
                    "timestamp": current_time
                }

                try:
                    url = f"{args.server}/api/zones/{args.zone}/update"
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        logger.info(f"[{args.zone}] Telemetry Sent -> Vehicles: {total_vehicles} | {density_label} | FPS: {fps:.1f}")
                    else:
                        logger.warning(f"Server returned status {resp.status_code}: {resp.text}")
                except Exception as ex:
                    logger.debug(f"Telemetry push failed: {ex}")

            # Display window unless headless
            if not args.headless:
                # Simple status badge
                cv2.putText(
                    frame,
                    f"Zone: {args.zone} | Vehicles: {total_vehicles} | {density_label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    density_color,
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(f"CityFlow Edge Node - {args.zone}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        client.close()
        logger.info("Camera client terminated cleanly.")


if __name__ == "__main__":
    main()
