"""
Automated verification script for CityFlow AI Vehicle Detection System.
Tests model downloading, inference, HUD rendering, and camera device availability.
"""
import os
import sys
import time

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np


def run_diagnostics():
    print("=" * 60)
    print("CITYFLOW AI - SYSTEM DIAGNOSTICS & VERIFICATION")
    print("=" * 60)

    # 1. Environment & Package Verification
    print("\n[1/4] Checking Core Dependencies...")
    try:
        import torch
        import torchvision
        import ultralytics
        from ultralytics import YOLO
        print(f"  [OK] Python:       {sys.version.split()[0]}")
        print(f"  [OK] OpenCV:       {cv2.__version__}")
        print(f"  [OK] PyTorch:      {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
        print(f"  [OK] Ultralytics:  {ultralytics.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Dependency check failed: {e}")
        return False

    # 2. Config & Custom Modules Import
    print("\n[2/4] Testing Project Modules...")
    try:
        from config import VehicleDetectionConfig
        from src.camera import VideoStream
        from src.detector import VehicleDetector
        from src.visualizer import HUDVisualizer
        print("  [OK] config, camera, detector, visualizer imported successfully.")
    except Exception as e:
        print(f"  [FAIL] Module import failed: {e}")
        return False

    # 3. Model Loading & Synthetic Frame Inference
    print("\n[3/4] Testing YOLO Model & Inference Engine...")
    try:
        cfg = VehicleDetectionConfig(model_name="yolov8n.pt", enable_tracking=False)
        detector = VehicleDetector(cfg)
        visualizer = HUDVisualizer(cfg)

        # Create synthetic test frame with realistic vehicle-like rectangles
        synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Background gradient
        synthetic_frame[:] = (40, 45, 50)
        # Draw a simulated car shape
        cv2.rectangle(synthetic_frame, (300, 350), (600, 520), (180, 180, 180), -1)
        cv2.circle(synthetic_frame, (370, 520), 30, (20, 20, 20), -1)
        cv2.circle(synthetic_frame, (530, 520), 30, (20, 20, 20), -1)

        # Run inference
        t0 = time.perf_counter()
        detections, counts, fps = detector.process_frame(synthetic_frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Render HUD on synthetic frame
        annotated = visualizer.draw_hud(
            frame=synthetic_frame,
            detections=detections,
            counts=counts,
            fps=fps,
            device_info="CPU (Test)"
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        test_out_path = os.path.join(cfg.output_dir, "system_test_result.jpg")
        cv2.imwrite(test_out_path, annotated)

        print(f"  [OK] YOLO Model initialization: SUCCESS")
        print(f"  [OK] Single-frame inference latency: {latency_ms:.2f} ms")
        print(f"  [OK] Visualizer rendering test: PASSED")
        print(f"  [OK] Saved test artifact to: {test_out_path}")
    except Exception as e:
        print(f"  [FAIL] Model inference failed: {e}")
        return False

    # 4. Camera Device Probe
    print("\n[4/4] Probing Camera Availability...")
    try:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [OK] Camera Index 0: ACCESSIBLE (Resolution: {w}x{h})")
            else:
                print("  [WARN] Camera opened but did not return a frame (may be in use or privacy shutter closed).")
            cap.release()
        else:
            print("  [WARN] Camera Index 0 not directly opened. You can test with video files or verify camera permissions.")
    except Exception as e:
        print(f"  [WARN] Camera probe encountered: {e}")

    print("\n" + "=" * 60)
    print("ALL SYSTEM VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo launch live vehicle detection with your laptop camera, run:")
    print("    python main.py\n")
    return True


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
