"""
Demo Traffic Video Generator
Generates a realistic animated multi-lane highway scene with moving vehicles
(cars, trucks, buses, motorcycles) for offline testing of the CityFlow AI system.
"""
import os
import cv2
import numpy as np


def create_traffic_demo_video(
    output_path: str = "outputs/sample_traffic.mp4",
    num_frames: int = 300,
    width: int = 1280,
    height: int = 720,
    fps: int = 30
):
    """Creates a synthetic multi-lane traffic test video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Vehicle definitions: (x, y, speed, width, height, color, type_label)
    vehicles = [
        {"x": 100, "y": 280, "speed": 4.5, "w": 120, "h": 55, "color": (220, 60, 40), "name": "car"},
        {"x": 500, "y": 290, "speed": 3.8, "w": 130, "h": 58, "color": (50, 180, 240), "name": "car"},
        {"x": 900, "y": 275, "speed": 5.0, "w": 115, "h": 52, "color": (240, 240, 240), "name": "car"},
        {"x": 250, "y": 380, "speed": 3.2, "w": 220, "h": 80, "color": (60, 120, 200), "name": "truck"},
        {"x": 750, "y": 375, "speed": 3.5, "w": 240, "h": 85, "color": (40, 160, 80), "name": "bus"},
        {"x": 50, "y": 500, "speed": 5.5, "w": 110, "h": 50, "color": (200, 50, 180), "name": "car"},
        {"x": 420, "y": 510, "speed": 6.0, "w": 65, "h": 35, "color": (30, 30, 30), "name": "motorcycle"},
        {"x": 820, "y": 495, "speed": 4.8, "w": 135, "h": 60, "color": (230, 190, 50), "name": "car"},
        {"x": 1100, "y": 505, "speed": 5.2, "w": 125, "h": 55, "color": (180, 180, 190), "name": "car"},
    ]

    for frame_idx in range(num_frames):
        # Base background - asphalt highway
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (55, 60, 65)  # Asphalt dark gray

        # Top and bottom green shoulders
        frame[0:200, :] = (45, 90, 50)     # Top grass
        frame[620:720, :] = (45, 90, 50)   # Bottom grass

        # Highway borders (guard rails)
        cv2.line(frame, (0, 200), (width, 200), (200, 200, 200), 5)
        cv2.line(frame, (0, 620), (width, 620), (200, 200, 200), 5)

        # Lane dividers (dashed lines)
        dash_offset = int((frame_idx * 12) % 60)
        for lane_y in [340, 460]:
            for x in range(-dash_offset, width + 60, 60):
                cv2.line(frame, (x, lane_y), (x + 35, lane_y), (240, 240, 240), 3)

        # Center yellow continuous double lane
        cv2.line(frame, (0, 204), (width, 204), (0, 215, 255), 3)
        cv2.line(frame, (0, 616), (width, 616), (0, 215, 255), 3)

        # Render each vehicle
        for v in vehicles:
            vx = int(v["x"])
            vy = int(v["y"])
            vw = v["w"]
            vh = v["h"]
            col = v["color"]

            # Vehicle shadow
            cv2.rectangle(frame, (vx + 5, vy + 5), (vx + vw + 5, vy + vh + 5), (30, 32, 35), -1)

            # Vehicle Body
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), col, -1)
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (30, 30, 30), 2)

            # Windshields & Windows
            if v["name"] in ["car", "bus", "truck"]:
                win_col = (210, 230, 245)
                # Front windshield
                cv2.rectangle(frame, (vx + int(vw * 0.7), vy + 5), (vx + int(vw * 0.88), vy + vh - 5), win_col, -1)
                # Rear window
                cv2.rectangle(frame, (vx + int(vw * 0.12), vy + 5), (vx + int(vw * 0.28), vy + vh - 5), win_col, -1)
                # Roof
                cv2.rectangle(frame, (vx + int(vw * 0.28), vy + 8), (vx + int(vw * 0.7), vy + vh - 8), col, -1)

            # Headlights & Taillights
            cv2.circle(frame, (vx + vw - 2, vy + 8), 4, (100, 255, 255), -1)
            cv2.circle(frame, (vx + vw - 2, vy + vh - 8), 4, (100, 255, 255), -1)
            cv2.circle(frame, (vx + 2, vy + 8), 3, (0, 0, 255), -1)
            cv2.circle(frame, (vx + 2, vy + vh - 8), 3, (0, 0, 255), -1)

            # Wheels
            cv2.rectangle(frame, (vx + int(vw * 0.15), vy - 3), (vx + int(vw * 0.35), vy + 2), (20, 20, 20), -1)
            cv2.rectangle(frame, (vx + int(vw * 0.65), vy - 3), (vx + int(vw * 0.85), vy + 2), (20, 20, 20), -1)
            cv2.rectangle(frame, (vx + int(vw * 0.15), vy + vh - 2), (vx + int(vw * 0.35), vy + vh + 3), (20, 20, 20), -1)
            cv2.rectangle(frame, (vx + int(vw * 0.65), vy + vh - 2), (vx + int(vw * 0.85), vy + vh + 3), (20, 20, 20), -1)

            # Move vehicle
            v["x"] += v["speed"]
            if v["x"] > width + 100:
                v["x"] = -vw - 50

        writer.write(frame)

    writer.release()
    print(f"Generated demo traffic video: {output_path} ({num_frames} frames @ {fps} fps)")
    return output_path


if __name__ == "__main__":
    create_traffic_demo_video()
