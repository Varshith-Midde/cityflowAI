# CityFlow AI — Centralized Traffic Intelligence & Smart City Platform 🚗 🏍️ 🚌 🚚 🚑

An enterprise-grade, real-time Computer Vision and Municipal Operations platform that addresses growing pressures on urban resources, transport networks, and logistics infrastructure. CityFlow AI analyzes vehicle density and traffic flow across city corridors to provide **real-time congestion alerts, optimal rider routing, and 9 intelligent smart city operations modules**.

---

## 🌟 9 Smart City Intelligence Modules

1. **🚑 Smart Ambulance Routing**: AI-powered emergency green-wave routing bypassing traffic bottlenecks to reduce hospital arrival times.
2. **🚦 AI Adaptive Traffic Signals**: Demand-responsive signal timing adjusting green phase duration based on computer vision vehicle queue counts.
3. **🚌 Smart Public Transport**: Real-time bus corridor delay monitoring with dynamic detour suggestions around blocked corridors.
4. **🗑️ Smart Waste Collection**: IoT fill-level monitoring with dynamic collection tours that eliminate empty or redundant truck trips.
5. **🌊 Urban Flood Warning System**: Precipitation and ultrasonic water-depth monitoring with automated road closure alerts.
6. **📦 Smart Logistics Route Optimization**: Multi-stop delivery sequencing (VRP) factoring in live congestion to reduce fuel burn and carbon emissions.
7. **💥 Road Accident Detection**: Computer vision velocity anomaly and stopped-vehicle detection in travel lanes with rapid dispatch escalation.
8. **🅿️ Intelligent Smart Parking**: Parking occupancy tracking and nearest-available slot navigation for commuters.
9. **🚨 Integrated Emergency Command Hub**: Consolidated crisis triage system with AI priority scoring (1–10) and multi-agency dispatch logs.

---

## 📁 Project Architecture

```
cityflow AI/
├── config.py                 # Centralized configuration (server, colors, thresholds)
├── city_model.py             # City graph: zones, road edges, Dijkstra router
├── city_zones.json           # 12 interconnected municipal zones with GPS coords
├── server.py                 # Centralized FastAPI + WebSocket server
├── camera_client.py          # Edge camera node bridging YOLO detector -> server
│
├── modules/                  # 9 Smart City Modules
│   ├── ambulance/            # 1. Emergency green-wave routing & mission simulator
│   ├── signals/              # 2. Adaptive AI traffic light network
│   ├── transit/              # 3. Public bus lines & dynamic detours
│   ├── waste/                # 4. Smart garbage bins & green collection tour
│   ├── flood/                # 5. Urban water-level sensors & flood warnings
│   ├── logistics/            # 6. Multi-stop freight VRP optimization
│   ├── accidents/            # 7. CV accident anomaly detection
│   ├── parking/              # 8. Parking occupancy & spot finder
│   └── emergency/            # 9. Multi-agency incident command hub
│
├── dashboard/                # Live Web Dashboard (Leaflet.js + WebSockets)
│   ├── index.html            # Dark-mode dashboard with interactive city map
│   ├── style.css             # Cyber glassmorphism styling
│   └── app.js                # Live streaming telemetry & module controllers
│
├── simulation/               # Autonomous city traffic & sensor simulation
│   └── traffic_sim.py        # Realistic urban traffic flow fluctuations
│
├── src/                      # Core Computer Vision Engine
│   ├── camera.py             # DirectShow camera & video stream capture
│   ├── detector.py           # YOLO inference, ByteTrack, speed & direction
│   └── visualizer.py         # Cyber HUD & bounding box overlay
│
├── ui_app.py                 # Tkinter desktop application (for PC testing)
├── main.py                   # OpenCV direct CLI mode
├── test_smart_city.py        # Automated 10-point municipal verification suite
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Centralized Smart City Platform (Web Dashboard) 🌐
Start the FastAPI backend server:
```bash
python server.py
```
Open your browser to:
```
http://127.0.0.1:8000
```
- **Interactive City Map**: View all 12 zones with live color-coded traffic density (Green, Yellow, Red, Blocked).
- **Rider Route Planner**: Calculate the fastest path between any two zones with congestion bypass.
- **One-Click Simulations**: Test ambulance dispatches, traffic signal rebalancing, storm flooding, or simulated accidents.

---

### 3. Connect a Camera Node to the Central Server 📹
Run the YOLO computer vision detector on your laptop webcam or a video file and stream live vehicle counts into a specific city zone:
```bash
# Stream laptop webcam into Hi-Tech City Junction:
python camera_client.py --zone zone_hitech_junction --source 0

# Or stream a traffic video file:
python camera_client.py --zone zone_cyber_towers --source "path/to/traffic_video.mp4"
```
The central web dashboard will instantly light up and update in real-time as vehicles are detected!

---

### 4. Run the Tkinter Desktop Application 💻
If you want to test vehicle detection locally on your laptop screen with the original desktop UI:
```bash
python ui_app.py
```

---

### 5. Run the Automated Verification Test Suite
Verify that all 9 modules, Dijkstra routing, REST endpoints, and WebSocket services are operating correctly:
```bash
python test_smart_city.py
```

---

## 🎮 REST API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/city` | City boundaries, map center coordinates, and aggregate KPIs |
| `GET` | `/api/zones` | All 12 traffic zones with real-time density, speed, and status |
| `POST` | `/api/zones/{zone_id}/update` | Camera edge node pushes live vehicle telemetry |
| `GET` | `/api/route?from=Z1&to=Z2` | Congestion-aware Dijkstra optimal path for riders |
| `POST` | `/api/ambulance/dispatch` | Dispatches emergency EMS with green-wave signal preemption |
| `POST` | `/api/signals/optimize` | AI dynamic green-phase split rebalancing |
| `POST` | `/api/accidents/trigger` | Reports or simulates a traffic collision and lane closure |
| `GET` | `/api/parking/recommend` | Recommends closest parking garage with available slots |
| `POST` | `/api/transit/optimize` | Recommends bus detours around traffic jams |
| `POST` | `/api/logistics/optimize` | Solves multi-stop delivery sequences minimizing fuel/CO2 |
| `POST` | `/api/waste/collect` | Computes green garbage truck route for bins needing pickup |
| `POST` | `/api/flood/simulate` | Simulates precipitation and tests urban runoff sensors |
| `GET` | `/api/incidents` | Central incident queue prioritized from 1 to 10 |
| `WS` | `/ws/live` | Real-time WebSocket feed pushing state updates every second |
