"""
CityFlow AI — Centralized Smart City Intelligence Platform Server.
High-performance FastAPI & WebSocket backend integrating Computer Vision edge nodes,
Adaptive Traffic Signals, Emergency Ambulance Routing, Public Transit,
Logistics Fleets, Smart Waste, Urban Flood Warnings, and Emergency Operations.
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from city_model import CityRoadGraph
from modules.ambulance.simulator import AmbulanceFleetManager
from modules.signals.signal_controller import TrafficSignalNetwork
from modules.accidents.detector import AccidentDetectionEngine
from modules.parking.manager import SmartParkingManager
from modules.transit.route_optimizer import SmartTransitSystem
from modules.logistics.route_planner import SmartLogisticsOptimizer
from modules.waste.bin_simulator import SmartWasteManager
from modules.flood.risk_analyzer import UrbanFloodWarningSystem
from modules.emergency.incident_manager import CentralEmergencyIncidentHub
from simulation.traffic_sim import SmartCitySimulationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CityFlowServer")

# Global Platform Singletons
city_graph = CityRoadGraph("cities_data.json" if os.path.exists("cities_data.json") else "city_zones.json")
ambulance_manager = AmbulanceFleetManager(city_graph)
signal_network = TrafficSignalNetwork(city_graph)
accident_engine = AccidentDetectionEngine(city_graph)
parking_manager = SmartParkingManager(city_graph)
transit_system = SmartTransitSystem(city_graph)
logistics_optimizer = SmartLogisticsOptimizer(city_graph)
waste_manager = SmartWasteManager(city_graph)
flood_system = UrbanFloodWarningSystem(city_graph)
incident_hub = CentralEmergencyIncidentHub(city_graph)
traffic_sim = SmartCitySimulationEngine(city_graph)

# RoadGuard Reports In-Memory Store
roadguard_reports: List[Dict[str, Any]] = [
    {
        "id": "RG-HYD-101",
        "city_id": "hyderabad",
        "title": "Deep Pothole at Hi-Tech Junction Left Lane",
        "description": "Critical asphalt pothole ~15cm deep causing vehicular swerving during rush hour.",
        "damage_type": "Severe Pothole (Grade 4 Hazard)",
        "severity": "CRITICAL",
        "confidence_score": 96.4,
        "status": "REPAIRING",  # REPORTED -> UNDER_REVIEW -> REPAIRING -> RESOLVED
        "zone_id": "zone_hitech_junction",
        "zone_name": "Hi-Tech City Junction",
        "lat": 17.4485,
        "lng": 78.3740,
        "reported_at": "2026-09-05 08:30:15",
        "reported_by": "Citizen Volunteer #412",
        "image_url": "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=600&auto=format&fit=crop&q=80",
        "municipal_action": "Cold-mix asphalt patch dispatched via GHMC Quick-Response Unit #3.",
        "repair_progress_pct": 65,
        "assigned_crew": "GHMC Ward 112 Rapid Repair Team"
    },
    {
        "id": "RG-HYD-102",
        "city_id": "hyderabad",
        "title": "Structural Asphalt Fissure on Cyber Towers Flyover",
        "description": "Fissure along central divider expanding due to heavy bus vibrations.",
        "damage_type": "Structural Asphalt Fissure",
        "severity": "HIGH",
        "confidence_score": 92.1,
        "status": "UNDER_REVIEW",
        "zone_id": "zone_cyber_towers",
        "zone_name": "Cyber Towers Circle",
        "lat": 17.4504,
        "lng": 78.3809,
        "reported_at": "2026-09-05 11:14:22",
        "reported_by": "Citizen Mobile App",
        "image_url": "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=600&auto=format&fit=crop&q=80",
        "municipal_action": "Inspection scheduled by GHMC Civil Engineering Inspector.",
        "repair_progress_pct": 25,
        "assigned_crew": "GHMC Inspection Wing"
    },
    {
        "id": "RG-HYD-103",
        "city_id": "hyderabad",
        "title": "Open Drainage Grate near Madhapur Market",
        "description": "Displaced iron storm drain grate posing severe hazard to two-wheelers.",
        "damage_type": "Hazardous Manhole / Grate Dislodged",
        "severity": "CRITICAL",
        "confidence_score": 98.7,
        "status": "REPORTED",
        "zone_id": "zone_market_square",
        "zone_name": "Madhapur Market Square",
        "lat": 17.4450,
        "lng": 78.3950,
        "reported_at": "2026-09-05 16:45:00",
        "reported_by": "Rider Auto-Telemetry",
        "image_url": "https://images.unsplash.com/photo-1578575437130-527eed3abbec?w=600&auto=format&fit=crop&q=80",
        "municipal_action": "Triage alert pushed to Emergency Command Hub.",
        "repair_progress_pct": 10,
        "assigned_crew": "GHMC Zone Emergency Marshall"
    },
    {
        "id": "RG-BLR-201",
        "city_id": "bengaluru",
        "title": "Crater Pothole near Silk Board Ramp",
        "description": "Multiple connected potholes after monsoon rain causing 400m traffic backlog.",
        "damage_type": "Severe Pothole (Grade 4 Hazard)",
        "severity": "CRITICAL",
        "confidence_score": 97.2,
        "status": "REPAIRING",
        "zone_id": "blr_silk_board",
        "zone_name": "Central Silk Board Junction",
        "lat": 12.9176,
        "lng": 77.6234,
        "reported_at": "2026-09-05 09:20:00",
        "reported_by": "BBMP Commuter Portal",
        "image_url": "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=600&auto=format&fit=crop&q=80",
        "municipal_action": "Python Pothole Recycler machine deployed.",
        "repair_progress_pct": 50,
        "assigned_crew": "BBMP South Road Infrastructure Unit"
    }
]

# Saved Places Store: 1 Home, 1 Work, up to 5 Custom places
saved_places_store: Dict[str, Any] = {
    "home": {
        "id": "place_home",
        "type": "HOME",
        "name": "Home",
        "zone_id": "zone_market_square",
        "zone_name": "Madhapur Market Square",
        "address": "Madhapur Heights, Sector 2",
        "icon": "fa-house"
    },
    "work": {
        "id": "place_work",
        "type": "WORK",
        "name": "Work",
        "zone_id": "zone_it_corridor",
        "zone_name": "Silicon IT Corridor",
        "address": "Tech Park Tower B, 4th Floor",
        "icon": "fa-briefcase"
    },
    "custom": [
        {
            "id": "custom_1",
            "type": "CUSTOM",
            "name": "Gym & Fitness Club",
            "zone_id": "zone_cyber_towers",
            "zone_name": "Cyber Towers Circle",
            "address": "Cyber Heights 2nd Ave",
            "icon": "fa-dumbbell"
        },
        {
            "id": "custom_2",
            "type": "CUSTOM",
            "name": "University Campus",
            "zone_id": "zone_gachibowli_stadium",
            "zone_name": "Gachibowli Stadium Junction",
            "address": "Campus Gate 1, Stadium Road",
            "icon": "fa-graduation-cap"
        }
    ]
}

# Connected WebSocket Clients
active_ws_clients: List[WebSocket] = []


class ConnectionManager:
    """Manages active browser WebSocket sessions for real-time map push."""

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        active_ws_clients.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(active_ws_clients)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in active_ws_clients:
            active_ws_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(active_ws_clients)}")

    async def broadcast(self, message: Dict[str, Any]):
        if not active_ws_clients:
            return
        payload = json.dumps(message)
        to_remove = []
        for client in active_ws_clients:
            try:
                await client.send_text(payload)
            except Exception:
                to_remove.append(client)
        for dead_client in to_remove:
            if dead_client in active_ws_clients:
                active_ws_clients.remove(dead_client)


ws_manager = ConnectionManager()


async def background_city_heartbeat():
    """Periodic 1-second city background clock advancing signals, ambulances, and telemetry."""
    while True:
        try:
            # 1. Step traffic signals
            signal_network.step()
            # 2. Step active ambulances
            ambulance_manager.step()
            # 3. Step waste fill
            waste_manager.step()

            # Compile real-time city snapshot
            city_snapshot = {
                "type": "CITY_STATE_UPDATE",
                "timestamp": asyncio.get_event_loop().time(),
                "city_id": city_graph.active_city_id,
                "city_name": city_graph.city_name,
                "state": getattr(city_graph, "state", "India"),
                "authority": getattr(city_graph, "authority", "Municipal Corporation"),
                "zones": [z.to_dict() for z in city_graph.get_all_zones().values()],
                "signals": signal_network.get_all(),
                "ambulances": ambulance_manager.get_all(),
                "accidents": accident_engine.get_active(),
                "incidents": incident_hub.get_open_incidents(),
                "red_zones": city_graph.get_top_red_zones(limit=3),
                "stats": get_city_kpis()
            }
            await ws_manager.broadcast(city_snapshot)
        except Exception as e:
            logger.error(f"Error in background heartbeat: {e}")

        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing CityFlow AI Smart City Server...")
    traffic_sim.start()
    task = asyncio.create_task(background_city_heartbeat())
    yield
    logger.info("Shutting down CityFlow AI Server...")
    traffic_sim.stop()
    task.cancel()


app = FastAPI(
    title="CityFlow AI — Centralized Traffic Intelligence Platform",
    description="Centralized REST and WebSocket API for real-time traffic density, smart routing, and 9 municipal intelligence modules.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_city_kpis() -> Dict[str, Any]:
    """Aggregates high-level city operational KPIs."""
    zones = list(city_graph.get_all_zones().values())
    total_vehicles = sum(z.vehicle_count for z in zones)
    congested = sum(1 for z in zones if "HEAVY" in z.density_label or z.is_blocked)
    avg_speed = sum(z.avg_speed_kmh for z in zones) / max(1, len(zones))

    # RoadGuard active report count
    active_rg = [r for r in roadguard_reports if r.get("status") != "RESOLVED"]
    critical_rg = [r for r in active_rg if r.get("severity") in ["CRITICAL", "HIGH"]]

    return {
        "city_id": getattr(city_graph, "active_city_id", "hyderabad"),
        "city_name": city_graph.city_name,
        "state": getattr(city_graph, "state", "Telangana"),
        "authority": getattr(city_graph, "authority", "Greater Hyderabad Municipal Corporation (GHMC)"),
        "total_active_vehicles": total_vehicles,
        "congested_zones_count": congested,
        "total_zones": len(zones),
        "congestion_rate_pct": round((congested / max(1, len(zones))) * 100, 1),
        "city_avg_speed_kmh": round(avg_speed, 1),
        "active_accidents": len(accident_engine.get_active()),
        "open_incidents": len(incident_hub.get_open_incidents()),
        "active_ambulances": len(ambulance_manager.get_all()),
        "active_roadguard_reports": len(active_rg),
        "critical_road_hazards": len(critical_rg)
    }


# ==========================================================
# REST API ENDPOINTS: MULTI-CITY & PLATFORM METADATA
# ==========================================================

@app.get("/api/cities")
def get_all_cities():
    """Returns all supported Indian cities and municipal authorities."""
    return city_graph.get_available_cities()


class CitySwitchRequest(BaseModel):
    city_id: str


@app.post("/api/city/switch")
def switch_active_city(req: CitySwitchRequest):
    """Dynamically switches active city road network."""
    success = city_graph.load_city(req.city_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"City '{req.city_id}' not found in catalog")

    return {
        "status": "switched",
        "city_id": city_graph.active_city_id,
        "city_name": city_graph.city_name,
        "authority": city_graph.authority,
        "state": city_graph.state,
        "center": {"lat": city_graph.center_lat, "lng": city_graph.center_lng, "zoom": city_graph.zoom},
        "stats": get_city_kpis()
    }


@app.get("/api/city")
def get_city_info(city_id: Optional[str] = None):
    """Returns city metadata, map center, authority, and high-level boundaries."""
    if city_id and city_id != city_graph.active_city_id:
        city_graph.load_city(city_id)

    return {
        "city_id": city_graph.active_city_id,
        "city_name": city_graph.city_name,
        "authority": getattr(city_graph, "authority", "Municipal Corporation"),
        "state": getattr(city_graph, "state", "India"),
        "center": {"lat": city_graph.center_lat, "lng": city_graph.center_lng, "zoom": city_graph.zoom},
        "red_zones": city_graph.get_top_red_zones(limit=3),
        "stats": get_city_kpis()
    }


# ==========================================================
# REST API ENDPOINTS: SMART TRAFFIC, PREDICTION & 3 RED ZONES
# ==========================================================

@app.get("/api/traffic/summary")
def get_traffic_summary():
    """Returns live city traffic summary and strictly top 3 Red Zones."""
    return {
        "city_id": city_graph.active_city_id,
        "city_name": city_graph.city_name,
        "authority": city_graph.authority,
        "stats": get_city_kpis(),
        "red_zones": city_graph.get_top_red_zones(limit=3)
    }


@app.get("/api/traffic/red-zones")
def get_traffic_red_zones():
    """Returns strictly the top 3 highest congested Red Zones."""
    return city_graph.get_top_red_zones(limit=3)


@app.get("/api/traffic/predict")
def predict_traffic(
    from_zone: str = Query(..., alias="from"),
    to_zone: str = Query(..., alias="to")
):
    """AI Traffic Prediction model forecasting 15m, 30m, and 60m congestion trends."""
    return city_graph.predict_traffic(from_zone, to_zone)


@app.get("/api/routes/alternatives")
def get_alternative_routes(
    from_zone: str = Query(..., alias="from"),
    to_zone: str = Query(..., alias="to")
):
    """Calculates primary fastest route plus 2 alternative routes with Red Zone avoidance."""
    routes = city_graph.calculate_alternative_routes(from_zone, to_zone)
    if not routes:
        raise HTTPException(status_code=404, detail="No viable routes between specified zones")
    return {
        "from_zone": from_zone,
        "to_zone": to_zone,
        "routes": routes,
        "red_zones": city_graph.get_top_red_zones(limit=3)
    }


# ==========================================================
# REST API ENDPOINTS: SAVED PLACES (Home, Work, Custom x5)
# ==========================================================

class SavedPlaceRequest(BaseModel):
    type: str  # HOME, WORK, CUSTOM
    name: str
    zone_id: str
    zone_name: Optional[str] = ""
    address: Optional[str] = ""
    icon: Optional[str] = "fa-location-dot"


@app.get("/api/saved-places")
def get_saved_places():
    """Returns user's saved places (1 Home, 1 Work, and up to 5 Custom)."""
    return saved_places_store


@app.post("/api/saved-places")
def save_or_update_place(req: SavedPlaceRequest):
    """Saves or updates a place. Validates 1 Home, 1 Work, max 5 Custom places."""
    p_type = req.type.upper()
    zone = city_graph.get_zone(req.zone_id)
    zone_name = zone.name if zone else (req.zone_name or req.zone_id)

    if p_type == "HOME":
        saved_places_store["home"] = {
            "id": "place_home",
            "type": "HOME",
            "name": req.name or "Home",
            "zone_id": req.zone_id,
            "zone_name": zone_name,
            "address": req.address or f"{zone_name}, Sector 1",
            "icon": "fa-house"
        }
        return {"status": "saved", "place": saved_places_store["home"]}
    elif p_type == "WORK":
        saved_places_store["work"] = {
            "id": "place_work",
            "type": "WORK",
            "name": req.name or "Work",
            "zone_id": req.zone_id,
            "zone_name": zone_name,
            "address": req.address or f"{zone_name}, Commercial Complex",
            "icon": "fa-briefcase"
        }
        return {"status": "saved", "place": saved_places_store["work"]}
    else:  # CUSTOM
        if len(saved_places_store.get("custom", [])) >= 5:
            raise HTTPException(status_code=400, detail="Maximum of 5 custom saved places reached.")
        
        place_id = f"custom_{int(time.time() * 1000)}"
        new_place = {
            "id": place_id,
            "type": "CUSTOM",
            "name": req.name,
            "zone_id": req.zone_id,
            "zone_name": zone_name,
            "address": req.address or zone_name,
            "icon": req.icon or "fa-location-dot"
        }
        saved_places_store.setdefault("custom", []).append(new_place)
        return {"status": "added", "place": new_place}


@app.delete("/api/saved-places/{place_id}")
def delete_custom_place(place_id: str):
    """Deletes a custom saved place."""
    custom_list = saved_places_store.get("custom", [])
    initial_len = len(custom_list)
    saved_places_store["custom"] = [p for p in custom_list if p["id"] != place_id]
    if len(saved_places_store["custom"]) == initial_len:
        raise HTTPException(status_code=404, detail="Place ID not found")
    return {"status": "deleted", "place_id": place_id}


# ==========================================================
# REST API ENDPOINTS: ROADGUARD (Citizen Reporting & AI Triage)
# ==========================================================

class CitizenReportRequest(BaseModel):
    title: str
    description: str
    zone_id: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    photo_url: Optional[str] = None
    damage_type: Optional[str] = None
    reported_by: Optional[str] = "Citizen App"
    city_id: Optional[str] = None


@app.get("/api/roadguard/reports")
def get_roadguard_reports(
    city_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None
):
    """Returns citizen road damage reports with filtering."""
    results = roadguard_reports
    if city_id:
        results = [r for r in results if r.get("city_id") == city_id]
    if status:
        results = [r for r in results if r.get("status") == status]
    if severity:
        results = [r for r in results if r.get("severity") == severity]
    return results


@app.post("/api/roadguard/report")
def submit_citizen_report(req: CitizenReportRequest):
    """
    Submits a citizen damaged road report.
    Automatically runs AI Damage Detection analyzing severity, type, and confidence.
    """
    city_id = req.city_id or city_graph.active_city_id

    # Resolve Zone & Coordinates
    zone = city_graph.get_zone(req.zone_id) if req.zone_id else None
    if not zone and city_graph.zones:
        # Fallback to nearest or first zone
        zone = next(iter(city_graph.zones.values()))

    zone_id = zone.id if zone else "zone_hitech_junction"
    zone_name = zone.name if zone else "City Highway Corridor"
    lat = req.lat if req.lat is not None else (zone.lat if zone else city_graph.center_lat)
    lng = req.lng if req.lng is not None else (zone.lng if zone else city_graph.center_lng)

    # AI Damage Analysis Simulation
    d_type = req.damage_type or "Pothole (Grade 4 Hazard)"
    if "crack" in req.description.lower() or "crack" in (req.damage_type or "").lower():
        d_type = "Structural Asphalt Fissure"
        severity = "HIGH"
        confidence = 93.4
    elif "manhole" in req.description.lower() or "grate" in req.description.lower():
        d_type = "Hazardous Manhole / Grate Dislodged"
        severity = "CRITICAL"
        confidence = 98.2
    elif "subsidence" in req.description.lower() or "cave" in req.description.lower():
        d_type = "Road Subsidence & Cavitation"
        severity = "CRITICAL"
        confidence = 96.8
    elif "water" in req.description.lower() or "flood" in req.description.lower():
        d_type = "Asphalt Erosion & Waterlogging"
        severity = "MODERATE"
        confidence = 89.5
    else:
        d_type = "Severe Pothole (Grade 4 Hazard)"
        severity = "CRITICAL"
        confidence = 95.7

    report_id = f"RG-{city_id[:3].upper()}-{len(roadguard_reports) + 101}"
    now_str = time.time()
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sample_photos = [
        "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1578575437130-527eed3abbec?w=600&auto=format&fit=crop&q=80"
    ]
    img = req.photo_url or sample_photos[len(roadguard_reports) % len(sample_photos)]

    new_report = {
        "id": report_id,
        "city_id": city_id,
        "title": req.title,
        "description": req.description,
        "damage_type": d_type,
        "severity": severity,
        "confidence_score": confidence,
        "status": "REPORTED",  # REPORTED -> UNDER_REVIEW -> REPAIRING -> RESOLVED
        "zone_id": zone_id,
        "zone_name": zone_name,
        "lat": lat,
        "lng": lng,
        "reported_at": date_str,
        "reported_by": req.reported_by or "Citizen Mobile App",
        "image_url": img,
        "municipal_action": f"Auto-escalated to {city_graph.authority} Rapid Response Dispatch.",
        "repair_progress_pct": 5,
        "assigned_crew": f"{city_graph.authority} Maintenance Wing"
    }

    roadguard_reports.insert(0, new_report)

    # Log in Central Emergency Hub if Critical
    if severity in ["CRITICAL", "HIGH"]:
        incident_hub.create_incident(
            source_module="ROADGUARD",
            title=f"Road Hazard ({severity}): {req.title}",
            zone_id=zone_id,
            severity_level=severity,
            description=f"AI confirmed {d_type} (Confidence {confidence}%). {req.description}",
            auto_actions=["Alerted Municipal Ward Team", "Emergency Route Advisory Issued"]
        )

    return {"status": "created", "report": new_report}


class RoadGuardStatusUpdate(BaseModel):
    status: str  # REPORTED, UNDER_REVIEW, REPAIRING, RESOLVED
    municipal_action: Optional[str] = None
    assigned_crew: Optional[str] = None
    repair_progress_pct: Optional[int] = None


@app.post("/api/roadguard/reports/{report_id}/status")
def update_roadguard_status(report_id: str, req: RoadGuardStatusUpdate):
    """Updates the municipal repair status of a RoadGuard report."""
    report = next((r for r in roadguard_reports if r["id"] == report_id), None)
    if not report:
        raise HTTPException(status_code=404, detail="RoadGuard report not found")

    report["status"] = req.status.upper()
    if req.municipal_action:
        report["municipal_action"] = req.municipal_action
    if req.assigned_crew:
        report["assigned_crew"] = req.assigned_crew

    if req.repair_progress_pct is not None:
        report["repair_progress_pct"] = req.repair_progress_pct
    else:
        if report["status"] == "REPORTED":
            report["repair_progress_pct"] = 10
        elif report["status"] == "UNDER_REVIEW":
            report["repair_progress_pct"] = 35
        elif report["status"] == "REPAIRING":
            report["repair_progress_pct"] = 70
        elif report["status"] == "RESOLVED":
            report["repair_progress_pct"] = 100

    return {"status": "updated", "report": report}


@app.get("/api/roadguard/alerts")
def get_roadguard_emergency_alerts(route_zones: Optional[str] = None):
    """
    Checks if critical road damage exists on a user's frequent route or city.
    Returns active high-priority road alerts with detours.
    """
    zones_filter = route_zones.split(",") if route_zones else []
    active_alerts = []

    for r in roadguard_reports:
        if r.get("status") != "RESOLVED" and r.get("severity") in ["CRITICAL", "HIGH"]:
            if not zones_filter or r.get("zone_id") in zones_filter:
                active_alerts.append({
                    "id": r["id"],
                    "title": r["title"],
                    "damage_type": r["damage_type"],
                    "severity": r["severity"],
                    "zone_name": r["zone_name"],
                    "zone_id": r["zone_id"],
                    "status": r["status"],
                    "alert_message": f"⚠️ Commute Alert: {r['damage_type']} reported at {r['zone_name']}! Detour recommended."
                })

    return {"active_alerts": active_alerts, "total_active": len(active_alerts)}


@app.get("/api/zones")
def get_all_zones():
    """Returns all traffic zones and their live state."""
    return [z.to_dict() for z in city_graph.get_all_zones().values()]


@app.get("/api/zones/{zone_id}")
def get_single_zone(zone_id: str):
    zone = city_graph.get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone.to_dict()


class ZoneTelemetryUpdate(BaseModel):
    vehicle_count: int
    density_label: str
    density_color: Optional[List[int]] = None
    class_counts: Optional[Dict[str, int]] = None
    inbound_flow: Optional[int] = 0
    outbound_flow: Optional[int] = 0
    fps: Optional[float] = None


@app.post("/api/zones/{zone_id}/update")
def update_zone_from_camera(zone_id: str, data: ZoneTelemetryUpdate):
    """Called by camera edge nodes to push live vehicle detection telemetry."""
    color = tuple(data.density_color) if data.density_color and len(data.density_color) == 3 else (56, 239, 125)
    success = city_graph.update_zone_telemetry(
        zone_id=zone_id,
        vehicle_count=data.vehicle_count,
        density_label=data.density_label,
        density_color=color,
        inbound=data.inbound_flow or 0,
        outbound=data.outbound_flow or 0
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    return {"status": "success", "zone_id": zone_id, "updated": True}


# Module: Smart Routing for Riders & Commuters
@app.get("/api/route")
def calculate_optimal_route(
    from_zone: str = Query(..., alias="from"),
    to_zone: str = Query(..., alias="to"),
    emergency: bool = Query(False)
):
    """Calculates congestion-aware optimal path for riders or emergency vehicles."""
    route = city_graph.calculate_route(from_zone, to_zone, is_emergency=emergency)
    if not route:
        raise HTTPException(status_code=404, detail="No viable route could be found between specified zones")
    return route.to_dict()


# Module 1: Smart Ambulance Routing
class AmbulanceDispatchRequest(BaseModel):
    pickup_zone_id: str
    hospital_zone_id: Optional[str] = None


@app.post("/api/ambulance/dispatch")
def dispatch_ambulance(req: AmbulanceDispatchRequest):
    mission = ambulance_manager.dispatch(req.pickup_zone_id, req.hospital_zone_id)
    if not mission:
        raise HTTPException(status_code=400, detail="Could not route ambulance to requested location")

    # Log in central incident hub
    incident_hub.create_incident(
        source_module="AMBULANCE",
        title=f"Emergency EMS Dispatch to {mission.pickup_zone_id}",
        zone_id=req.pickup_zone_id,
        severity_level="CRITICAL",
        description=f"Unit {mission.unit_name} dispatched via green-wave corridor.",
        auto_actions=[f"Green-wave signals activated on route: {', '.join(mission.route)}"]
    )
    return mission.to_dict()


@app.get("/api/ambulance/active")
def get_active_ambulances():
    return ambulance_manager.get_all()


# Module 2: AI Traffic Signal Optimization
@app.get("/api/signals")
def get_all_signals():
    return signal_network.get_all()


@app.post("/api/signals/optimize")
def trigger_signals_optimization():
    actions = signal_network.optimize_signals()
    return {"status": "optimized", "adjustments": actions}


@app.post("/api/signals/{signal_id}/preempt")
def preempt_signal(signal_id: str, phase: str = "GREEN", duration_sec: int = 45):
    success = signal_network.set_preemption(signal_id, phase, duration_sec)
    if not success:
        raise HTTPException(status_code=404, detail="Signal not found")
    return {"status": "preempted", "signal_id": signal_id, "phase": phase, "duration_sec": duration_sec}


# Module 7: Accident Detection
class AccidentTriggerRequest(BaseModel):
    zone_id: str
    severity: Optional[str] = "MODERATE"
    vehicles_involved: Optional[int] = 2
    description: Optional[str] = "Multi-vehicle collision reported"


@app.get("/api/accidents")
def get_accidents():
    return {
        "active": accident_engine.get_active(),
        "all": accident_engine.get_all()
    }


@app.post("/api/accidents/trigger")
def trigger_accident(req: AccidentTriggerRequest):
    acc = accident_engine.trigger_accident(
        zone_id=req.zone_id,
        severity=req.severity or "MODERATE",
        vehicles_involved=req.vehicles_involved or 2,
        description=req.description or "Collision detected"
    )
    if not acc:
        raise HTTPException(status_code=404, detail="Zone not found")

    # Raise emergency incident and automatically dispatch ambulance if severe
    auto_actions = [f"Road blocked at {acc.zone_name}"]
    if acc.severity in ["MODERATE", "CRITICAL"]:
        amb = ambulance_manager.dispatch(req.zone_id)
        if amb:
            acc.ambulance_dispatched = True
            auto_actions.append(f"Dispatched EMS Unit {amb.unit_name}")

    incident_hub.create_incident(
        source_module="ACCIDENT",
        title=f"Traffic Collision ({acc.severity}) at {acc.zone_name}",
        zone_id=req.zone_id,
        severity_level="CRITICAL" if acc.severity == "CRITICAL" else "HIGH",
        description=acc.description,
        auto_actions=auto_actions
    )
    return acc.to_dict()


@app.post("/api/accidents/{accident_id}/resolve")
def resolve_accident(accident_id: str):
    success = accident_engine.resolve_accident(accident_id)
    if not success:
        raise HTTPException(status_code=404, detail="Accident record not found")
    return {"status": "resolved", "accident_id": accident_id}


# Module 8: Smart Parking
@app.get("/api/parking")
def get_parking_facilities():
    return parking_manager.get_all()


@app.get("/api/parking/recommend")
def recommend_parking(to_zone: str = Query(..., alias="destination")):
    rec = parking_manager.find_nearest_available(to_zone)
    if not rec:
        raise HTTPException(status_code=404, detail="No parking available near destination")
    return rec


# Module 3: Smart Public Transport
@app.get("/api/transit")
def get_transit_corridors():
    return transit_system.get_all()


@app.post("/api/transit/optimize")
def optimize_transit():
    recs = transit_system.optimize_transit_network()
    return {"status": "optimized", "recommendations": recs}


# Module 6: Smart Logistics Route Optimization
@app.get("/api/logistics")
def get_logistics_fleet():
    return logistics_optimizer.get_all()


@app.post("/api/logistics/optimize")
def optimize_logistics():
    plans = logistics_optimizer.optimize_all_fleet_routes()
    return {"status": "optimized", "fleet_plans": plans}


# Module 4: Smart Waste Collection
@app.get("/api/waste")
def get_waste_bins():
    return waste_manager.get_all()


@app.post("/api/waste/collect")
def plan_waste_collection():
    plan = waste_manager.plan_collection_route()
    return plan


@app.post("/api/waste/empty/{bin_id}")
def empty_waste_bin(bin_id: str):
    success = waste_manager.empty_bin(bin_id)
    if not success:
        raise HTTPException(status_code=404, detail="Bin not found")
    return {"status": "emptied", "bin_id": bin_id}


# Module 5: Urban Flood Warning
class FloodSimRequest(BaseModel):
    rainfall_rate_mmh: float


@app.get("/api/flood")
def get_flood_sensors():
    return flood_system.get_all()


@app.post("/api/flood/simulate")
def simulate_rainfall(req: FloodSimRequest):
    alerts = flood_system.set_rainfall(req.rainfall_rate_mmh)
    for alert in alerts:
        if alert["level"] == "CRITICAL":
            incident_hub.create_incident(
                source_module="FLOOD",
                title=f"Severe Water Inundation at {alert['zone_name']}",
                zone_id=alert["zone_id"],
                severity_level="CRITICAL",
                description=alert["message"],
                auto_actions=["Road closed", "Rerouting public transit and logistics"]
            )
    return {"rainfall_rate_mmh": req.rainfall_rate_mmh, "alerts": alerts}


@app.post("/api/flood/clear")
def clear_flood():
    flood_system.clear_flood()
    return {"status": "cleared"}


# Module 9: Central Incident Management
@app.get("/api/incidents")
def get_all_incidents():
    return {
        "open": incident_hub.get_open_incidents(),
        "all": incident_hub.get_all()
    }


@app.post("/api/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: str):
    success = incident_hub.resolve_incident(incident_id)
    if not success:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "resolved", "incident_id": incident_id}


# Simulation Traffic Intensity Control
@app.post("/api/simulation/traffic")
def set_simulation_intensity(multiplier: float = Query(1.0, ge=0.1, le=3.0)):
    traffic_sim.traffic_intensity = multiplier
    return {"status": "updated", "traffic_intensity": multiplier}


# ==========================================================
# WEBSOCKET REAL-TIME STREAMING
# ==========================================================

@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send initial snapshot immediately upon connection
        snapshot = {
            "type": "INITIAL_CITY_SNAPSHOT",
            "city_id": city_graph.active_city_id,
            "city": city_graph.city_name,
            "state": getattr(city_graph, "state", "India"),
            "authority": getattr(city_graph, "authority", "Municipal Corporation"),
            "center": {"lat": city_graph.center_lat, "lng": city_graph.center_lng, "zoom": city_graph.zoom},
            "zones": [z.to_dict() for z in city_graph.get_all_zones().values()],
            "signals": signal_network.get_all(),
            "ambulances": ambulance_manager.get_all(),
            "accidents": accident_engine.get_active(),
            "incidents": incident_hub.get_open_incidents(),
            "red_zones": city_graph.get_top_red_zones(limit=3),
            "stats": get_city_kpis()
        }
        await websocket.send_text(json.dumps(snapshot))

        while True:
            # Keep socket alive and handle incoming client commands
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("command")
            if cmd == "PING":
                await websocket.send_text(json.dumps({"type": "PONG"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        ws_manager.disconnect(websocket)


# Mount Web Dashboard Static Directory
dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
if not os.path.exists(dashboard_dir):
    os.makedirs(dashboard_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
