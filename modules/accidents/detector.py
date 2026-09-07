"""
AI-Based Real-Time Road Accident Detection & Anomaly System.
Analyzes vehicle tracks, sudden velocity drops, bounding box overlaps,
and stopped vehicles in active travel lanes to detect traffic accidents.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from city_model import CityRoadGraph


@dataclass
class AccidentIncident:
    id: str
    zone_id: str
    zone_name: str
    camera_id: str
    severity: str  # MINOR, MODERATE, CRITICAL
    confidence: float
    description: str
    vehicles_involved: int
    detected_at: float = field(default_factory=time.time)
    is_active: bool = True
    ambulance_dispatched: bool = False
    lanes_blocked: str = "Center & Right Lane"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "camera_id": self.camera_id,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "description": self.description,
            "vehicles_involved": self.vehicles_involved,
            "detected_at": self.detected_at,
            "is_active": self.is_active,
            "ambulance_dispatched": self.ambulance_dispatched,
            "lanes_blocked": self.lanes_blocked,
            "elapsed_min": round((time.time() - self.detected_at) / 60.0, 1)
        }


class AccidentDetectionEngine:
    """Manages accident detection logic and active collision records."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.active_accidents: Dict[str, AccidentIncident] = {}
        self.history: List[AccidentIncident] = []

    def evaluate_cv_telemetry(
        self,
        zone_id: str,
        stopped_vehicles: int,
        proximity_alerts: int,
        avg_speed_drop_pct: float
    ) -> Optional[AccidentIncident]:
        """
        Evaluates computer vision tracking telemetry from camera feeds:
        - stopped_vehicles: count of vehicles stationary in traffic flow
        - proximity_alerts: number of vehicles abnormally close/colliding
        - avg_speed_drop_pct: sudden drop in zone speed
        """
        zone = self.graph.get_zone(zone_id)
        if not zone:
            return None

        # Accident Anomaly Trigger condition
        if proximity_alerts >= 2 or (stopped_vehicles >= 3 and avg_speed_drop_pct > 65.0):
            severity = "CRITICAL" if stopped_vehicles >= 4 else "MODERATE"
            desc = f"Collision anomaly detected: {stopped_vehicles} vehicles halted with sudden speed drop of {int(avg_speed_drop_pct)}%."

            return self.trigger_accident(
                zone_id=zone_id,
                severity=severity,
                vehicles_involved=stopped_vehicles or 2,
                description=desc,
                confidence=0.92
            )
        return None

    def trigger_accident(
        self,
        zone_id: str,
        severity: str = "MODERATE",
        vehicles_involved: int = 2,
        description: str = "Multi-vehicle collision detected by camera node.",
        confidence: float = 0.95
    ) -> Optional[AccidentIncident]:
        """Manually or automatically flags an accident event in a zone."""
        zone = self.graph.get_zone(zone_id)
        if not zone:
            return None

        accident_id = f"ACC-{str(uuid.uuid4())[:6].upper()}"
        incident = AccidentIncident(
            id=accident_id,
            zone_id=zone_id,
            zone_name=zone.name,
            camera_id=zone.camera_id or "CAM-AUTO",
            severity=severity,
            confidence=confidence,
            description=description,
            vehicles_involved=vehicles_involved,
            lanes_blocked="Center & Right Lane" if severity in ["MODERATE", "CRITICAL"] else "Shoulder Only"
        )

        # Update zone state to reflect obstruction
        zone.is_blocked = True
        zone.blockage_reason = f"Accident: {severity} ({accident_id})"
        zone.congestion_factor = min(1.0, zone.congestion_factor + 0.5)

        self.active_accidents[accident_id] = incident
        self.history.append(incident)
        return incident

    def resolve_accident(self, accident_id: str) -> bool:
        """Clears the accident and unblocks the road."""
        incident = self.active_accidents.pop(accident_id, None)
        if not incident:
            return False

        incident.is_active = False
        zone = self.graph.get_zone(incident.zone_id)
        if zone:
            # Check if any other accident exists in the same zone
            has_other = any(a.zone_id == zone.id for a in self.active_accidents.values())
            if not has_other:
                zone.is_blocked = False
                zone.blockage_reason = ""
                zone.congestion_factor = max(0.1, zone.congestion_factor - 0.4)
        return True

    def get_active(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.active_accidents.values()]

    def get_all(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.history]
