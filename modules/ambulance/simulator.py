"""
Ambulance Mission Fleet Simulator.
Manages live active emergency dispatches, real-time vehicle GPS progress,
and signals preemption triggers.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from modules.ambulance.router import AmbulanceRouter
from city_model import CityRoadGraph


@dataclass
class AmbulanceMission:
    id: str
    unit_name: str
    pickup_zone_id: str
    hospital_zone_id: str
    status: str  # DISPATCHED, EN_ROUTE, ARRIVED
    route: List[str]
    current_index: int = 0
    current_lat: float = 0.0
    current_lng: float = 0.0
    started_at: float = field(default_factory=time.time)
    estimated_arrival_sec: float = 120.0
    preempt_signal_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "unit_name": self.unit_name,
            "pickup_zone_id": self.pickup_zone_id,
            "hospital_zone_id": self.hospital_zone_id,
            "status": self.status,
            "route": self.route,
            "current_zone_id": self.route[self.current_index] if self.current_index < len(self.route) else self.hospital_zone_id,
            "current_lat": self.current_lat,
            "current_lng": self.current_lng,
            "progress_pct": int((self.current_index / max(1, len(self.route) - 1)) * 100) if len(self.route) > 1 else 100,
            "elapsed_sec": round(time.time() - self.started_at, 1)
        }


class AmbulanceFleetManager:
    """Oversees active emergency ambulance units and automated movement."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.router = AmbulanceRouter(city_graph)
        self.active_missions: Dict[str, AmbulanceMission] = {}

    def dispatch(self, pickup_zone_id: str, hospital_zone_id: Optional[str] = None) -> Optional[AmbulanceMission]:
        plan = self.router.plan_emergency_route(pickup_zone_id, hospital_zone_id)
        if not plan:
            return None

        route_path = plan["route"]["path"]
        if not route_path:
            return None

        mission_id = f"AMB-{str(uuid.uuid4())[:6].upper()}"
        start_zone = self.graph.get_zone(route_path[0])
        hosp_id = plan["destination_hospital"]

        mission = AmbulanceMission(
            id=mission_id,
            unit_name=f"Metro EMS Unit {len(self.active_missions) + 1}",
            pickup_zone_id=pickup_zone_id,
            hospital_zone_id=hosp_id,
            status="EN_ROUTE",
            route=route_path,
            current_index=0,
            current_lat=start_zone.lat if start_zone else 17.4435,
            current_lng=start_zone.lng if start_zone else 78.3772,
            estimated_arrival_sec=plan["route"]["estimated_time_sec"],
            preempt_signal_ids=[s["signal_id"] for s in plan.get("preempt_signals", []) if s.get("signal_id")]
        )

        # Trigger green-wave for upcoming signals along this route
        self._apply_signal_preemption(mission)

        self.active_missions[mission.id] = mission
        return mission

    def step(self) -> None:
        """Advances active ambulances along their route waypoints."""
        for mission_id, mission in list(self.active_missions.items()):
            if mission.status == "ARRIVED":
                continue

            if mission.current_index < len(mission.route) - 1:
                mission.current_index += 1
                curr_zone = self.graph.get_zone(mission.route[mission.current_index])
                if curr_zone:
                    mission.current_lat = curr_zone.lat
                    mission.current_lng = curr_zone.lng

                # Keep green-wave signal active
                self._apply_signal_preemption(mission)
            else:
                mission.status = "ARRIVED"

    def _apply_signal_preemption(self, mission: AmbulanceMission) -> None:
        """Forces GREEN lights at the ambulance's immediate and next intersection."""
        lookahead = mission.route[mission.current_index : mission.current_index + 2]
        for zid in lookahead:
            zone = self.graph.get_zone(zid)
            if zone and zone.signal_id:
                zone.signal_state = "GREEN"
                zone.signal_timer_sec = 45  # Extended green wave

    def get_all(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.active_missions.values()]
