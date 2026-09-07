"""
Centralized AI-Powered Emergency Incident Management System.
Aggregates alerts from Accidents, Urban Floods, Severe Congestion, Signal Failures,
and Waste Overflow Hazards. Assigns triage priority scores and coordinates multi-agency dispatches.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from city_model import CityRoadGraph


@dataclass
class IncidentReport:
    id: str
    source_module: str  # ACCIDENT, FLOOD, TRAFFIC, SIGNAL, WASTE
    title: str
    zone_id: str
    zone_name: str
    severity_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    priority_score: int  # 1 (lowest) to 10 (highest emergency)
    description: str
    status: str = "OPEN"  # OPEN, IN_PROGRESS, RESOLVED
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    actions_dispatched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_module": self.source_module,
            "title": self.title,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "severity_level": self.severity_level,
            "priority_score": self.priority_score,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "elapsed_min": round((time.time() - self.created_at) / 60.0, 1),
            "actions_dispatched": self.actions_dispatched
        }


class CentralEmergencyIncidentHub:
    """Triage and response command center for municipal emergency operations."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.incidents: Dict[str, IncidentReport] = {}
        self.history: List[IncidentReport] = []

    def create_incident(
        self,
        source_module: str,
        title: str,
        zone_id: str,
        severity_level: str = "HIGH",
        description: str = "",
        auto_actions: Optional[List[str]] = None
    ) -> IncidentReport:
        zone = self.graph.get_zone(zone_id)
        zname = zone.name if zone else zone_id

        # Calculate Priority Score (1 - 10)
        score_map = {"LOW": 3, "MEDIUM": 5, "HIGH": 8, "CRITICAL": 10}
        base_score = score_map.get(severity_level.upper(), 5)

        # Higher priority if near hospital or major transit hub
        if zone and zone.type in ["hospital", "intersection", "transit_hub"]:
            base_score = min(10, base_score + 1)

        inc_id = f"INC-{str(uuid.uuid4())[:6].upper()}"
        report = IncidentReport(
            id=inc_id,
            source_module=source_module.upper(),
            title=title,
            zone_id=zone_id,
            zone_name=zname,
            severity_level=severity_level.upper(),
            priority_score=base_score,
            description=description,
            actions_dispatched=auto_actions or []
        )

        self.incidents[inc_id] = report
        self.history.append(report)
        return report

    def resolve_incident(self, incident_id: str, resolution_notes: str = "Resolved by operator") -> bool:
        report = self.incidents.get(incident_id)
        if not report:
            return False
        report.status = "RESOLVED"
        report.resolved_at = time.time()
        report.actions_dispatched.append(resolution_notes)
        return True

    def get_open_incidents(self) -> List[Dict[str, Any]]:
        # Sort by priority score descending
        sorted_reports = sorted(
            [inc for inc in self.incidents.values() if inc.status != "RESOLVED"],
            key=lambda x: x.priority_score,
            reverse=True
        )
        return [r.to_dict() for r in sorted_reports]

    def get_all(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.history]
