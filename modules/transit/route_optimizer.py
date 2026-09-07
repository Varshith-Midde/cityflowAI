"""
AI-Based Public Bus & Transit Route Optimization System.
Monitors commuter demand, schedules, and road delays to optimize
bus dispatch frequency, headway spacing, and detour routing around traffic bottlenecks.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from city_model import CityRoadGraph


@dataclass
class BusStop:
    zone_id: str
    name: str
    waiting_passengers: int = 15
    scheduled_interval_min: int = 10


@dataclass
class BusLine:
    id: str
    name: str
    route_zones: List[str]
    active_buses: int = 4
    current_delay_min: float = 0.0
    status: str = "ON_TIME"  # ON_TIME, DELAYED, REROUTING
    stops: List[BusStop] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "route_zones": self.route_zones,
            "active_buses": self.active_buses,
            "current_delay_min": round(self.current_delay_min, 1),
            "status": self.status,
            "stops": [
                {"zone_id": s.zone_id, "name": s.name, "waiting_passengers": s.waiting_passengers}
                for s in self.stops
            ]
        }


class SmartTransitSystem:
    """Manages public bus routes, dynamic detour routing, and schedule optimization."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.bus_lines: Dict[str, BusLine] = {}
        self._init_routes()

    def _init_routes(self) -> None:
        """Initializes primary bus transit corridors."""
        # Line 101: IT Metro Express
        line_1 = BusLine(
            id="BUS-101",
            name="Route 101: Cyber IT Express",
            route_zones=[
                "zone_residential_avenue",
                "zone_it_corridor",
                "zone_hitech_junction",
                "zone_cyber_towers",
                "zone_metro_station"
            ],
            active_buses=5
        )
        line_1.stops = [
            BusStop(zid, self.graph.get_zone(zid).name if self.graph.get_zone(zid) else zid)
            for zid in line_1.route_zones
        ]
        self.bus_lines[line_1.id] = line_1

        # Line 204: Medical & Market Circular
        line_2 = BusLine(
            id="BUS-204",
            name="Route 204: Metro Hospital & Market Loop",
            route_zones=[
                "zone_metro_station",
                "zone_central_hospital",
                "zone_emergency_hq",
                "zone_residential_avenue",
                "zone_lake_road",
                "zone_market_square"
            ],
            active_buses=3
        )
        line_2.stops = [
            BusStop(zid, self.graph.get_zone(zid).name if self.graph.get_zone(zid) else zid)
            for zid in line_2.route_zones
        ]
        self.bus_lines[line_2.id] = line_2

    def optimize_transit_network(self) -> List[Dict[str, Any]]:
        """
        Analyzes traffic congestion along each bus corridor:
        - If a zone on the bus line is BLOCKED or HEAVY CONGESTION:
          Calculates dynamic detour routing to bypass the bottleneck
        - Computes dynamic delay estimates for passengers
        """
        recommendations = []
        for line_id, line in self.bus_lines.items():
            total_delay = 0.0
            reroute_needed = False
            bottlenecks = []

            for zid in line.route_zones:
                zone = self.graph.get_zone(zid)
                if zone:
                    if zone.is_blocked:
                        reroute_needed = True
                        bottlenecks.append(f"{zone.name} (BLOCKED)")
                        total_delay += 12.0
                    elif "HEAVY" in zone.density_label:
                        bottlenecks.append(f"{zone.name} (HEAVY)")
                        total_delay += 6.0
                    elif "MODERATE" in zone.density_label:
                        total_delay += 2.0

            line.current_delay_min = total_delay
            if reroute_needed:
                line.status = "REROUTING"
                # Compute alternative detour between first and last stop
                alt_route = self.graph.calculate_route(
                    line.route_zones[0], line.route_zones[-1], avoid_flooded=True
                )
                detour_path = alt_route.path if alt_route else line.route_zones
                rec = {
                    "line_id": line_id,
                    "action": "DYNAMIC_DETOUR",
                    "reason": f"Avoid blockage at {', '.join(bottlenecks)}",
                    "suggested_path": detour_path,
                    "delay_saved_min": max(3.0, round(total_delay * 0.6, 1))
                }
            elif total_delay > 8.0:
                line.status = "DELAYED"
                rec = {
                    "line_id": line_id,
                    "action": "HEADWAY_COMPRESSION",
                    "reason": f"Corridor congestion delay ({round(total_delay, 1)}m)",
                    "suggested_dispatch": "+1 Auxiliary Bus from Metro Depot",
                    "delay_saved_min": 4.0
                }
            else:
                line.status = "ON_TIME"
                rec = {
                    "line_id": line_id,
                    "action": "NORMAL_OPERATION",
                    "reason": "Traffic flow smooth along corridor",
                    "suggested_dispatch": "Standard 10m frequency",
                    "delay_saved_min": 0.0
                }

            recommendations.append(rec)

        return recommendations

    def get_all(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.bus_lines.values()]
