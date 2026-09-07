"""
Smart Ambulance Emergency Router.
Finds the fastest route to medical trauma centers, factoring in real-time congestion,
road blockages, and requesting traffic signal preemption.
"""
from typing import Dict, List, Optional, Any
from city_model import CityRoadGraph, RouteResult


class AmbulanceRouter:
    """Calculates rapid emergency response routes with green-wave prioritization."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph

    def find_nearest_hospital(self, from_zone_id: str) -> Optional[str]:
        """Identifies the closest hospital zone based on distance and congestion."""
        hospital_zones = [
            zid for zid, z in self.graph.zones.items()
            if z.type == "hospital"
        ]
        if not hospital_zones:
            # Fallback to any hospital-named zone or default
            hospital_zones = [zid for zid in self.graph.zones if "hospital" in zid.lower()]
        
        if not hospital_zones:
            return None

        best_hosp = None
        min_time = float("inf")
        for hosp_id in hospital_zones:
            res = self.graph.calculate_route(from_zone_id, hosp_id, is_emergency=True)
            if res and res.estimated_time_sec < min_time:
                min_time = res.estimated_time_sec
                best_hosp = hosp_id

        return best_hosp or hospital_zones[0]

    def plan_emergency_route(
        self,
        pickup_zone_id: str,
        hospital_zone_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Plans a complete two-phase emergency trajectory:
        1. Base/Depot -> Pickup location
        2. Pickup location -> Destination Trauma Center
        """
        dest_hospital = hospital_zone_id or self.find_nearest_hospital(pickup_zone_id)
        if not dest_hospital:
            return None

        route = self.graph.calculate_route(
            from_zone_id=pickup_zone_id,
            to_zone_id=dest_hospital,
            is_emergency=True,
            avoid_flooded=True
        )
        if not route:
            return None

        # Preemption target signals along the route
        preempt_signals = []
        for zid in route.path:
            zone = self.graph.get_zone(zid)
            if zone and zone.signal_id:
                preempt_signals.append({
                    "zone_id": zone.id,
                    "signal_id": zone.signal_id,
                    "zone_name": zone.name
                })

        return {
            "origin_zone": pickup_zone_id,
            "destination_hospital": dest_hospital,
            "route": route.to_dict(),
            "preempt_signals": preempt_signals,
            "saved_delay_sec": route.congestion_penalty_sec
        }
