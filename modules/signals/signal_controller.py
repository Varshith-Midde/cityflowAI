"""
Adaptive Traffic Signal Controller and AI Demand Optimizer.
Dynamically balances intersection green times based on real-time vehicle density,
minimizing congestion, queue lengths, and vehicle idle emissions.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Any
from city_model import CityRoadGraph, TrafficZone


@dataclass
class IntersectionSignal:
    id: str
    zone_id: str
    name: str
    current_phase: str = "GREEN"  # GREEN, YELLOW, RED
    time_remaining_sec: int = 30
    min_green_sec: int = 15
    max_green_sec: int = 75
    yellow_sec: int = 4
    all_red_sec: int = 2
    is_preempted: bool = False
    cycle_count: int = 0
    total_vehicles_served: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "zone_id": self.zone_id,
            "name": self.name,
            "phase": self.current_phase,
            "time_remaining_sec": self.time_remaining_sec,
            "is_preempted": self.is_preempted,
            "cycle_count": self.cycle_count,
            "total_vehicles_served": self.total_vehicles_served
        }


class TrafficSignalNetwork:
    """Manages all smart traffic signals across the city graph."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.signals: Dict[str, IntersectionSignal] = {}
        self._init_signals()

    def _init_signals(self) -> None:
        """Finds all zones with signal_id and instantiates controllers."""
        for zid, zone in self.graph.zones.items():
            if zone.signal_id:
                sig = IntersectionSignal(
                    id=zone.signal_id,
                    zone_id=zid,
                    name=f"{zone.name} Signal",
                    current_phase="GREEN" if len(self.signals) % 2 == 0 else "RED",
                    time_remaining_sec=30
                )
                self.signals[sig.id] = sig
                zone.signal_state = sig.current_phase

    def optimize_signals(self) -> List[Dict[str, Any]]:
        """
        AI Optimization algorithm:
        Inspects live vehicle queue at each intersection and adapts green phase timing:
        - If density is HEAVY (>20 vehicles): Allocates +15s to +25s green time
        - If density is MODERATE (5-20 vehicles): Keeps standard 30s
        - If density is LOW/SMOOTH (<5 vehicles): Truncates green time to allow cross-traffic
        """
        optimizations = []
        for sig_id, sig in self.signals.items():
            zone = self.graph.get_zone(sig.zone_id)
            if not zone or sig.is_preempted:
                continue

            v_count = zone.vehicle_count
            old_timer = sig.time_remaining_sec

            # Calculate optimal target green time
            if v_count > 25 or "HEAVY" in zone.density_label:
                target_green = min(sig.max_green_sec, 30 + int(v_count * 1.2))
                reason = "Extended Green for Congestion Clearing"
            elif v_count < 4 and "SMOOTH" in zone.density_label:
                target_green = sig.min_green_sec
                reason = "Early Truncation for Low Flow Demand"
            else:
                target_green = 35
                reason = "Balanced Flow Timing"

            if sig.current_phase == "GREEN":
                sig.time_remaining_sec = max(5, target_green)
                zone.signal_timer_sec = sig.time_remaining_sec

            optimizations.append({
                "signal_id": sig_id,
                "zone_id": zone.id,
                "zone_name": zone.name,
                "vehicle_count": v_count,
                "adjusted_green_sec": target_green,
                "reason": reason
            })

        return optimizations

    def step(self) -> None:
        """Countdown timer update called every second."""
        for sig_id, sig in self.signals.items():
            sig.time_remaining_sec -= 1
            zone = self.graph.get_zone(sig.zone_id)

            if sig.time_remaining_sec <= 0:
                # Phase Transition
                if sig.current_phase == "GREEN":
                    sig.current_phase = "YELLOW"
                    sig.time_remaining_sec = sig.yellow_sec
                elif sig.current_phase == "YELLOW":
                    sig.current_phase = "RED"
                    sig.time_remaining_sec = 25  # Red duration
                    sig.cycle_count += 1
                elif sig.current_phase == "RED":
                    sig.current_phase = "GREEN"
                    # Dynamically set green time based on current traffic
                    q_len = zone.vehicle_count if zone else 10
                    sig.time_remaining_sec = min(sig.max_green_sec, max(sig.min_green_sec, int(q_len * 1.5) + 15))

            if zone:
                zone.signal_state = sig.current_phase
                zone.signal_timer_sec = sig.time_remaining_sec

    def set_preemption(self, signal_id: str, state: str = "GREEN", duration_sec: int = 45) -> bool:
        """Manual or emergency green-wave preemption override."""
        sig = self.signals.get(signal_id)
        if not sig:
            return False
        sig.current_phase = state
        sig.time_remaining_sec = duration_sec
        sig.is_preempted = True
        zone = self.graph.get_zone(sig.zone_id)
        if zone:
            zone.signal_state = state
            zone.signal_timer_sec = duration_sec
        return True

    def get_all(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.signals.values()]
