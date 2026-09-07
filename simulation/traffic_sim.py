"""
CityFlow Urban Simulation Engine.
Simulates autonomous traffic fluctuation, vehicle arrivals, rush-hour waves,
signal cycling, waste accumulation, and rainfall events for full smart-city testing.
"""
import random
import time
import threading
from typing import Dict, Any, Callable, Optional
from city_model import CityRoadGraph


class SmartCitySimulationEngine:
    """Orchestrates periodic simulated updates across all city systems."""

    def __init__(
        self,
        city_graph: CityRoadGraph,
        broadcast_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.graph = city_graph
        self.broadcast_callback = broadcast_callback
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

        # Rush hour state multiplier (1.0 = normal, 2.2 = peak rush)
        self.traffic_intensity = 1.0
        self.sim_tick = 0

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.is_running = False

    def _run_loop(self) -> None:
        while self.is_running:
            try:
                self.sim_tick += 1
                self._update_traffic_fluctuations()
                time.sleep(2.0)
            except Exception as e:
                time.sleep(2.0)

    def _update_traffic_fluctuations(self) -> None:
        """Simulates realistic traffic flow fluctuations across zones."""
        for zid, zone in self.graph.zones.items():
            if zone.is_blocked:
                continue

            # Random small flow variations based on zone type
            base = 15
            if zone.type == "intersection":
                base = 28
            elif zone.type == "commercial":
                base = 32
            elif zone.type == "expressway":
                base = 45
            elif zone.type == "residential":
                base = 8

            target_count = int(base * self.traffic_intensity + random.randint(-4, 5))
            target_count = max(0, target_count)

            # Smoothly drift count toward target
            zone.vehicle_count = int((zone.vehicle_count * 0.7) + (target_count * 0.3))

            # Update density label
            if zone.vehicle_count > 35:
                zone.density_label = "HEAVY CONGESTION"
                zone.density_color = (255, 75, 43)  # Red
            elif zone.vehicle_count > 18:
                zone.density_label = "MODERATE TRAFFIC"
                zone.density_color = (246, 211, 101)  # Yellow/Gold
            else:
                zone.density_label = "SMOOTH (LOW)"
                zone.density_color = (56, 239, 125)  # Green

            zone.update_telemetry(
                vehicle_count=zone.vehicle_count,
                density_label=zone.density_label,
                density_color=zone.density_color,
                inbound=random.randint(1, 6),
                outbound=random.randint(1, 6)
            )
