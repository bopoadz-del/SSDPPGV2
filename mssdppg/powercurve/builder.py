from __future__ import annotations

from typing import Dict, List, Tuple

from mssdppg.physics.dp2d import DPParams
from mssdppg.physics.energy import sim_power_mean


def aero_power_kw(
    wind_speed: float,
    swept_area_m2: float,
    cp: float,
    rho: float,
    drivetrain_eff: float,
    availability: float,
    inverter_rating_kw: float,
) -> float:
    power_w = 0.5 * rho * swept_area_m2 * cp * wind_speed**3
    power_w *= drivetrain_eff * availability
    power_kw = power_w / 1000.0
    return min(power_kw, inverter_rating_kw)


def sim_power_kw(
    wind_speed: float,
    params: DPParams,
    inverter_rating_kw: float,
) -> float:
    power_kw = sim_power_mean(wind_speed, params)
    return min(power_kw, inverter_rating_kw)


def build_power_curve(
    speeds: List[float],
    mode: str,
    aero_params: Dict[str, float],
    sim_params: DPParams,
) -> List[Tuple[float, float]]:
    curve = []
    for speed in speeds:
        if mode == "sim":
            power_kw = sim_power_kw(speed, sim_params, aero_params["inverter_rating_kw"])
        else:
            power_kw = aero_power_kw(speed, **aero_params)
        curve.append((speed, power_kw))
    return curve
