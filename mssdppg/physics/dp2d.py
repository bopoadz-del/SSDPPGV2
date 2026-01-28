from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Dict


@dataclass
class DPParams:
    l1: float
    l2: float
    m_upper_arm: float
    m_middle: float
    m_lower_arm: float
    m_tip: float
    n_pendulums: int


def simulate(
    wind_speed: float,
    params: DPParams,
    duration_s: float = 30.0,
    dt: float = 0.03,
) -> List[Dict[str, float]]:
    """Deterministic 2D double-pendulum-inspired simulation.

    Returns list of frames with time and power_kw.
    """
    omega1 = 0.5 + 0.05 * wind_speed
    omega2 = 0.8 + 0.07 * wind_speed
    inertia = (
        params.m_upper_arm * params.l1**2
        + params.m_middle * params.l2**2
        + params.m_lower_arm * (params.l1 + params.l2) ** 2
        + params.m_tip * (params.l1 + params.l2) ** 2
    )
    torque_scale = 0.08 * wind_speed**2 * params.n_pendulums
    frames: List[Dict[str, float]] = []
    steps = int(duration_s / dt)
    for i in range(steps):
        t = i * dt
        theta1 = 0.4 * math.sin(omega1 * t)
        theta2 = 0.3 * math.sin(omega2 * t + math.pi / 6)
        angular_velocity = omega1 * 0.4 * math.cos(omega1 * t) + omega2 * 0.3 * math.cos(
            omega2 * t + math.pi / 6
        )
        torque = torque_scale * (math.sin(theta1) + 0.5 * math.sin(theta2))
        mech_power = abs(torque * angular_velocity) * inertia * 0.001
        power_kw = mech_power / 1000.0
        frames.append({"t": t, "theta1": theta1, "theta2": theta2, "power_kw": power_kw})
    return frames
