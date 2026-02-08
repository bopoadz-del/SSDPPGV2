from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Dict


@dataclass
class DP3DParams:
    l1: float
    l2: float
    m_upper_arm: float
    m_middle: float
    m_lower_arm: float
    m_tip: float
    n_pendulums: int


def simulate(
    wind_speed: float,
    params: DP3DParams,
    duration_s: float = 30.0,
    dt: float = 0.03,
) -> List[Dict[str, float]]:
    """Simplified 3D deterministic DP simulation."""
    omega = 0.6 + 0.06 * wind_speed
    inertia = (
        params.m_upper_arm * params.l1**2
        + params.m_middle * params.l2**2
        + params.m_lower_arm * (params.l1 + params.l2) ** 2
        + params.m_tip * (params.l1 + params.l2) ** 2
    )
    torque_scale = 0.1 * wind_speed**2 * params.n_pendulums
    frames: List[Dict[str, float]] = []
    steps = int(duration_s / dt)
    for i in range(steps):
        t = i * dt
        phi = 0.35 * math.sin(omega * t)
        psi = 0.25 * math.sin(omega * t + math.pi / 4)
        angular_velocity = omega * (0.35 * math.cos(omega * t) + 0.25 * math.cos(omega * t + math.pi / 4))
        torque = torque_scale * (math.sin(phi) + math.sin(psi)) * 0.5
        mech_power = abs(torque * angular_velocity) * inertia * 0.0012
        power_kw = mech_power / 1000.0
        frames.append({"t": t, "phi": phi, "psi": psi, "power_kw": power_kw})
    return frames
