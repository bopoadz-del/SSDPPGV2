from __future__ import annotations

from typing import List, Dict

from .dp2d import DPParams, simulate as simulate_2d
from .dp3d import DP3DParams, simulate as simulate_3d


def sim_power_mean(
    wind_speed: float,
    params: DPParams,
    duration_s: float = 40.0,
    dt: float = 0.03,
    steady_state_s: float = 12.0,
) -> float:
    frames = simulate_2d(wind_speed, params, duration_s=duration_s, dt=dt)
    return _mean_power(frames, steady_state_s)


def sim_power_mean_3d(
    wind_speed: float,
    params: DP3DParams,
    duration_s: float = 40.0,
    dt: float = 0.03,
    steady_state_s: float = 12.0,
) -> float:
    frames = simulate_3d(wind_speed, params, duration_s=duration_s, dt=dt)
    return _mean_power(frames, steady_state_s)


def downsample_frames(frames: List[Dict[str, float]], step: int = 5) -> List[Dict[str, float]]:
    return frames[::step]


def _mean_power(frames: List[Dict[str, float]], steady_state_s: float) -> float:
    steady = [f["power_kw"] for f in frames if f["t"] >= steady_state_s]
    if not steady:
        return 0.0
    return sum(steady) / len(steady)
