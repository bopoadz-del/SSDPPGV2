from __future__ import annotations

import math
from typing import Dict, List


def smoothing_metrics(power_series: List[float]) -> Dict[str, float]:
    if not power_series:
        return {"power_std": 0.0, "coefficient_of_variation": 0.0, "ramp_rate_p95": 0.0, "smoothing_index": 0.0}
    mean_power = sum(power_series) / len(power_series)
    variance = sum((p - mean_power) ** 2 for p in power_series) / len(power_series)
    power_std = math.sqrt(variance)
    cv = power_std / mean_power if mean_power > 0 else 0.0
    ramps = [abs(power_series[i] - power_series[i - 1]) for i in range(1, len(power_series))]
    ramps_sorted = sorted(ramps)
    idx = int(0.95 * (len(ramps_sorted) - 1)) if ramps_sorted else 0
    ramp_rate_p95 = ramps_sorted[idx] if ramps_sorted else 0.0
    smoothing_index = 1 / (1 + cv) if cv > 0 else 1.0
    return {
        "power_std": power_std,
        "coefficient_of_variation": cv,
        "ramp_rate_p95": ramp_rate_p95,
        "smoothing_index": smoothing_index,
    }


def smoothing_curve(
    modules: int,
    phase_offset_deg: float,
    inverter_rating_kw: float,
    base_power_kw: float,
    steps: int = 200,
) -> Dict[str, List[float]]:
    """Compute smoothing index vs clipping loss.

    Returns arrays for clipping_loss_pct and smoothing_index.
    """
    clipping_loss_pct = []
    smoothing_index = []
    for clip in range(0, 41, 5):
        clipped_rating = inverter_rating_kw * (1 - clip / 100)
        series = []
        for i in range(steps):
            t = i / steps * 2 * math.pi
            total = 0.0
            for m in range(modules):
                phase = math.radians(phase_offset_deg) * m
                total += base_power_kw * (0.6 + 0.4 * math.sin(t + phase))
            total = max(total, 0.0)
            total = min(total, clipped_rating)
            series.append(total)
        metrics = smoothing_metrics(series)
        smoothing_index.append(metrics["smoothing_index"])
        clipping_loss_pct.append(clip)
    return {"clipping_loss_pct": clipping_loss_pct, "smoothing_index": smoothing_index}
