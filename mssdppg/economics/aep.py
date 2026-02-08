from __future__ import annotations

from typing import List, Tuple


HOURS_PER_YEAR = 8760


def aep_from_bins(curve: List[Tuple[float, float]], bins: List[Tuple[float, float]]) -> float:
    power_lookup = {speed: power for speed, power in curve}
    energy_kwh = 0.0
    for speed, prob in bins:
        power_kw = power_lookup.get(speed, 0.0)
        energy_kwh += power_kw * prob * HOURS_PER_YEAR
    return energy_kwh
