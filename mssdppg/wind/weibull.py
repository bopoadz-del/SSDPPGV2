from __future__ import annotations

import math
from typing import List, Tuple


def pdf(v: float, k: float, c: float) -> float:
    if v < 0:
        return 0.0
    return (k / c) * (v / c) ** (k - 1) * math.exp(-((v / c) ** k))


def cdf(v: float, k: float, c: float) -> float:
    if v < 0:
        return 0.0
    return 1 - math.exp(-((v / c) ** k))


def bin_probabilities(
    speeds: List[float],
    k: float,
    c: float,
    bin_width: float = 1.0,
) -> List[Tuple[float, float]]:
    """Return list of (speed_midpoint, probability) for bins."""
    bins = []
    for v in speeds:
        v0 = max(v - bin_width / 2, 0)
        v1 = v + bin_width / 2
        prob = cdf(v1, k, c) - cdf(v0, k, c)
        bins.append((v, prob))
    total = sum(p for _, p in bins) or 1.0
    return [(v, p / total) for v, p in bins]
