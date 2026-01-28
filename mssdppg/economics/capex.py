from __future__ import annotations

from typing import Dict


def total_capex(inputs: Dict[str, float]) -> float:
    base = (
        inputs.get("mechanical_usd", 0.0)
        + inputs.get("electrical_usd", 0.0)
        + inputs.get("civil_bos_usd", 0.0)
        + inputs.get("soft_costs_usd", 0.0)
    )
    contingency_pct = inputs.get("contingency_pct", 0.0) / 100.0
    return base * (1 + contingency_pct)
