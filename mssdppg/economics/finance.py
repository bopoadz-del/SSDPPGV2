from __future__ import annotations


def capital_recovery_factor(wacc_pct: float, years: int) -> float:
    rate = wacc_pct / 100.0
    if rate == 0:
        return 1 / max(years, 1)
    return (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)
