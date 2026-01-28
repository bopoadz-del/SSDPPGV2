from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Scenario:
    name: str
    swept_area_m2: float
    cp: float
    rho: float
    drivetrain_eff: float
    availability: float
    inverter_rating_kw: float
    weibull_k: float
    weibull_c: float
    l1: float
    l2: float
    m_upper_arm: float
    m_middle: float
    m_lower_arm: float
    m_tip: float
    n_pendulums: int


DEFAULT_SCENARIO = Scenario(
    name="Baseline",
    swept_area_m2=120.0,
    cp=0.42,
    rho=1.225,
    drivetrain_eff=0.9,
    availability=0.95,
    inverter_rating_kw=250.0,
    weibull_k=2.0,
    weibull_c=7.5,
    l1=5.0,
    l2=3.0,
    m_upper_arm=120.0,
    m_middle=90.0,
    m_lower_arm=60.0,
    m_tip=40.0,
    n_pendulums=4,
)


INVESTOR_SCENARIOS = [
    {
        "name": "Conservative",
        "capex_multiplier": 1.1,
        "land_cost_usd_per_m2": 12.0,
        "wacc_pct": 9.0,
    },
    {
        "name": "Base",
        "capex_multiplier": 1.0,
        "land_cost_usd_per_m2": 8.0,
        "wacc_pct": 8.0,
    },
    {
        "name": "Aggressive",
        "capex_multiplier": 0.9,
        "land_cost_usd_per_m2": 6.0,
        "wacc_pct": 7.0,
    },
]


def scenario_dict() -> dict:
    return asdict(DEFAULT_SCENARIO)
