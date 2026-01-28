from __future__ import annotations

from typing import Dict

from .capex import total_capex
from .finance import capital_recovery_factor


def lcoe_table(inputs: Dict[str, float]) -> Dict[str, float]:
    capex_total = total_capex(inputs)
    aep_kwh = inputs.get("aep_kwh", 0.0)
    project_life_years = int(inputs.get("project_life_years", 20))
    wacc_pct = inputs.get("wacc_pct", 8.0)
    fixed_om = inputs.get("fixed_om_usd_per_year", 0.0)
    variable_om = inputs.get("variable_om_usd_per_kwh", 0.0)
    crf = capital_recovery_factor(wacc_pct, project_life_years)
    annualized_cost = capex_total * crf + fixed_om + variable_om * aep_kwh
    lcoe = annualized_cost / aep_kwh if aep_kwh > 0 else 0.0
    return {
        "capex_total": capex_total,
        "crf": crf,
        "annualized_cost": annualized_cost,
        "aep_kwh": aep_kwh,
        "lcoe": lcoe,
    }
