from __future__ import annotations

from typing import Dict


def land_metrics(inputs: Dict[str, float]) -> Dict[str, float]:
    modules_count = inputs.get("modules_count", 1)
    module_spacing_m2 = inputs.get("module_spacing_m2")
    density_modules_per_ha = inputs.get("density_modules_per_ha")
    if module_spacing_m2 is None and density_modules_per_ha:
        module_spacing_m2 = 10000 / density_modules_per_ha
    if module_spacing_m2 is None:
        module_spacing_m2 = 200.0
    land_area_m2 = modules_count * module_spacing_m2
    land_area_ha = land_area_m2 / 10000
    land_cost_usd_per_m2 = inputs.get("land_cost_usd_per_m2", 0.0)
    land_cost_total = land_area_m2 * land_cost_usd_per_m2
    return {
        "land_area_m2": land_area_m2,
        "land_area_ha": land_area_ha,
        "land_cost_total": land_cost_total,
    }
