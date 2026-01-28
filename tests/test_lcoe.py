from mssdppg.economics.lcoe import lcoe_table


def test_lcoe_decreases_with_higher_aep():
    base_inputs = {
        "mechanical_usd": 1000000,
        "electrical_usd": 0.0,
        "civil_bos_usd": 0.0,
        "soft_costs_usd": 0.0,
        "contingency_pct": 0.0,
        "project_life_years": 20,
        "wacc_pct": 8.0,
        "fixed_om_usd_per_year": 0.0,
        "variable_om_usd_per_kwh": 0.0,
    }
    low = lcoe_table({**base_inputs, "aep_kwh": 100000})
    high = lcoe_table({**base_inputs, "aep_kwh": 200000})
    assert low["lcoe"] > 0
    assert high["lcoe"] > 0
    assert high["lcoe"] < low["lcoe"]
