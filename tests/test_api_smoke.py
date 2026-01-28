from app import create_app


def test_api_scenarios_and_lcoe():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "default" in data
    lcoe_payload = {
        "mechanical_usd": 1000000,
        "electrical_usd": 0.0,
        "civil_bos_usd": 0.0,
        "soft_costs_usd": 0.0,
        "contingency_pct": 0.0,
        "aep_kwh": 150000,
        "project_life_years": 20,
        "wacc_pct": 8.0,
        "fixed_om_usd_per_year": 0.0,
        "variable_om_usd_per_kwh": 0.0,
    }
    resp = client.post("/api/lcoe", json=lcoe_payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "lcoe" in data
