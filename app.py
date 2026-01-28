from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from mssdppg.scenarios import DEFAULT_SCENARIO, INVESTOR_SCENARIOS, scenario_dict
from mssdppg.physics.dp2d import DPParams, simulate as simulate_2d
from mssdppg.physics.energy import downsample_frames
from mssdppg.wind.weibull import bin_probabilities
from mssdppg.wind.histogram import parse_histogram
from mssdppg.wind.presets import WEIBULL_PRESETS
from mssdppg.powercurve.builder import build_power_curve
from mssdppg.powercurve.cache import make_key, read_cache, write_cache
from mssdppg.economics.aep import aep_from_bins
from mssdppg.economics.capex import total_capex
from mssdppg.economics.lcoe import lcoe_table
from mssdppg.economics.land import land_metrics
from mssdppg.economics.smoothing import smoothing_curve


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/scenarios")
    def scenarios() -> Any:
        return jsonify(
            {
                "default": scenario_dict(),
                "weibull_presets": WEIBULL_PRESETS,
                "investor_scenarios": INVESTOR_SCENARIOS,
            }
        )

    @app.post("/api/power_curve")
    def power_curve() -> Any:
        payload = request.get_json(force=True)
        speeds = payload.get("speeds") or list(range(0, 26))
        mode = payload.get("mode", "aero")
        aero_params = payload.get("aero_params", {})
        sim_inputs = payload.get("sim_params", {})
        cache_payload = {
            "speeds": speeds,
            "mode": mode,
            "aero_params": aero_params,
            "sim_params": sim_inputs,
        }
        key = make_key(cache_payload)
        cached = read_cache(key)
        if cached:
            return jsonify({"cache_key": key, "curve": cached["curve"], "cached": True})
        sim_params = DPParams(
            l1=sim_inputs.get("l1", DEFAULT_SCENARIO.l1),
            l2=sim_inputs.get("l2", DEFAULT_SCENARIO.l2),
            m_upper_arm=sim_inputs.get("m_upper_arm", DEFAULT_SCENARIO.m_upper_arm),
            m_middle=sim_inputs.get("m_middle", DEFAULT_SCENARIO.m_middle),
            m_lower_arm=sim_inputs.get("m_lower_arm", DEFAULT_SCENARIO.m_lower_arm),
            m_tip=sim_inputs.get("m_tip", DEFAULT_SCENARIO.m_tip),
            n_pendulums=int(sim_inputs.get("n_pendulums", DEFAULT_SCENARIO.n_pendulums)),
        )
        curve = build_power_curve(speeds, mode, aero_params, sim_params)
        write_cache(key, {"curve": curve})
        return jsonify({"cache_key": key, "curve": curve, "cached": False})

    @app.post("/api/aep")
    def aep() -> Any:
        payload = request.get_json(force=True)
        curve = payload.get("curve", [])
        wind_mode = payload.get("wind_mode", "weibull")
        if wind_mode == "histogram":
            histogram = payload.get("histogram", [])
            bins = [(float(v), float(p)) for v, p in histogram]
        else:
            k = float(payload.get("weibull_k", DEFAULT_SCENARIO.weibull_k))
            c = float(payload.get("weibull_c", DEFAULT_SCENARIO.weibull_c))
            speeds = [v for v, _ in curve]
            bins = bin_probabilities(speeds, k, c)
        aep_kwh = aep_from_bins(curve, bins)
        return jsonify({"aep_kwh": aep_kwh, "bins": bins})

    @app.post("/api/lcoe")
    def lcoe() -> Any:
        payload = request.get_json(force=True)
        base_table = lcoe_table(payload)
        investor_rows = []
        for scenario in INVESTOR_SCENARIOS:
            inputs = dict(payload)
            inputs["wacc_pct"] = scenario["wacc_pct"]
            inputs["mechanical_usd"] = inputs.get("mechanical_usd", 0.0) * scenario["capex_multiplier"]
            inputs["electrical_usd"] = inputs.get("electrical_usd", 0.0) * scenario["capex_multiplier"]
            inputs["civil_bos_usd"] = inputs.get("civil_bos_usd", 0.0) * scenario["capex_multiplier"]
            inputs["soft_costs_usd"] = inputs.get("soft_costs_usd", 0.0) * scenario["capex_multiplier"]
            land_cost = scenario["land_cost_usd_per_m2"] * payload.get("land_area_m2", 0.0)
            inputs["mechanical_usd"] += land_cost
            row = lcoe_table(inputs)
            row["name"] = scenario["name"]
            investor_rows.append(row)
        return jsonify({"lcoe": base_table, "investor_scenarios": investor_rows})

    @app.post("/api/site_rollup")
    def site_rollup() -> Any:
        payload = request.get_json(force=True)
        modules_count = int(payload.get("modules_count", 1))
        module_aep_kwh = payload.get("module_aep_kwh", 0.0)
        module_avg_kw = module_aep_kwh / 8760 if module_aep_kwh else 0.0
        land = land_metrics(payload)
        capex_inputs = payload.get("capex_inputs", {})
        capex_total = total_capex(capex_inputs) * modules_count + land["land_cost_total"]
        total_aep_kwh = module_aep_kwh * modules_count
        total_avg_kw = module_avg_kw * modules_count
        lcoe_inputs = {
            "mechanical_usd": capex_total,
            "electrical_usd": 0.0,
            "civil_bos_usd": 0.0,
            "soft_costs_usd": 0.0,
            "contingency_pct": 0.0,
            "aep_kwh": total_aep_kwh,
            "project_life_years": payload.get("project_life_years", 20),
            "wacc_pct": payload.get("wacc_pct", 8.0),
            "fixed_om_usd_per_year": payload.get("fixed_om_usd_per_year", 0.0),
            "variable_om_usd_per_kwh": payload.get("variable_om_usd_per_kwh", 0.0),
        }
        lcoe_output = lcoe_table(lcoe_inputs)
        return jsonify(
            {
                "total_capex_usd": capex_total,
                "total_aep_kwh": total_aep_kwh,
                "total_avg_kw": total_avg_kw,
                "land_area_m2": land["land_area_m2"],
                "land_area_ha": land["land_area_ha"],
                "lcoe": lcoe_output,
            }
        )

    @app.post("/api/simulate")
    def simulate() -> Any:
        payload = request.get_json(force=True)
        wind_speed = float(payload.get("wind_speed", 8.0))
        params = DPParams(
            l1=payload.get("l1", DEFAULT_SCENARIO.l1),
            l2=payload.get("l2", DEFAULT_SCENARIO.l2),
            m_upper_arm=payload.get("m_upper_arm", DEFAULT_SCENARIO.m_upper_arm),
            m_middle=payload.get("m_middle", DEFAULT_SCENARIO.m_middle),
            m_lower_arm=payload.get("m_lower_arm", DEFAULT_SCENARIO.m_lower_arm),
            m_tip=payload.get("m_tip", DEFAULT_SCENARIO.m_tip),
            n_pendulums=int(payload.get("n_pendulums", DEFAULT_SCENARIO.n_pendulums)),
        )
        frames = simulate_2d(wind_speed, params, duration_s=payload.get("duration_s", 20.0))
        frames = downsample_frames(frames, step=payload.get("downsample", 5))
        return jsonify({"frames": frames})

    @app.post("/api/smoothing")
    def smoothing() -> Any:
        payload = request.get_json(force=True)
        curve = smoothing_curve(
            modules=int(payload.get("number_of_modules", 4)),
            phase_offset_deg=float(payload.get("phase_offset_deg", 30.0)),
            inverter_rating_kw=float(payload.get("inverter_rating_kw", 250.0)),
            base_power_kw=float(payload.get("base_power_kw", 120.0)),
        )
        return jsonify(curve)

    @app.post("/api/histogram_parse")
    def histogram_parse() -> Any:
        payload = request.get_json(force=True)
        parsed = parse_histogram(payload.get("csv_text", ""))
        return jsonify({"histogram": parsed})

    return app


def cli() -> None:
    parser = argparse.ArgumentParser(description="MSSDPPG Live Simulator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scenarios")
    lcoe_cmd = sub.add_parser("lcoe")
    lcoe_cmd.add_argument("--aep-kwh", type=float, default=1500000)
    lcoe_cmd.add_argument("--capex", type=float, default=1200000)
    args = parser.parse_args()
    if args.command == "scenarios":
        print(json.dumps(scenario_dict(), indent=2))
        return
    if args.command == "lcoe":
        table = lcoe_table(
            {
                "mechanical_usd": args.capex,
                "electrical_usd": 0.0,
                "civil_bos_usd": 0.0,
                "soft_costs_usd": 0.0,
                "contingency_pct": 0.0,
                "aep_kwh": args.aep_kwh,
                "project_life_years": 20,
                "wacc_pct": 8.0,
                "fixed_om_usd_per_year": 0.0,
                "variable_om_usd_per_kwh": 0.0,
            }
        )
        print(json.dumps(table, indent=2))


app = create_app()


if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] in {"scenarios", "lcoe"}:
        cli()
    else:
        port = int(os.environ.get("PORT", "5000"))
        app.run(host="0.0.0.0", port=port, debug=True)
