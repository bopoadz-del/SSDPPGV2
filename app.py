"""
MSSDPPG Web UI with Real-time 3D Visualization
Flask backend for simulation and data streaming
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import os
import json
import threading
from datetime import datetime
import time
from queue import Queue, Empty
from dataclasses import replace as dc_replace

from MSSDPPG_UltraRealistic_v2 import (
    SCENARIOS, standard_wind_profile, load_wind_csv,
    Pendulum2D, Pendulum3D, run_one
)

from mssdppg.rubric_defaults import RUBRIC_CONFIGS, ALIASES, calc_arm_masses
from mssdppg.wind import speed_bins, weibull_bin_probs, parse_histogram_csv, histogram_to_bins, WEIBULL_PRESETS
from mssdppg.economics import (
    CapexBreakdown, FinanceParams, lcoe_usd_per_kwh, aep_from_power_curve, apply_losses,
    site_rollup, smoothing_metrics, apply_phase_offsets, even_phase_offsets
)
from mssdppg.power_curve import build_power_curve

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'mssdppg-secret-key'

# Simulation data queue for polling-based updates
# NOTE: we drain this queue on each /api/sim-status call (no duplicates, bounded payload)
sim_queue: Queue = Queue()
sim_result: dict = {}

# Global simulation state
sim_state = {
    'running': False,
    'progress': 0,
    'current_frame': 0,
    'total_frames': 0,
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scenarios')
def get_scenarios():
    """Get available scenarios (physics + rubric defaults)."""
    out = {}
    for name, scenario in SCENARIOS.items():
        # Hide deprecated aliases in UI list; still accessible by API if requested.
        is_alias = name in ALIASES
        rub_key = name if name in RUBRIC_CONFIGS else ALIASES.get(name)
        rub = RUBRIC_CONFIGS.get(rub_key) if rub_key else None

        out[name] = {
            'display_name': scenario.name,
            'L1': scenario.L1,
            'L2': scenario.L2,
            'vane_w': scenario.vane_w,
            'vane_h': scenario.vane_h,
            'm_upper_arm': scenario.m_upper_arm,
            'm_middle': scenario.m_middle,
            'm_lower_arm': scenario.m_lower_arm,
            'm_tip': scenario.m_tip,
            'n_pendulums': scenario.n_pendulums,
            'max_angle_deg': float(np.rad2deg(scenario.max_angle_rad)),
            'bearing_mu': scenario.bearing_mu,
            'drag_cd': scenario.drag_cd,
            'color': scenario.color,
            'deprecated_alias': bool(is_alias),
            # Rubric layer (if available)
            'rubric': (None if rub is None else {
                'key': rub.key,
                'rated_wind_mps': rub.rated_wind_mps,
                'expected_power_kw': rub.expected_power_kw,
                'cost_usd': rub.cost_usd,
                'footprint_m2': rub.footprint_m2,
                'shafts': rub.shafts,
                'notes': rub.notes,
                'mass_rule': {
                    'middle_fixed_kg': rub.m_middle,
                    'tip_fixed_kg': rub.m_tip,
                    'arm_mass_scaling': '(L/2)^2',
                }
            }),
        }
    return jsonify(out)


@app.route('/api/config')
def get_config():
    """Get simulation configuration options"""
    return jsonify({
        'durations': ['6h', '12h'],
        'modes': ['2d', '3d', 'dual'],
        'controls': ['lock', 'pushpull', 'none'],
        'assist': ['on', 'off'],
        'scenarios': list(SCENARIOS.keys())
    })

@app.route('/api/simulate', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    global sim_result
    if sim_state['running']:
        return jsonify({'error': 'Simulation already running'}), 400

    params = request.json or {}

    # Clear any prior frames/results
    sim_result = {}
    try:
        while True:
            sim_queue.get_nowait()
    except Empty:
        pass

    sim_state['running'] = True
    sim_state['progress'] = 0
    sim_state['current_frame'] = 0
    sim_state['total_frames'] = 0

    # Run simulation in background thread
    thread = threading.Thread(
        target=run_simulation_background,
        args=(params,)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/sim-status')
def sim_status():
    """Get current simulation status and data"""
    # Drain a bounded number of frames to avoid duplicates and huge payloads.
    frames = []
    max_frames = 250
    for _ in range(max_frames):
        try:
            frames.append(sim_queue.get_nowait())
        except Empty:
            break

    return jsonify({
        'running': sim_state['running'],
        'progress': sim_state['progress'],
        'frame_count': sim_state['current_frame'],
        'total_frames': sim_state.get('total_frames', 0),
        'frames': frames
    })

@app.route('/api/sim-result')
def sim_result_endpoint():
    """Get simulation results"""
    return jsonify(sim_result)

def run_simulation_background(params):
    """Run simulation and stream data via queue"""
    global sim_result
    try:
        scenario_key = params.get('scenario', '4x40ft')
        duration_h = int(params.get('duration', '6').replace('h', ''))
        sim_mode = params.get('mode', '2d')
        control = params.get('control', 'lock')
        assist = params.get('assist', 'on') == 'on'

        duration_s = duration_h * 3600

        # Load wind profile
        wind_constant = params.get('wind_constant', None)
        wind_file = params.get('windfile', '')
        if wind_constant not in (None, '', 0):
            try:
                v0 = float(wind_constant)
            except Exception:
                v0 = None
            if v0 is not None:
                t_wind = np.array([0.0, duration_s], dtype=float)
                v_wind = np.array([v0, v0], dtype=float)
            else:
                t_wind, v_wind = standard_wind_profile(duration_s)
        elif wind_file and os.path.exists(wind_file):
            t_wind, v_wind = load_wind_csv(wind_file)
        else:
            t_wind, v_wind = standard_wind_profile(duration_s)

        S = SCENARIOS[scenario_key]

        # Optional parameter overrides from UI (with rubric mass scaling)
        overrides = params.get('overrides') or {}
        if isinstance(overrides, dict) and overrides:
            S = _apply_overrides(S, overrides)

        # Run simulation with data streaming
        sim_result = run_simulation_with_streaming(
            sim_mode, S, duration_s, control, assist,
            dict(offset=0.12, k_phi=10.0, c_phi=0.6, Km_phi=3.0),
            t_wind, v_wind
        )

    except Exception as e:
        sim_result = {'error': str(e)}
    finally:
        sim_state['running'] = False

def run_simulation_with_streaming(sim_mode, S, duration_s, control_mode, assist,
                                  spatial_params, t_wind, v_wind, pend_2d=None, pend_3d=None):
    """Run simulation and stream frame data.

    This uses the same physics (Pendulum2D/Pendulum3D) as the offline script, but
    advances time in accelerated fixed steps and only emits frames at 1 Hz for the UI.
    """

    results = {'summary': {}}

    # UI frame rate (simulated seconds per emitted frame)
    frame_dt = 1.0
    # Internal integration step (smaller => more stable; larger => faster)
    dt_step = 0.2

    if sim_mode == '2d':
        pend = Pendulum2D(S, control_mode=control_mode, assist=assist)
        y = np.array([0.15, 0.0, 0.0, 0.0], dtype=float)
        eom_func = pend.eom
    elif sim_mode == '3d':
        pend = Pendulum3D(S, control_mode=control_mode, assist=assist, **spatial_params)
        y = np.array([0.15, 0.0, 0.0, 0.0, pend.phi0, pend.wphi0], dtype=float)
        eom_func = pend.eom3d
    else:
        # For 'dual' mode, stream 2D (the UI visualizer is planar); summary still computed.
        pend = Pendulum2D(S, control_mode=control_mode, assist=assist)
        y = np.array([0.15, 0.0, 0.0, 0.0], dtype=float)
        eom_func = pend.eom

    sim_state['total_frames'] = int(duration_s / frame_dt) + 1

    t = 0.0
    next_emit_t = 0.0
    emitted = 0

    # For summary
    t_hist = []
    p_hist = []

    while t < duration_s:
        v_wind_curr = float(np.interp(t, t_wind, v_wind))
        pend.v_wind = v_wind_curr
        pend.dt_local = dt_step

        dydt = np.array(eom_func(t, y), dtype=float)
        y = y + dydt * dt_step

        # Power at this step (kW, system-scaled)
        if pend.P_upper_hist:
            P_total_W = (pend.P_upper_hist[-1] + pend.P_lower_hist[-1] + pend.P_shaft_hist[-1] * 0.95 * 0.88)
            power_kW = (P_total_W * S.n_pendulums) / 1000.0
        else:
            power_kW = 0.0

        t_hist.append(t)
        p_hist.append(power_kW)

        # Emit frame at 1 Hz simulated time
        if t + 1e-9 >= next_emit_t:
            frame = {
                't': float(t),
                'theta1': float(y[0]),
                'omega1': float(y[1]),
                'theta2': float(y[2]),
                'omega2': float(y[3]),
                'wind': float(v_wind_curr),
                'power_kW': float(power_kW),
                'frame': int(emitted),
            }
            if sim_mode == '3d' and len(y) >= 6:
                frame['phi'] = float(y[4])
                frame['dphi'] = float(y[5])

            # Keep queue bounded (drop oldest if UI isn't polling fast enough)
            while sim_queue.qsize() > 2000:
                try:
                    sim_queue.get_nowait()
                except Empty:
                    break
            sim_queue.put(frame)

            emitted += 1
            sim_state['current_frame'] = emitted
            sim_state['progress'] = int((t / duration_s) * 100)
            next_emit_t += frame_dt

        t = min(duration_s, t + dt_step)

    # Summary
    if len(p_hist) >= 2:
        tt = np.array(t_hist, dtype=float)
        pp = np.array(p_hist, dtype=float)
        results['summary'] = {
            'avg_kW': float(np.mean(pp)),
            'peak_kW': float(np.max(pp)),
            'energy_kWh': float(np.trapz(pp, tt) / 3600.0),
            'coil_Tmax_C': float(np.max(pend.T_coil_hist) - 273.15) if pend.T_coil_hist else 0.0,
        }

    return results


# ---------------- Wind + Economics API ----------------

@app.route('/api/weibull-presets')
def api_weibull_presets():
    return jsonify({name: {'k': p.k, 'c': p.c} for name, p in WEIBULL_PRESETS.items()})

@app.route('/api/upload-wind-hist', methods=['POST'])
def api_upload_wind_hist():
    """Upload a histogram CSV and return parsed probabilities."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    try:
        df = parse_histogram_csv(f.read())
        return jsonify({'rows': df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/wind-prob', methods=['POST'])
def api_wind_prob():
    """Return binned probabilities for Weibull or histogram."""
    payload = request.json or {}
    method = (payload.get('method') or 'weibull').lower()
    v_max = float(payload.get('v_max', 25.0))
    step = float(payload.get('step', 0.5))
    mids, edges = speed_bins(0.0, v_max, step)

    if method == 'weibull':
        k = float(payload.get('k', 2.0))
        c = float(payload.get('c', 8.0))
        prob = weibull_bin_probs(edges, k, c).tolist()
        return jsonify({'speed_mps': mids.tolist(), 'prob': prob, 'edges': edges.tolist()})

    if method == 'hist':
        rows = payload.get('rows') or []
        import pandas as pd
        df = pd.DataFrame(rows)
        if df.empty or 'speed_mps' not in df or 'prob' not in df:
            return jsonify({'error': 'Histogram rows must include speed_mps and prob'}), 400
        prob = histogram_to_bins(df, edges).tolist()
        return jsonify({'speed_mps': mids.tolist(), 'prob': prob, 'edges': edges.tolist()})

    return jsonify({'error': f'Unknown method: {method}'}), 400

def _resolve_scenario_key(key: str) -> str:
    if key in SCENARIOS:
        return key
    ali = ALIASES.get(key)
    return ali if ali in SCENARIOS else list(SCENARIOS.keys())[0]

def _apply_overrides(S, overrides: dict):
    """Apply numeric overrides; if L1/L2 change, optionally rescale arm masses."""
    if not overrides:
        return S
    def _num(key, default, cast=float):
        if key not in overrides or overrides[key] in (None, ''):
            return default
        try:
            return cast(overrides[key])
        except Exception:
            return default

    L1 = _num('L1', S.L1)
    L2 = _num('L2', S.L2)

    # If user did not explicitly override arm masses, rescale them using rubric rule.
    m_upper = S.m_upper_arm
    m_lower = S.m_lower_arm
    if ('m_upper_arm' not in overrides) or ('m_lower_arm' not in overrides):
        m_upper, m_lower = calc_arm_masses(L1, L2)

    return dc_replace(
        S,
        L1=L1,
        L2=L2,
        vane_w=_num('vane_w', S.vane_w),
        vane_h=_num('vane_h', S.vane_h),
        m_upper_arm=_num('m_upper_arm', m_upper),
        m_middle=_num('m_middle', S.m_middle),
        m_lower_arm=_num('m_lower_arm', m_lower),
        m_tip=_num('m_tip', S.m_tip),
        n_pendulums=_num('n_pendulums', S.n_pendulums, cast=int),
    )

@app.route('/api/power-curve', methods=['POST'])
def api_power_curve():
    """Compute a power curve by running short constant-wind simulations."""
    payload = request.json or {}
    scenario_key = _resolve_scenario_key(payload.get('scenario', '4x40ft_asymmetric'))
    mode = payload.get('mode', '2d')
    control = payload.get('control', 'lock')
    assist = bool(payload.get('assist', True))
    duration_s = float(payload.get('duration_s', 120.0))
    dt_step = float(payload.get('dt_step', 0.2))

    S = SCENARIOS[scenario_key]
    S = _apply_overrides(S, payload.get('overrides') or {})

    # Build speeds list
    speeds = payload.get('speeds')
    if not speeds:
        v_max = float(payload.get('v_max', 20.0))
        v_step = float(payload.get('v_step', 1.0))
        speeds = list(np.arange(0.0, v_max + 1e-9, v_step))
    speeds = [float(x) for x in speeds]

    curve = build_power_curve(S, speeds=speeds, mode=mode, control=control, assist=assist, duration_s=duration_s, dt_step=dt_step)
    curve['scenario'] = scenario_key
    curve['mode'] = mode
    curve['control'] = control
    curve['assist'] = assist
    return jsonify(curve)

@app.route('/api/lcoe', methods=['POST'])
def api_lcoe():
    """Compute AEP + LCOE from wind resource + power curve + finance inputs."""
    payload = request.json or {}
    scenario_key = _resolve_scenario_key(payload.get('scenario', '4x40ft_asymmetric'))
    mode = payload.get('mode', '2d')
    control = payload.get('control', 'lock')
    assist = bool(payload.get('assist', True))

    # power curve
    curve = payload.get('power_curve')
    if not curve:
        # compute a coarse curve quickly
        speeds = list(np.arange(0.0, 20.0 + 1e-9, 1.0))
        S0 = _apply_overrides(SCENARIOS[scenario_key], payload.get('overrides') or {})
        curve = build_power_curve(S0, speeds=speeds, mode=mode, control=control, assist=assist, duration_s=60.0, dt_step=0.25)

    speed_curve = np.array(curve['speed_mps'], dtype=float)
    power_curve = np.array(curve.get('mean_kw') or curve.get('power_kw') or curve.get('kw') or curve.get('mean_kW') or curve.get('mean'), dtype=float)

    # wind probability
    wind = payload.get('wind') or {'method': 'weibull', 'k': 2.0, 'c': 8.0}
    v_max = float(wind.get('v_max', 25.0))
    step = float(wind.get('step', 0.5))
    mids, edges = speed_bins(0.0, v_max, step)
    method = (wind.get('method') or 'weibull').lower()

    if method == 'weibull':
        prob = weibull_bin_probs(edges, float(wind.get('k', 2.0)), float(wind.get('c', 8.0)))
    elif method == 'hist':
        import pandas as pd
        df = pd.DataFrame(wind.get('rows') or [])
        if df.empty:
            return jsonify({'error': 'wind.rows required for hist method'}), 400
        prob = histogram_to_bins(df, edges)
    else:
        return jsonify({'error': f'Unknown wind.method {method}'}), 400

    # interpolate power curve onto bin midpoints
    p_on_bins = np.interp(mids, speed_curve, power_curve, left=0.0, right=float(power_curve[-1] if power_curve.size else 0.0))

    physics = payload.get('physics') or {}
    drivetrain_eff = float(physics.get('drivetrain_eff', 0.88))
    availability = float(physics.get('availability', 0.95))
    inverter_kw = physics.get('inverter_kw')
    inverter_kw = float(inverter_kw) if inverter_kw not in (None, '', 0) else None

    # Optional aero curve (Betz-like): P(v)=0.5*rho*A*Cp*v^3
    curve_source = (physics.get('curve_source') or 'simulation').lower()
    swept_area_m2 = float(physics.get('swept_area_m2', 0.0) or 0.0)
    cp = float(physics.get('cp', 0.35) or 0.35)
    rho = float(physics.get('rho', 1.225) or 1.225)

    aero_kw = (0.5 * rho * max(0.0, swept_area_m2) * max(0.0, cp) * (mids ** 3)) / 1000.0
    base_kw = aero_kw if (curve_source == 'aero' and swept_area_m2 > 0) else p_on_bins

    delivered_kw, clip_loss_kw = apply_losses(p_on_bins, drivetrain_eff=drivetrain_eff, availability=availability, inverter_kw=inverter_kw)

    annual_kwh = aep_from_power_curve(delivered_kw, prob)

    capex_p = payload.get('capex') or {}
    capex = CapexBreakdown(
        mechanical=float(capex_p.get('mechanical', 0.0)),
        electrical=float(capex_p.get('electrical', 0.0)),
        civil_bos=float(capex_p.get('civil_bos', 0.0)),
        soft_costs=float(capex_p.get('soft_costs', 0.0)),
        contingency_pct=float(capex_p.get('contingency_pct', 0.0)),
    )
    capex_total = capex.total()

    fin_p = payload.get('finance') or {}
    finance = FinanceParams(
        wacc_pct=float(fin_p.get('wacc_pct', 10.0)),
        project_life_years=int(fin_p.get('project_life_years', 20)),
        om_pct_of_capex=float(fin_p.get('om_pct_of_capex', 3.0)),
    )

    lcoe = lcoe_usd_per_kwh(capex_total, annual_kwh, finance)

    diesel_price = payload.get('diesel_price_per_kwh')
    payback = None
    if diesel_price not in (None, '', 0):
        diesel_price = float(diesel_price)
        annual_savings = annual_kwh * diesel_price
        payback = (capex_total / annual_savings) if annual_savings > 0 else None

    # table across presets (same capex/finance)
    preset_rows = []
    for nm, pr in WEIBULL_PRESETS.items():
        prob_p = weibull_bin_probs(edges, pr.k, pr.c)
        annual_kwh_p = aep_from_power_curve(delivered_kw, prob_p)
        lcoe_p = lcoe_usd_per_kwh(capex_total, annual_kwh_p, finance)
        preset_rows.append({'preset': nm, 'k': pr.k, 'c': pr.c, 'annual_kwh': annual_kwh_p, 'lcoe_usd_per_kwh': lcoe_p})

    return jsonify({
        'scenario': scenario_key,
        'annual_kwh': annual_kwh,
        'lcoe_usd_per_kwh': lcoe,
        'capex_total_usd': capex_total,
        'finance': finance.__dict__,
        'physics': {
            'drivetrain_eff': drivetrain_eff,
            'availability': availability,
            'inverter_kw': inverter_kw,
        },
        'payback_years_vs_diesel': payback,
        'curve_source': curve_source,
        'curves_on_bins': {'speed_mps': mids.tolist(), 'simulation_kw': p_on_bins.tolist(), 'aero_kw': aero_kw.tolist(), 'delivered_kw': delivered_kw.tolist(), 'prob': prob.tolist()},
        'weibull_presets_table': preset_rows,
    })

@app.route('/api/site-rollup', methods=['POST'])
def api_site_rollup():
    payload = request.json or {}
    scenario_key = _resolve_scenario_key(payload.get('scenario', '4x40ft_asymmetric'))
    modules = int(payload.get('modules', 1))
    spacing_factor = float(payload.get('spacing_factor', 2.0))
    land_cost_per_m2 = float(payload.get('land_cost_per_m2', 0.0))

    rub_key = scenario_key if scenario_key in RUBRIC_CONFIGS else ALIASES.get(scenario_key)
    rub = RUBRIC_CONFIGS.get(rub_key) if rub_key else None

    capex_per_module = float(payload.get('capex_per_module', rub.cost_usd if rub else 0.0))
    module_footprint_m2 = float(payload.get('module_footprint_m2', rub.footprint_m2 if rub else 0.0))
    annual_kwh_per_module = float(payload.get('annual_kwh_per_module', 0.0))

    roll = site_rollup(
        capex_per_module=capex_per_module,
        modules=modules,
        annual_kwh_per_module=annual_kwh_per_module,
        module_footprint_m2=module_footprint_m2,
        spacing_factor=spacing_factor,
        land_cost_per_m2=land_cost_per_m2,
    )
    roll.update({'scenario': scenario_key})
    return jsonify(roll)

@app.route('/api/phase-analysis', methods=['POST'])
def api_phase_analysis():
    payload = request.json or {}
    base_series = payload.get('base_series_kw') or []
    dt_s = float(payload.get('dt_s', 1.0))
    modules = int(payload.get('modules', 1))
    period_s = float(payload.get('period_s', max(1.0, dt_s * len(base_series))))
    inverter_kw = payload.get('inverter_kw')
    inverter_kw = float(inverter_kw) if inverter_kw not in (None, '', 0) else None

    base = np.array(base_series, dtype=float)
    offsets = payload.get('offsets_s')
    if not offsets:
        offsets = even_phase_offsets(modules, period_s)
    offsets = [float(x) for x in offsets]

    agg = apply_phase_offsets(base, offsets, dt_s=dt_s)

    # clipping on time-series
    if inverter_kw is not None:
        clipped = np.minimum(agg, inverter_kw)
        clip_loss_kwh = float(np.sum(np.maximum(agg - clipped, 0.0)) * dt_s / 3600.0)
    else:
        clipped = agg
        clip_loss_kwh = 0.0

    energy_kwh = float(np.sum(clipped) * dt_s / 3600.0)

    return jsonify({
        'modules': modules,
        'dt_s': dt_s,
        'offsets_s': offsets,
        'metrics_unclipped': smoothing_metrics(agg),
        'metrics_clipped': smoothing_metrics(clipped),
        'energy_kwh': energy_kwh,
        'clip_loss_kwh': clip_loss_kwh,
        'series_kw': {
            'agg': agg.tolist(),
            'clipped': clipped.tolist()
        }
    })


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
