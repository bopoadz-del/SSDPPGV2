# MSSDPPG Live Simulator + Wind/Economics Cockpit

This repo contains:

- **Ultra‑Realistic simulator** (2D + 3D) with control modes (Lock–Release / Push–Pull / None)
- **Live web UI** (Flask + vanilla JS + Plotly) with:
  - Live power / angles / wind plots
  - Wind resource modeling (**Weibull** presets or **histogram CSV** upload)
  - **Power curve** builder (short constant‑wind simulations)
  - **AEP + LCOE** calculator (finance + losses + inverter clipping)
  - **Phase offsets & curtailment** analysis (smoothing + clipping loss)
  - **Whole‑site rollups** (modules, spacing factor, land cost)
- **Rubric‑aligned presets** (Scenario presets follow Rubric v5.0: fixed middle/tip masses per scale; arm masses scale with (L/2)^2)

## Source documents

The authoritative references are included in `docs/`:

- `MSSDPPG_FINAL_RUBRIC_v5.0_CORRECTED.md.pdf`
- `MSSDPPG_Patent_Application_COMPLETE.md.pdf`
- `MSSDPPG_Simulation_Code_Hints.md`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

python app.py
```

Open the UI:

- http://localhost:5000

## CLI simulation (headless)

```bash
python MSSDPPG_UltraRealistic_v2.py --scenario 4x40ft_asymmetric --duration 6h --mode 2d --control lock --assist
```

Useful flags (see `--help`):

- `--kickstart` : initialize with a large release angle (stress/free‑run testing)
- `--wind_constant 7.0` : run at constant wind instead of a profile
- `--no_safety_clamp` : bypass motion safety limits (use carefully)
- `--export_csv` : dump time series to `outputs/`

## Wind histogram CSV format

Upload a CSV with either:

- `speed_mps, prob`  
or  
- `speed_mps, count` (will be normalized to probability)

Example:

```csv
speed_mps,count
2,10
3,25
4,40
5,30
6,15
```

## Outputs

Generated artifacts go to `outputs/` (examples):

- simulation time series CSVs
- event detection CSVs (mega free‑fall / lock candidates)
- cached power‑curve results (`powercurve_cache_*.json`)

## Notes on interpretation

- The **economics layer** can use:
  - **Simulation curve** (recommended for internal consistency), or
  - **Aero Cp curve**: `P(v) = 0.5 * ρ * A * Cp * v^3` (useful sanity check).  
  Choose in the UI under **Economics → Curve source**.

- The simulator’s electrical scaling and losses are modeled at a **system level** and may need calibration against measured prototype data.
