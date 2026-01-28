# MSSDPPG Live Simulator

MSSDPPG Live Simulator provides a Flask web app and CLI for deterministic simulator-based power curves, AEP, LCOE, and whole-site rollups. It includes multi-tab UI tooling for physics, wind resource, power curves, economics, smoothing, and site metrics.

## Local setup

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` to access the UI.

## CLI usage

```bash
python app.py scenarios
python app.py lcoe --aep-kwh 1500000 --capex 1200000
```

## Tests

```bash
pytest -q
```

## Render deployment

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `bash start.sh`

The app uses `gunicorn` and binds to `0.0.0.0:$PORT`.

## Example histogram CSV

```csv
speed_mps,prob
3,0.03
4,0.06
5,0.12
6,0.16
7,0.18
8,0.16
9,0.12
10,0.09
11,0.05
12,0.03
```

## File tree

```
repo/
  app.py
  start.sh
  Procfile
  requirements.txt
  .gitignore
  README.md
  docs/
    MSSDPPG_FINAL_RUBRIC_v5.0_CORRECTED.md.pdf
    MSSDPPG_Patent_Application_COMPLETE.md.pdf
    MSSDPPG_Simulation_Code_Hints.md
  mssdppg/
    __init__.py
    scenarios.py
    physics/
      __init__.py
      dp2d.py
      dp3d.py
      energy.py
    wind/
      __init__.py
      weibull.py
      histogram.py
      presets.py
    powercurve/
      __init__.py
      builder.py
      cache.py
    economics/
      __init__.py
      capex.py
      aep.py
      lcoe.py
      finance.py
      land.py
      smoothing.py
    exports/
      __init__.py
      tables.py
  static/
    app.js
    styles.css
  templates/
    index.html
  tests/
    test_wind.py
    test_lcoe.py
    test_api_smoke.py
```
