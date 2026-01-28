# 🌬️ MSSDPPG Web UI with 3D Visualization

A modern, interactive web interface for the Modularized Self-Sustained Double Pendulum Power Generator simulator with real-time 3D visualization.

## Features

### 🎨 User Interface
- **Dark Theme Design** - Modern, responsive web-based UI
- **Real-time Controls** - Adjust parameters and run simulations instantly
- **Interactive 3D Visualization** - Three.js powered 3D pendulum animation
- **Live Charts** - Real-time power output, wind speed, and angle tracking
- **Activity Logging** - Real-time feedback on simulation progress

### ⚙️ Simulation Control
- **4 Predefined Scenarios**
  - 4×40ft Container (48 pendulums)
  - 1×20ft Container (24 pendulums)
  - Tower Facade (8 pendulums)
  - Mega 15m (1 pendulum)

- **Simulation Modes**
  - 2D Planar
  - 3D Spatial (with lateral dynamics)
  - Dual (both simultaneously)

- **Control Strategies**
  - Lock-Release (default)
  - Magnetic Push-Pull

- **Duration Options**
  - 6 hours
  - 12 hours

- **Assist Mode**
  - On (active assistance)
  - Off (passive mode)

### 📊 Real-time Monitoring
- Power generation output (kW)
- Wind speed profile (m/s)
- Pendulum angles (θ1, θ2)
- Coil temperature
- Energy generated (kWh)

### 🖥️ 3D Visualization
- **Interactive Camera** - Orbit around the pendulum with mouse controls
- **Auto-rotation** - Automatic camera rotation to visualize motion
- **Wireframe Mode** - Toggle wireframe for internal structure visibility
- **Visual Elements**
  - Golden hinge joint
  - Cyan pendulum arms
  - Red tip weight
  - Green wind-catching vane
  - Red wind speed indicator arrow

## Quick Start

### Installation

1. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   # Option 1: Using the startup script
   bash run_ui.sh

   # Option 2: Direct Python
   python3 app.py
   ```

3. **Open in browser**:
   ```
   http://localhost:5000
   ```

## Usage

### Running a Simulation

1. **Configure Parameters**:
   - Select a scenario from the dropdown
   - Choose simulation duration (6h or 12h)
   - Select simulation mode (2D, 3D, or Dual)
   - Choose control strategy
   - Enable/disable assist mode

2. **Start Simulation**:
   - Click the **"▶ Run Simulation"** button
   - Watch the progress bar update in real-time

3. **Monitor Results**:
   - View 3D pendulum motion in the center panel
   - Track real-time metrics in the left control panel
   - Observe power, wind, and angle charts on the right

### 3D Visualization Controls

| Action | Control |
|--------|---------|
| **Rotate View** | Drag with mouse |
| **Zoom** | Scroll wheel |
| **Reset Camera** | Click "Reset View" button |
| **Toggle Wireframe** | Click "Wireframe" button |
| **Auto Rotate** | Click "Auto Rotate" button |

## Architecture

### Backend (Flask)
- `app.py` - Flask server with simulation API
- `/api/scenarios` - Get available scenarios
- `/api/config` - Get configuration options
- `/api/simulate` - Start a new simulation
- `/api/sim-status` - Poll for real-time updates
- `/api/sim-result` - Get final results

### Frontend
- `templates/index.html` - Main UI layout
- `static/style.css` - Styling and responsive design
- `static/app.js` - Application logic and polling
- `static/visualizer.js` - Three.js 3D visualization

## API Endpoints

### GET /api/scenarios
Returns available scenarios with their parameters.

### GET /api/config
Returns configuration options (durations, modes, controls).

### POST /api/simulate
Start a new simulation.
```json
{
    "scenario": "4x40ft",
    "duration": "6h",
    "mode": "2d",
    "control": "lock",
    "assist": "on"
}
```

### GET /api/sim-status
Get current simulation status.
```json
{
    "running": true,
    "progress": 45,
    "frame_count": 120,
    "frames": [...]
}
```

### GET /api/sim-result
Get final simulation results and statistics.

## Performance Metrics

The UI displays real-time metrics:
- **Avg Power**: Average power output in kW
- **Peak Power**: Maximum power output in kW
- **Total Energy**: Energy generated in kWh
- **Max Temp**: Maximum coil temperature in °C

## Requirements

- Python 3.7+
- Flask >= 3.0.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0 (for core simulator)

## Browser Support

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

**Minimum requirements**: WebGL support for 3D visualization

## Troubleshooting

### Port 5000 already in use
```bash
python3 app.py --port 5001
```

### Simulation not starting
- Check Flask logs in terminal
- Ensure all dependencies are installed
- Verify wind_profile_standard.csv exists

### 3D visualization not loading
- Check browser console for WebGL errors
- Ensure browser supports WebGL
- Try a different browser

## Future Enhancements

- [ ] WebSocket support for faster updates
- [ ] Data export (CSV, JSON)
- [ ] Custom wind profile upload
- [ ] Batch simulation runs
- [ ] Optimization mode (parameter sweep)
- [ ] Historical data comparison
- [ ] Mobile responsive design improvements

## License

Same as main MSSDPPG project

## Support

For issues or questions:
1. Check the logs in terminal
2. Review browser console (F12)
3. Ensure all dependencies are installed
4. Verify file permissions on wind_profile_standard.csv
