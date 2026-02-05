# GPU Fractal Viewer

Real-time GPU-accelerated fractal explorer using PyCUDA and Pygame. Renders 11 different fractal types with interactive pan, zoom, color cycling, 8 color palettes, and save/load for your favorite locations.

## Fractal Types

- **Mandelbrot** - Classic z^2 + c
- **Burning Ship** - Uses absolute values, creates ship-like shapes
- **Tricorn** - Complex conjugate variant (Mandelbar)
- **Julia: Dragon** - c = -0.8 + 0.156i
- **Julia: Dendrite** - c = 0.0 + 1.0i
- **Julia: Spiral** - c = -0.4 + 0.6i
- **Julia: San Marco** - c = -0.75 + 0.0i
- **Julia: Douady Rabbit** - c = -0.123 + 0.745i
- **Multibrot z^3** - Higher-power Mandelbrot
- **Multibrot z^4** - Even higher-power variant
- **Phoenix** - Feedback fractal using previous iteration values

## Features

- Two-pass GPU rendering (compute + colorize) for efficient color cycling
- Standard double precision + double-double precision for deep zoom (auto-switches at 10^13x)
- 8 color palettes (Classic, Inferno, Ocean, Electric, Emerald, Twilight, Grayscale, Rainbow)
- 15 preset locations across all fractal types
- Save/load locations with custom names (stored as JSON in `locations/` folder)
- Color cycling with adjustable speed
- Resizable window
- Coordinate tracking with Python Decimal for lossless accumulation

## Requirements

- NVIDIA GPU with CUDA support
- Python 3
- PyCUDA, Pygame, NumPy

## Quick Start

```bash
./run.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 fractal_viewer.py
```

## Controls

| Input | Action |
|---|---|
| Mouse drag | Pan |
| Scroll wheel | Zoom (toward cursor) |
| `[` / `]` | Previous / next fractal type |
| `,` / `.` | Previous / next preset location |
| Left / Right arrow | Previous / next color palette |
| Up / Down arrow | Increase / decrease iterations (+/-50) |
| `S` | Save current location (opens name dialog) |
| `L` | Load a saved location (opens list) |
| `C` | Toggle color cycling |
| `R` | Reset to default view |
| `Tab` | Show / hide control panel |
| `Esc` | Quit |

All controls are also available in the on-screen panel.

## Save / Load

Press `S` or click **Save** to name and save your current view. All settings are preserved: position, zoom, fractal type, palette, iterations, and color cycling state.

Press `L` or click **Load** to open a list of saved locations. Click any entry to load it, or click the **X** button to delete it. Scroll with the mouse wheel if the list is long.

Saves are stored as JSON files in the `locations/` folder next to the script.
