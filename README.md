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
- Save high-res images with selectable resolution presets per aspect ratio
- 8 aspect ratios (16:9, 21:9, 32:9, 48:9, 4:3, 1:1, 3:2, 16:10) with letterboxed preview
- Color cycling with adjustable speed
- View rotation (right-click drag)
- Double-click smooth zoom into any point
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
| Right-click drag | Rotate view |
| Scroll wheel | Zoom (toward cursor) |
| Double-click | Smooth zoom into point |
| `[` / `]` | Previous / next fractal type |
| `,` / `.` | Previous / next preset location |
| Left / Right arrow | Previous / next color palette |
| Up / Down arrow | Increase / decrease iterations (+/-50) |
| `A` | Next aspect ratio |
| `S` | Save current location (opens name dialog) |
| `L` | Load a saved location (opens list) |
| `C` | Toggle color cycling |
| `R` | Reset to default view |
| `Tab` | Show / hide control panel |
| `Esc` | Quit |

All controls are also available in the on-screen panel.

## Save / Load

Press `S` or click **Save** to name and save your current view. All settings are preserved: position, zoom, rotation, fractal type, palette, iterations, and color cycling state.

Press `L` or click **Load** to open a list of saved locations. Click any entry to load it, or click the **X** button to delete it. Scroll with the mouse wheel if the list is long.

Saves are stored as JSON files in the `locations/` folder next to the script.

## Save Image

Press **Save Image** in the panel to render a high-resolution PNG. The dialog lets you pick a filename and cycle through resolution presets with Left/Right arrow keys (or the `< >` buttons). Available resolutions depend on the current aspect ratio — for example, 16:9 offers 1920x1080 up to 7680x4320, while 48:9 offers 5760x1080 up to 11520x2160.

Images are saved to the `images/` folder next to the script.

## Aspect Ratio

Use the **ASPECT** section in the panel (or press `A`) to cycle through 8 aspect ratios. The fractal render area adjusts within the window using letterboxing/pillarboxing — the window itself stays the same size, so the control panel remains fully accessible even at extreme ratios like 48:9 or 32:9.
