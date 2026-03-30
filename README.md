# behave_py

A pure Python port of the US Forest Service [BEHAVE](https://www.firelab.org/project/behave) wildfire fire-behavior modeling system. The library targets **100% feature parity** with the C++ reference implementation — same equations, same constant values, and matching method signatures (translated to snake_case). All computation runs on **NumPy arrays**, so it works equally well for a single cell or a 10-million-cell raster.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Unit System](#unit-system)
7. [Components](#components)
8. [Example — Landscape Fire Behavior](#example--landscape-fire-behavior)
9. [Running the Tests](#running-the-tests)
10. [Key Design Notes](#key-design-notes)

---

## Features

| Module | Capability |
|--------|------------|
| **Surface fire** | Rothermel (1972) spread model — ROS, flame length, fireline intensity, heat per unit area, backing/flanking spread, fire geometry |
| **Crown fire** | Rothermel (1991) & Scott–Reinhardt (2005) — transition ratio, active ratio, fire-type classification (surface / torching / conditional / active crown) |
| **Tree mortality** | Crown scorch (Van Wagner), bole char, and crown damage equations for 300+ species / 9 GACC regions |
| **Spotting** | Albini firebrand lofting model for burning piles, surface fires, and torching trees |
| **Ignition** | Firebrand and lightning ignition probability |
| **Safety zones** | Minimum separation distance, zone radius, and personnel capacity from radiant heat |
| **Containment** | Fried & Fried (1995) single-resource Runge-Kutta ODE containment simulation (standalone function) |
| **Support tools** | Fine dead fuel moisture (Nelson 2000 LUT), slope tool, vapor pressure deficit calculator |
| **Unit system** | 18 physical-quantity classes with full metric ↔ US-customary conversion |

---

## Project Structure

```
behave_py/
├── src/
│   ├── behave_fire/                 # Main package
│   │   ├── behave.py                # BehaveRun — public facade / entry point
│   │   ├── __init__.py              # Package exports
│   │   ├── components/
│   │   │   ├── behave_units.py      # Unit conversion backbone (18 classes)
│   │   │   ├── fuel_models.py       # ~60 standard fuel model records (Scott-Burgan 40 + classic 13)
│   │   │   ├── fuel_models_array.py # NumPy lookup arrays for vectorised fuel access
│   │   │   ├── surface.py           # Surface fire (Rothermel 1972)
│   │   │   ├── crown.py             # Crown fire dynamics
│   │   │   ├── fire_size.py         # Elliptical fire geometry helpers
│   │   │   ├── mortality.py         # Tree mortality
│   │   │   ├── spot.py              # Spotting distances
│   │   │   ├── ignite.py            # Ignition probability
│   │   │   ├── safety.py            # Safety zone calculations
│   │   │   ├── contain.py           # Containment simulation (standalone function)
│   │   │   ├── species_master_table.py  # Tree species database
│   │   │   ├── fine_dead_fuel_moisture_tool.py
│   │   │   ├── slope_tool.py
│   │   │   └── vapor_pressure_deficit_calculator.py
│   │   └── tests/
│   │       ├── conftest.py          # Belt-and-suspenders sys.path guard for pytest
│   │       └── test_behave.py       # Standalone test script (mirrors testBehave.cpp)
│   └── examples/
│       ├── landscape_fire_behavior.py   # Full 10 km × 10 km raster demo
│       ├── data/                        # Auto-generated GeoTIFF inputs
│       └── results/                     # GeoTIFF outputs + summary PNG
├── pytest.ini
├── CODEBASE.md                      # Architecture reference
└── README.md
```

---

## Requirements

- Python ≥ 3.10
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) *(used in the landscape example)*
- [rasterio](https://rasterio.readthedocs.io/) *(used in the landscape example)*
- [matplotlib](https://matplotlib.org/) *(used in the landscape example)*

Install the runtime dependencies:

```bash
pip install numpy scipy rasterio matplotlib
```

---

## Installation

Clone the repository and add `src/` to your Python path:

```bash
git clone <repo-url>
cd behave_py
pip install numpy scipy rasterio matplotlib
```

Then either run scripts from the project root (the examples do their own `sys.path` setup) or install in editable mode if you have a `pyproject.toml` / `setup.py`:

```bash
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from behave_fire.components.fuel_models import FuelModels
from behave_fire.behave import BehaveRun

fm     = FuelModels()
runner = BehaveRun(fm)

# --- Surface fire on a single cell ---
results = runner.do_surface_run(
    fuel_model_grid      = np.array([[124]]),          # GR4
    m1h                  = np.array([[0.06]]),          # 6% (as fraction)
    m10h                 = np.array([[0.07]]),
    m100h                = np.array([[0.08]]),
    mlh                  = np.array([[0.60]]),
    mlw                  = np.array([[0.90]]),
    wind_speed           = np.array([[25.0]]),          # km/h
    wind_speed_units     = 6,                           # KilometersPerHour
    wind_direction       = np.array([[225.0]]),         # degrees from north
    wind_orientation_mode= 'RelativeToNorth',
    slope                = np.array([[20.0]]),          # percent
    slope_units          = 1,                           # Percent
    aspect               = np.array([[180.0]]),         # degrees
    canopy_cover         = np.array([[0.40]]),          # fraction (0–1)
    canopy_height        = np.array([[15.0]]),          # feet
    crown_ratio          = np.array([[0.55]]),          # fraction (0–1)
    wind_height_mode     = 'TenMeter',
    out_units={
        'spread_rate'        : 3,   # MetersPerMinute
        'flame_length'       : 4,   # Meters
        'fireline_intensity'  : 4,  # KilowattsPerMeter
    },
)

print(f"ROS          : {results['spread_rate'][0, 0]:.2f} m/min")
print(f"Flame length : {results['flame_length'][0, 0]:.2f} m")
print(f"FLI          : {results['fireline_intensity'][0, 0]:.1f} kW/m")
```

### Crown fire + mortality

```python
crown = runner.do_crown_run(
    surface_results     = results,
    fuel_model_grid     = np.array([[124]]),
    m1h=np.array([[0.06]]), m10h=np.array([[0.07]]),
    m100h=np.array([[0.08]]), mlh=np.array([[0.60]]),
    mlw=np.array([[0.90]]),
    wind_speed          = np.array([[25.0]]),
    wind_speed_units    = 6,
    wind_direction      = np.array([[225.0]]),
    wind_orientation_mode = 'RelativeToNorth',
    slope               = np.array([[20.0]]),
    slope_units         = 1,
    aspect              = np.array([[180.0]]),
    canopy_base_height  = np.array([[19.7]]),   # feet (6 m × 3.28084)
    canopy_height       = np.array([[49.2]]),   # feet (15 m × 3.28084)
    canopy_bulk_density = np.array([[0.0075]]), # lb/ft³ (0.12 kg/m³ ÷ 16.0185)
    moisture_foliar     = np.array([[100.0]]),  # percent
)

print(f"Fire type : {int(crown['fire_type'][0, 0])}")   # 0=surface … 3=active crown
```

> **Note on `do_crown_run` inputs:** `canopy_base_height` and `canopy_height` are always in **feet** and `canopy_bulk_density` is in **lb/ft³** — these parameters do not have associated unit enums and must be converted before calling.

---

## Unit System

All physical quantities have a designated **base unit** stored internally (US customary). Unit conversion happens at call boundaries via integer enum values passed through the `wind_speed_units`, `slope_units`, `out_units`, etc. parameters.

| Category | Base Unit | Common metric enum |
|---|---|---|
| Speed | ft/min | `6` = km/h, `3` = m/min |
| Length / height | feet | `4` = metres |
| Slope | degrees | `1` = percent |
| Temperature | °F | `1` = °C |
| Fireline intensity | BTU/ft/s | `4` = kW/m |
| Heat per unit area | BTU/ft² | `1` = kJ/m² |
| Area | ft² | `2` = hectares |
| Moisture | fraction (0–1) | pass fractions; use `FractionUnits` for % ↔ fraction |

> **Important:** Moisture inputs (`m1h`, `m10h`, `m100h`, `mlh`, `mlw`, `moisture_foliar`) are always **fractions** (e.g. `0.06` for 6%). Passing percent values without converting will produce silently wrong results.

The full set of 18 unit classes exported from the package:
`AreaUnits`, `BasalAreaUnits`, `LengthUnits`, `LoadingUnits`, `PressureUnits`,
`SurfaceAreaToVolumeUnits`, `SpeedUnits`, `FractionUnits`, `SlopeUnits`,
`DensityUnits`, `HeatOfCombustionUnits`, `HeatSinkUnits`, `HeatPerUnitAreaUnits`,
`HeatSourceAndReactionIntensityUnits`, `FirelineIntensityUnits`,
`TemperatureUnits`, `TimeUnits`.

See `src/behave/components/behave_units.py` for the full enum listings.

---

## Components

### `BehaveRun` — the facade

The single public entry point for array/raster fire behavior calculations. Key methods:

| Method | Description |
|--------|-------------|
| `do_surface_run(...)` | Run Rothermel surface fire model across a grid |
| `do_crown_run(...)` | Run crown fire model (requires surface results) |
| `calculate_scorch_height(...)` | Compute scorch height from FLI, wind, and temperature |
| `calculate_crown_scorch_mortality(...)` | Tree mortality from scorch, DBH, crown ratio |
| `calculate_spotting_from_surface_fire(...)` | Max spotting distance from surface fire |
| `calculate_spotting_from_torching_trees(...)` | Spotting from torching trees |
| `calculate_spotting_from_burning_pile(...)` | Spotting from a burning pile |
| `calculate_fire_area(...)` | Elliptical fire area at elapsed time |
| `calculate_fire_perimeter(...)` | Elliptical fire perimeter at elapsed time |
| `calculate_fire_length(...)` | Fire ellipse major axis at elapsed time |
| `calculate_fire_width(...)` | Fire ellipse minor axis at elapsed time |

#### Wind adjustment factor

`do_surface_run()` accepts an optional `waf_method` parameter:

- `'UseCrownRatio'` *(default)* — WAF is derived automatically from `canopy_cover`, `canopy_height`, and `crown_ratio` using Albini & Baughman (1979).
- `'UserInput'` — use the value supplied in `user_waf` directly, bypassing automatic calculation.

### Fuel models

`FuelModels` ships with all ~60 standard models (original 13 + Scott–Burgan 40+). Custom models can be defined by directly populating the `fuel_models_` dict with the required field set.

### Containment simulation

`run_contain_sim_array()` in `contain.py` is a standalone vectorized function (not a `BehaveRun` method). It implements a simplified single-resource Runge-Kutta ODE containment model and returns a status grid with the following codes:

| Code | Meaning |
|------|---------|
| `1` | REPORTED |
| `3` | CONTAINED |
| `4` | OVERRUN |
| `5` | EXHAUSTED |
| `8` | TIME_LIMIT_EXCEEDED |

### Crown fire type classification

| Value | Meaning |
|---|---|
| `0` | Surface fire |
| `1` | Torching / passive crown fire |
| `2` | Conditional crown fire |
| `3` | Active crown fire |

---

## Example — Landscape Fire Behavior

`src/examples/landscape_fire_behavior.py` demonstrates a full end-to-end run over a synthetic **10 km × 10 km** landscape at **30 m resolution** (334 × 334 cells):

1. **Generates** synthetic terrain (elevation, slope, aspect), fuel model map, forest structure, and weather GeoTIFFs → `src/examples/data/`
2. **Runs** surface fire, crown fire, scorch height, tree mortality, and spotting across the entire raster
3. **Saves** 20+ output GeoTIFFs → `src/examples/results/`
4. **Produces** a 12-panel summary PNG `fire_behavior_summary.png`

```bash
cd src/examples
python landscape_fire_behavior.py
```

Expected runtime: ~10–30 seconds on a modern laptop.

---

## Running the Tests

The test suite mirrors the C++ `testBehave.cpp` reference and uses a custom accumulator. Run it directly:

```bash
cd src/behave_fire
python tests/test_behave.py
```

A pytest-compatible subset can be run from the project root, but currently requires test files matching `test_*_array.py` to be present in `src/behave/tests/`:

```bash
# from project root
pytest
```

---

## Key Design Notes

- **Pure Python / NumPy** — no C extensions required; BLAS-level vectorisation via NumPy broadcasting.
- **US customary internally** — all state is stored in feet, BTU, °F, etc. Metric conversion happens only at call boundaries.
- **No input validation** — out-of-range inputs return `0.0` silently, matching the C++ guard pattern.
- **Crown fuel is always FM10 / WAF=0.4** — hardcoded per Rothermel (1991); not configurable.
- **Moisture as fraction** — `0.06` = 6%; see the [Unit System](#unit-system) section.
- **Crown canopy inputs in base units** — `canopy_base_height` and `canopy_height` must be in feet, `canopy_bulk_density` in lb/ft³; no unit enum is applied to these parameters.
- **Wind orientation strings** — `'RelativeToUpslope'` or `'RelativeToNorth'`; misspellings silently default to upslope.

For a full architecture reference, gotcha catalogue, and Mermaid data-flow diagram see [`CODEBASE.md`](CODEBASE.md).

---

## References

- Rothermel, R. C. (1972). *A mathematical model for predicting fire spread in wildland fuels.* USDA Forest Service Research Paper INT-115.
- Rothermel, R. C. (1991). *Predicting behavior and size of crown fires in the Northern Rocky Mountains.* USDA Forest Service Research Paper INT-438.
- Scott, J. H. & Reinhardt, E. D. (2001). *Assessing crown fire potential by linking models of surface and crown fire behavior.* USDA Forest Service Research Paper RMRS-RP-29.
- Scott, J. H. & Burgan, R. E. (2005). *Standard fire behavior fuel models: a comprehensive set for use with Rothermel's surface fire spread model.* USDA Forest Service General Technical Report RMRS-GTR-153.
- Albini, F. A. (1976). *Estimating wildfire behavior and effects.* USDA Forest Service General Technical Report INT-30.
- Albini, F. A. & Baughman, R. G. (1979). *Estimating windspeeds for predicting wildland fire behavior.* USDA Forest Service Research Paper INT-221.
- Butler, B. W. & Cohen, J. D. (1998). *Firefighter safety zones: a theoretical model based on radiative heating.* International Journal of Wildland Fire, 8(2), 73–77.
- Nelson, R. M. (2000). *Prediction of diurnal change in 10-h fuel stick moisture content.* Canadian Journal of Forest Research, 30(7), 1071–1087.
- Fried, J. S. & Fried, B. D. (1996). *Simulating wildfire containment with realistic tactics.* Forest Science, 42(3), 267–281.
- Ryan, K. C. & Reinhardt, E. D. (1988). *Predicting postfire mortality of seven western conifers.* Canadian Journal of Forest Research, 18(10), 1291–1297.





