# behave_fire

Native models, component helpers, units, and state tools calculate physical values in float64. Float16,
float32, and integer physical inputs widen once; masked real values become NaN,
and physical results are plain float64 arrays. Classifications remain int32,
with -1 for unavailable classifications. Python retains its original dtype and
masked-array behavior. These differences also apply when `backend="auto"`
selects Rust; see [Backend differences](docs/BACKEND_DIFFERENCES.md).

Auto uses Rust when available. Otherwise it selects Python and warns once per
process. A separate warning accompanies supported complex-input calls routed
to Python before native calculation. Explicit Rust rejects complex/object
arithmetic. General object input does not trigger automatic fallback.

A NumPy-based Python port of the US Forest Service [Behave<sup>7</sup>](https://www.firelab.org/project/behave) fire behavior modeling system, supporting scalar and array calculations. The package has two calculation backends: the original Python/NumPy implementation (the behavioral reference) and a native Rust implementation selected automatically when its compiled extension is installed. The same Rust core is also a standalone crate, `behave-fire`, for Rust applications.

Upstream alignment is ongoing; full C++ parity has not been established. Fuel data and heat constants follow `firelab/behave` at `f3b2943`, with an intentional correction of metric-model moisture percentages to fractions. Containment is a simplified growth model, and lightning ignition remains available only as a deprecated compatibility API. Rust performance is being validated against the public Python API; see [Performance](#performance).

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Choosing a Calculation Backend](#choosing-a-calculation-backend)
7. [Rust Crate and Raster Engine](#rust-crate-and-raster-engine)
8. [Performance](#performance)
9. [Unit System](#unit-system)
10. [Components](#components)
11. [Example — Landscape Fire Behavior](#example--landscape-fire-behavior)
12. [Running the Tests](#running-the-tests)
13. [Documentation](#documentation)
14. [Key Design Notes](#key-design-notes)
15. [References](#references)

---

## Features

| Module | Capability |
|--------|------------|
| **Surface fire** | Rothermel (1972) spread model — ROS, flame length, fireline intensity, heat per unit area, backing/flanking spread, fire geometry |
| **Crown fire** | Rothermel (1991) & Scott–Reinhardt (2005) — transition ratio, active ratio, fire-type classification (surface / torching / conditional / active crown) |
| **Tree mortality** | Crown scorch (Van Wagner), bole char, and crown damage equations for 300+ species / 9 GACC regions |
| **Spotting** | Albini firebrand lofting model for burning piles, surface fires, and torching trees |
| **Ignition** | Firebrand ignition probability; deprecated lightning calculation retained for compatibility |
| **Safety zones** | Minimum separation distance, zone radius, and personnel capacity from radiant heat |
| **Containment** | Simplified fixed-step ellipse growth and suppression-line accumulation (standalone function) |
| **Support tools** | Fine dead fuel moisture (Nelson 2000 LUT), slope tool, vapor pressure deficit calculator |
| **Unit system** | 17 physical-quantity classes with full metric ↔ US-customary conversion |
| **Backends** | Python reference, native Rust, and an automatic selector (`python` / `rust` / `auto`) |
| **Rust crate** | Standalone `behave-fire` core with typed scalar stages, raster columns, stage injection, and tiles |

---

## Project Structure

```
behave_fire/
├── src/
│   ├── behave_fire/                 # Python package
│   │   ├── __init__.py              # Package exports (automatic BehaveRun, FuelModels, units)
│   │   ├── auto.py                  # Automatic backend facade, available_backends, get_backend
│   │   ├── behave.py                # Explicit Python BehaveRun (the behavioral reference)
│   │   ├── python/                  # Explicit Python namespace (behave_fire.python)
│   │   ├── rust/                    # Explicit native namespace (behave_fire.rust) and bindings adapters
│   │   ├── _backend_contract.py     # Native extension API version and required capabilities
│   │   ├── _version.py              # Installed-version lookup with source-checkout fallback
│   │   └── components/              # Python reference implementations
│   │       ├── behave_units.py      # Unit conversion backbone (17 classes)
│   │       ├── fuel_models.py       # Standard fuel model records (Scott-Burgan 40, classic 13, metric)
│   │       ├── fuel_models_array.py # NumPy lookup arrays for vectorised fuel access
│   │       ├── surface.py           # Surface fire (Rothermel 1972)
│   │       ├── crown.py             # Crown fire dynamics
│   │       ├── fire_size.py         # Elliptical fire geometry helpers
│   │       ├── mortality.py         # Tree mortality
│   │       ├── spot.py              # Spotting distances
│   │       ├── ignite.py            # Ignition probability
│   │       ├── safety.py            # Safety zone calculations
│   │       ├── contain.py           # Containment simulation (standalone function)
│   │       ├── species_master_table.py  # Tree species database
│   │       ├── fine_dead_fuel_moisture_tool.py
│   │       ├── slope_tool.py
│   │       └── vapor_pressure_deficit_calculator.py
│   └── examples/
│       ├── landscape_fire_behavior.py   # Full 10 km × 10 km raster demo
│       ├── data/                        # Auto-generated GeoTIFF inputs
│       └── results/                     # GeoTIFF outputs + summary PNG
├── rust/                            # Cargo workspace
│   └── crates/
│       ├── behave-fire/             # Standalone Rust core (publishable crate)
│       │   ├── src/                 #   engine/ (scalar stages), raster/ (columns, tiles), surface/, crown/, ...
│       │   ├── examples/            #   standalone.rs, raster_tweaks.rs
│       │   └── tests/               #   Golden/parity tests with compact packaged fixtures
│       └── behave-py/               # Unpublished PyO3 bindings, built as behave_fire._rust
├── tests/                           # Python tests, fixtures, and the legacy harness (test_behave.py)
├── tools/                           # Parity, benchmark, packaging, and artifact-validation tools
├── docs/                            # Contracts, procedures, and measurements (see Documentation)
├── .github/workflows/               # CI, wheel validation, and release workflows
├── pyproject.toml                   # Maturin build configuration and Python metadata
├── uv.lock                          # Locked development/CI dependencies
├── pytest.ini
├── CHANGELOG.md
├── LICENSE / NOTICE / LICENSES/     # MIT license plus bundled NumPy-derived and Intel SVML BSD-3-Clause notices
└── README.md
```

---

## Requirements

- Python ≥ 3.11. The certification targets are standard CPython 3.11, 3.12, 3.13, and 3.14. PyPy and free-threaded CPython (3.13t/3.14t) are not supported.
- [NumPy](https://numpy.org/) ≥ 2.1.3 (≥ 2.3.3 on Python ≥ 3.14 or native Windows ARM64, where earlier releases have no binary wheel).
- Rust ≥ 1.78 only to build from source or to use the `behave-fire` crate. Wheel users do not need Rust.
- [SciPy](https://scipy.org/), [rasterio](https://rasterio.readthedocs.io/), and [matplotlib](https://matplotlib.org/) *(optional, for the landscape example only)*.

---

## Installation

### From a release

Release artifacts are built and validated by CI (see [docs/RELEASING.md](docs/RELEASING.md)); publication to PyPI is a separate, owner-authorized step. After a release is published:

```sh
pip install behave-fire
pip install "behave-fire[examples]"   # adds SciPy, rasterio, and matplotlib for the example
```

Each release provides:

| Artifact | Platforms | Backend after install |
|----------|-----------|-----------------------|
| Native `cp311-abi3` wheels | Linux x86-64 and aarch64 (manylinux 2_17), Alpine/musl x86-64 (musllinux 1_2), macOS x86-64 and arm64, Windows x86-64 and ARM64 | Rust, selected by `backend="auto"` |
| Universal `py3-none-any` wheel | Any other platform with a supported Python | Python (no extension) |
| Source distribution | Any (requires Rust to build) | Rust, built locally |

Installing a behave-fire wheel never requires compiling anything. Whether the *whole environment* installs without a compiler also depends on NumPy having a binary wheel for your interpreter and platform; use `pip install --only-binary=:all: behave-fire` to enforce that. Compiler-free claims are made only for the interpreter/platform combinations that CI actually installs and tests.

### From source

Source, Git, editable, and `--no-binary` installs build the extension with Maturin and require Rust and a native linker:

```sh
git clone https://github.com/gagreene/behave_fire.git
cd behave_fire
uv sync --locked --extra test --extra dev        # builds the extension and installs test tools
uv run --no-sync python -m pytest --require-rust
```

### Python-only use of a checkout

The Python implementation needs no build step. Put `src/` on the Python path (or run scripts from the repository root; the examples set their own path). With no extension present, `backend="auto"` selects Python and emits a once-per-process `BehaveBackendWarning`:

```sh
uv sync --locked --no-install-project --extra test --extra dev   # dependencies only; never installs the project
uv run --no-sync python -m pytest
```

Do not use `pip install -e .` for this route: it builds the extension and needs Rust.

---

## Quick Start

```python
import numpy as np
from behave_fire import BehaveRun, FuelModels

fm     = FuelModels()
runner = BehaveRun(fm)          # backend="auto": Rust when its extension is installed, otherwise Python

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
print(f"Backend      : {runner.backend_in_use}")
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

## Choosing a Calculation Backend

The package root selects an implementation when a model is constructed. Importing the package or its explicit Python namespace does not probe the native extension.

```python
from behave_fire import BehaveRun, FuelModels, available_backends, get_backend
from behave_fire.python import BehaveRun as PythonBehaveRun
from behave_fire.rust import BehaveRun as RustBehaveRun

fuels = FuelModels()
model = BehaveRun(fuels, backend="auto", threads=1)
print(model.backend_in_use)  # "python" or "rust"; read-only
python_model = BehaveRun(fuels, backend="python")
rust_model = BehaveRun(fuels, backend="rust", threads=4)
contain = get_backend("rust").contain.run_contain_sim_array
```

`available_backends()` reports Python and, when compatible and complete, Rust. `get_backend("python")` and `get_backend("rust")` expose their model, registries, and component modules. Standalone imports include `from behave_fire.python import contain` and `from behave_fire.rust import contain`. Historical imports such as `behave_fire.components.surface` continue to use Python. Nested component import paths under the new namespaces are not required; access their module attributes.

`backend="auto"` prefers a compatible, complete Rust extension. An absent extension (the universal wheel, or an unbuilt checkout), loader error, incompatible API, or incomplete advertised operation inventory selects Python and emits `BehaveBackendWarning`, a `RuntimeWarning` subclass, once per process, with its cause. An availability query can cache the extension status but does not consume that warning. Forced Rust raises a chained `ImportError` for any unavailable or incomplete extension. Explicit native imports permit compatible incomplete development extensions; an unsupported operation raises `NotImplementedError`. Calculation and facade errors propagate and never trigger a fallback.

Configure strict fallback handling before the first warned automatic selection:

```python
import warnings
from behave_fire import BehaveBackendWarning, BehaveRun, FuelModels

warnings.simplefilter("error", BehaveBackendWarning)
model = BehaveRun(FuelModels())
```

A warning promoted to an error does not consume the warning, so subsequent attempts remain strict. Changing filters after a warning was already emitted cannot make that process's once-only warning retroactively strict. Create a new process to reset the cached extension outcome and warning state.

The automatic facade uses composition. Its class identity and default `repr` therefore differ from the historical Python facade, and an automatic model is not an `isinstance` of `behave_fire.behave.BehaveRun`. Use `from behave_fire.python import BehaveRun` (the historical class alias) or `from behave_fire.behave import BehaveRun` for legacy subclassing and type checks. Root registry and unit enum identities are unchanged. Historical static calls on the root class, such as `BehaveRun.calculate_fire_area(...)`, keep using Python without probing Rust. Calls on model instances always use their selected backend. All 11 calculation signatures remain discoverable.

Each model caches its numerical fuel snapshot at construction. Later changes to the input registry do not update that model. The automatic facade accepts either registry type, converting a native registry snapshot for its Python route. Explicit Python accepts its original registry; use a native registry's `to_python()` conversion when needed. Native registries also expose `FuelModels.from_python(registry)`. Pickle and deepcopy preserve the selected backend, thread setting, and actual cached snapshot. Native unpickling requires a compatible native extension and never silently selects Python. Deep copies own independent numerical state; shallow copies retain the reference sharing behavior.

Native `threads=1` runs serially, `threads=0` uses available parallelism, and a positive count requests that many workers, capped by actual work. Negative and non-integer counts are rejected on the native route. The automatic Python route ignores `threads`; explicit Python retains its original constructor without that argument. Create another model to choose a different backend.

Rust widens float16 inputs to float32 before normal entry-point normalization. Automatic selection can therefore return float32 where Python returns float16; overflow, NaNs, and classification thresholds may also differ. Select Python when the historical float16 semantics are required. All intentional differences are listed in [docs/BACKEND_DIFFERENCES.md](docs/BACKEND_DIFFERENCES.md).

---

## Rust Crate and Raster Engine

`rust/crates/behave-fire` is an independently versioned Rust library (`behave_fire`) with no runtime or build dependencies. It does not need Python, NumPy, a C++ compiler, or this repository. The Python extension is a thin PyO3 layer (`rust/crates/behave-py`, never published) over the same core, so native Python results and standalone Rust results come from one implementation of the equations.

- **Complete calculation coverage:** surface and crown pipelines with every public stage, mortality, spotting, ignition, safety zones, fine dead fuel moisture, vapor pressure deficit, slope and ellipse helpers, and the fixed-step containment model. The `BehaveRun` Rust facade provides all eleven calculation routes.
- **Typed scalar stages (`engine`):** `SurfacePlan` and `CrownPlan` run a cell from numerical inputs and an immutable `PreparedFuels` snapshot. Intermediates such as `FuelbedIntermediates` can be computed, edited, and resumed.
- **Stage injection:** `SurfaceTweaks` and `CrownTweaks` replace reaction intensity, rates, wind or slope factors, or crown thresholds at the stage that would produce them, so downstream outputs use the injected value.
- **Raster columns (`raster`):** `SurfaceRaster` and `CrownRaster` borrow structure-of-arrays inputs (uniform scalars or per-cell slices), validate them once, and write typed output columns. `run_tile` processes a caller-chosen range. The library performs no raster file I/O.
- **Bounded parallelism:** scoped standard-library threads with deterministic output order; no global thread pool.

Run the complete default/custom-fuel example and the raster-override example from the crate directory:

```sh
cargo run --example standalone
cargo run --example raster_tweaks
```

The raster engine and tweak columns are Rust interfaces. Python users keep their existing staged helpers (for example `run_surface_fire(ri, ib, ...)`) for injection, and the Python binding does not yet route public calls through the raster engine (see [Performance](#performance)). See [docs/RUST_ENGINE.md](docs/RUST_ENGINE.md) for the stage and raster contract and [docs/CRATES_IO.md](docs/CRATES_IO.md) for packaging. The crate has not been published to crates.io; publication and name ownership are separate, owner-authorized steps.

---

## Performance

The current API4 [public serial checkpoint](docs/performance/2026-11-08-plain-normalization-public-checkpoint.md) passed **all 56 bounds** in both a fresh screen and the required seven-new-pair confirmation. Confirmed median Rust speedups were **2.93–4.09x at 100,000 cells** across Surface, Crown, and five dtype/mask layouts. Every 100k lower 95% bound exceeded the unchanged 2x requirement. Measurements used original-input Python timing on one Windows x86-64 host with CPython 3.12.8 and NumPy 2.1.3.

The previously failing heterogeneous Surface 10k warmed endpoint now passes: its confirmed median-ratio estimate is 2.65x and lower 95% bound is 2.23x against 1.25x required. Earlier failed samples remain preserved; they were not discarded or pooled into the new campaign.

Four-thread, million-cell, memory, and final release-platform certification remain pending. These serial results do not guarantee speed on other hosts or workloads. Earlier measurements remain in [the benchmark history](docs/RUST_BACKEND_BENCHMARKS.md). Select Python when original dtype/mask behavior is required.

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

The full set of 17 unit classes exported from the package:
`AreaUnits`, `BasalAreaUnits`, `LengthUnits`, `LoadingUnits`, `PressureUnits`,
`SurfaceAreaToVolumeUnits`, `SpeedUnits`, `FractionUnits`, `SlopeUnits`,
`DensityUnits`, `HeatOfCombustionUnits`, `HeatSinkUnits`, `HeatPerUnitAreaUnits`,
`HeatSourceAndReactionIntensityUnits`, `FirelineIntensityUnits`,
`TemperatureUnits`, `TimeUnits`.

See `src/behave_fire/components/behave_units.py` for the full enum listings.

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

`FuelModels` ships with the standard models: the original 13, Scott–Burgan 40, and the metric (V-, M-, F-) models, stored in US customary units. Custom models can be registered with `set_custom_fuel_model()` (with unit arguments) or by populating the `fuel_models_` dict with the required field set. The Rust registry offers typed add/remove/get operations.

### Containment simulation

`run_contain_sim_array()` in `contain.py` is a standalone vectorized function (not a `BehaveRun` method), also available as `get_backend("rust").contain.run_contain_sim_array`. A fixed-step loop accumulates ellipse growth and suppression line; it does not implement RK4 or the full C++ ContainSim solver. No perimeter-point arrays are returned. See [the containment documentation](docs/CONTAINMENT_MODULE.md) for shape requirements, limitations, and runnable examples. Status codes are:

| Code | Meaning |
|------|---------|
| `1` | REPORTED |
| `3` | CONTAINED |
| `4` | OVERRUN |
| `5` | EXHAUSTED |
| `8` | TIME_LIMIT_EXCEEDED |

### Ignition support

Firebrand ignition remains supported. `calculate_lightning_ignition_probability()` emits `DeprecationWarning` because upstream Behave removed the model, citing an unreliable scientific basis. The function and its package re-export retain their existing signatures and numerical results for compatibility. No replacement lightning model or removal version is specified.

### Crown fire type classification

| Value | Meaning |
|---|---|
| `0` | Surface fire |
| `1` | Torching / passive crown fire |
| `2` | Conditional crown fire |
| `3` | Active crown fire |

---

## Example — Landscape Fire Behavior

`src/examples/landscape_fire_behavior.py` demonstrates a full end-to-end run over a synthetic **10 km × 10 km** landscape at **30 m resolution** (334 × 334 cells). It needs the optional example dependencies (`pip install "behave-fire[examples]"`, or SciPy, rasterio, and matplotlib):

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

Python tests (the Python-only route, no Rust required; run from the repository root):

```bash
uv sync --locked --no-install-project --extra test --extra dev
uv run --no-sync python -m pytest
```

Native Python tests need a built extension (`uv sync --locked --extra test --extra dev`, which requires Rust). `--require-rust` makes a missing or incompatible extension a failure instead of a skip:

```bash
uv run --no-sync python -m pytest --require-rust
```

Pytest imports the package through `src` and collects `test_*_array.py` under root `tests/`. Fixtures are loaded relative to their test files from `tests/data/`. The native and parity suites compare Rust against the explicit Python backend (including dtype, mask, shape, and exception behavior) and cover packaging, dispatch, serialization, ownership, and workflow contracts. These checks do not establish full C++ numerical parity.

Rust tests (from the repository root; `rust/` has no `Cargo.toml` at the root, so pass the manifest path):

```bash
cargo test   --manifest-path rust/Cargo.toml -p behave-fire --locked
cargo fmt    --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets --locked -- -D warnings
cargo +1.78.0 test --manifest-path rust/Cargo.toml -p behave-fire --locked   # minimum supported Rust
```

Run the legacy accumulator harness separately, also from the project root:

```bash
python tests/test_behave.py
```

Expected lightning deprecation warnings may appear. The standalone harness locates `src/` relative to its own file and uses the same canonical `behave_fire` imports as pytest. No machine-specific paths or `PYTHONPATH` settings are required.

Useful tools under `tools/`: `compare_backends.py` (Rust versus the frozen Python oracle), `benchmark_backends.py` and `check_performance.py` (paired timing and acceptance gates), and the packaging checks described in [docs/RELEASING.md](docs/RELEASING.md).

---

## Documentation

| Document | Contents |
|----------|----------|
| [CHANGELOG.md](CHANGELOG.md) | Release notes, including unreleased changes |
| [docs/RUST_BACKEND.md](docs/RUST_BACKEND.md) | Backend, selection, ownership, dtype, and packaging contract |
| [docs/BACKEND_DIFFERENCES.md](docs/BACKEND_DIFFERENCES.md) | Intentional differences between the Rust and Python backends |
| [docs/RUST_ENGINE.md](docs/RUST_ENGINE.md) | Typed stages, raster API, injection, performance targets, and checkpoint results |
| [docs/RUST_BACKEND_BENCHMARKS.md](docs/RUST_BACKEND_BENCHMARKS.md) | Measured timings, memory, and the performance investigation |
| [docs/RELEASING.md](docs/RELEASING.md) | Building, validating, and releasing Python wheels and the source distribution |
| [docs/CRATES_IO.md](docs/CRATES_IO.md) | Packaging and publishing the standalone Rust crate |
| [docs/CONTRIBUTION_PROCEDURES.md](docs/CONTRIBUTION_PROCEDURES.md) | Branches, PRs, test records, and release routing |
| [docs/CONTAINMENT_MODULE.md](docs/CONTAINMENT_MODULE.md) | The containment function, its limits, and examples |
| [docs/FBFM_DESCRIPTIONS.md](docs/FBFM_DESCRIPTIONS.md) | Fire behavior fuel model descriptions |
| [AGENTS.md](AGENTS.md) | Repository rules for coding agents |

---

## Key Design Notes

- **Python reference, optional Rust** — the Python implementation is the behavioral reference; the Rust backend must match it within documented tolerances. Source-only Python remains available, and compatible native wheels select the Rust backend automatically.
- **US customary internally** — all state is stored in feet, BTU, °F, etc. Metric conversion happens only at call boundaries.
- **No input validation** — out-of-range inputs return `0.0` silently, matching the C++ guard pattern.
- **Crown fuel is always FM10 / WAF=0.4** — hardcoded per Rothermel (1991); not configurable.
- **Moisture as fraction** — `0.06` = 6%; see the [Unit System](#unit-system) section.
- **Crown canopy inputs in base units** — `canopy_base_height` and `canopy_height` must be in feet, `canopy_bulk_density` in lb/ft³; no unit enum is applied to these parameters.
- **Wind orientation strings** — `'RelativeToUpslope'` or `'RelativeToNorth'`; misspellings silently default to upslope.
- **Fuel snapshots** — each `BehaveRun` copies the fuel registry at construction; later registry edits do not affect an existing model.

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
