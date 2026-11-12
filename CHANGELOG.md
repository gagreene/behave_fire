# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Align fuel lookup bounds with official C++ `FuelModels::isFuelModelDefined`
  ([source](https://github.com/firelab/behave/blob/f3b2943e69abca942e795c5b1b45276f6338abb5/src/behave/fuelModels.cpp#L863)).
  Both backends map IDs outside 1–256, including nonfinite Python fuel IDs,
  to undefined row zero and retain the existing undefined-fuel zero-spread path.
  Tables contain 257 rows. Custom registration outside that range returns `False`;
  custom models 257–260 are no longer supported. Model 256 remains supported.
  This intentionally replaces endpoint clamping, without changing output storage
  types or introducing NaN, code, or flag rules for undefined fuel identifiers.

- Preserve Python's original input domain when auto selects Python because the
  extension is unavailable; skip Rust preflight/snapshots on that route and the
  unused auto registry snapshot for explicit Rust. Add proven NumPy complex
  scalar elapsed-time fallback and clarify the once-per-process warning policy.
- Validate staged grid fuel identifiers, binary flags, and supplied-fuelbed keys;
  allow a supplied source without an unused fuel-ID input. Support inferred
  high-rank shapes and owned output transfer above 32 axes. Fuel equality now
  describes public physical fields independently of private cache history.

- Automatic selection prefers Rust and warns once when unavailable; verified
  complex operations route to Python with a separate warning before computation.
  Rust uses float64/NaN physical arrays, while Python retains its original
  dtype/mask behavior. These installation-dependent differences are intentional.
- Certify the final Windows float64 model campaign: all 40 cases and 80
  first/warm confidence bounds pass. At 100k/1M cells, warmed speedups are
  3.18-5.52x serial and 6.06-13.44x with four workers on the declared host.
  Retain complete arrays, confidence/memory/scalar/allocation evidence, and
  independent additional-layout/containment/continuous diagnostics.

### Staged Rust raster sessions

- Add public Python `behave_fire.rust.SurfaceGrid` and `CrownGrid`, backed by the
  standalone Rust `GridSession` and borrowed `FieldView`. Sessions own inputs,
  options, fuels or supplied Surface fuelbeds, and persistent overrides.
- Support explicit stages, typed physical/code/flag edits, persistent overrides,
  requested/kept fields, and worker selection. `run_remaining` preserves injected
  values (including NaN) and recovers freed dependencies. Explicit producer runs
  replace their transient injections; `run_fused` restarts from inputs and
  persistent overrides. Python getters return independent arrays.
- These grid APIs require Rust and accept real numeric inputs. They have no
  Python fallback or pickle support. Existing facades retain API4 serialization.
  Both languages have a `staged_raster` example using base-unit injection.

### Float64 models and backend routing

- Move public Rust Surface/Crown models to the standalone float64 raster engine.
  Real physical inputs widen once; masks become NaN; physical outputs are plain
  float64 arrays. Classifications stay int32, with -1 for unavailable values.
  Preserve output shapes and selected-branch dependencies. This intentionally
  differs from Python dtype/mask behavior, including under automatic selection.
- Complete float64 normalization across all 89 numerical operations and scalar
  state tools. Keep controls, status/classification codes, and flags typed.
  Remove NumPy dtype, masked-payload, object/weak-scalar, fixed-width integer,
  float32, and SVML emulation. The standalone core now forbids unsafe code.
  API4 facade/tool serialization remains value-only.
- Add direct preinitialized component buffers, real dependency groups for
  independent outputs, and shared fuelbed field equations for components and
  staged sessions. Empty groups never evaluate fake prototype cells.
- Automatic FireSize state moves completely to a verified Python counterpart
  before supported complex operations, emits a warning, and stays pinned there.
  Other complex state methods fail clearly without native error replay.
- Warn once when auto cannot use Rust, including on pure wheels. Proven complex
  input compatibility selects a complete Python operation before coercion, with
  an independent warning. General object data and native errors do not trigger
  fallback. Preserve frozen registry snapshots and actual last-completed backend
  across copies and serialization.

### Rust core output storage

- Reuse immutable wind coefficients and packing powers in Surface and across Crown forward spread and backsolve for static physical fuel. Physical-fuel and actual fuelbed dependency checks prevent stale reuse; dynamic fuel retains per-cell evaluation. Floating-point product order and availability thresholds are unchanged.
- Model batches write selected fields directly into final output buffers using safe disjoint worker slices. Integer classification columns retain int32 storage through export.
- The unreleased Rust `into_named_vectors` adapter is replaced by typed `into_named_columns`; Rust callers must match the returned `ColumnValues` storage.
- Add a configured Rust `run_model` entry point with float64 physical calculations, selected final columns, prepared fuels or supplied fuelbeds, and persistent per-cell overrides. Missing inputs propagate by field dependency; explicitly invalid cells produce NaN physical outputs and classification -1. See the standalone `configured_tiles` example.

### Breaking changes

- Drop Python 3.10 support; require Python 3.11 or newer. Standard CPython 3.11-3.14 are the certification targets. Free-threaded interpreters and PyPy are not supported.
- Source/Git/editable/`--no-binary` project builds now use Maturin and require Rust. Binary native and universal pure-Python wheels share metadata and runtime APIs.
- The package-root `BehaveRun` is now an automatic backend facade built by composition. Its class identity and default `repr` differ from the historical class, and an instance is not an `isinstance` of `behave_fire.behave.BehaveRun`. Use `behave_fire.python.BehaveRun` or `behave_fire.behave.BehaveRun` for legacy type checks and subclassing. Calculation signatures, output dictionaries, root `FuelModels`, species types, and unit enum identities are unchanged.
- With the Rust backend, float16 inputs are widened to float32, so results can be float32 where the Python backend returns float16, and overflow, NaN, and threshold outcomes can differ. Select `backend="python"` for the historical float16 behavior. See `docs/BACKEND_DIFFERENCES.md`.

### Packaging

- Add native `cp311-abi3` wheels for seven platforms (Linux x86-64 and aarch64 on manylinux 2_17, Alpine/musl x86-64 on musllinux 1_2, macOS x86-64 and arm64, Windows x86-64 and ARM64), a universal `py3-none-any` fallback wheel, isolated installed-artifact validation, and a tracked-input source-distribution check. The distribution inventory check requires exactly one native wheel per platform, one pure wheel, and one matching source archive. Free-threaded CPython and PyPy are not supported.
- Add build-only CI runs, including manual dispatch, a standalone-crate archive check with a Rust-only consumer, and release workflows for the wheels and the crate. None of them publishes without separate owner authorization. See `docs/RELEASING.md` and `docs/CRATES_IO.md`.
- Require NumPy >=2.3.3 for Python >=3.14 or native Windows ARM64; keep NumPy >=2.1.3 elsewhere.
- Add optional `test` dependencies without raster libraries, and a uv lockfile for dependency-only source tests.
- Preserve MIT and upstream BSD-3-Clause notices in both native and pure artifacts.

### Removed

- Removed the unpublished conda recipe and its obsolete publishing instructions. Python package delivery continues through wheels and source distributions.

### Fixed

- Corrected centimetre surface-area-to-volume conversion: `1 cm^-1` now converts to `30.48 ft^-1`, with the inverse conversion updated consistently.
- Converted depth, fuel loading, heat, and SAVR for metric fuel models 110–111, 155–159, 166–172, and 190–193 to internal US customary units, following upstream `f3b2943`. Existing fractional moisture-of-extinction corrections are preserved. Surface results for these models change.
- Aligned 23 Scott and Burgan static-model flags with upstream: SH2–8, TU2/4/5, TL1–9, and SB1–4. The flags now report static behavior; scalar and mixed-grid checks show unchanged surface results for their zero-herbaceous-load records.
- Corrected containment descriptions and examples to reflect the simplified fixed-step model, its input-shape limitation, and actual output. Containment calculations are unchanged.

### Changed

- Moved the test suite and fixtures to root `tests/`. Updated pytest discovery, package exclusions, documentation, and the standalone harness to use repository-relative or file-relative paths and canonical package imports.
- Aligned heat conversions with upstream's Btu convention: the kJ/kg-to-Btu/lb factor changes from `0.429923` to `0.429592` (about 0.077% lower), and the reverse factor changes from `1 / 0.429923` to `2.32779`. This is upstream convention alignment, not a dimensional correction; metric custom-fuel heat values and converted outputs change.

### Deprecated

- Lightning ignition probability now emits `DeprecationWarning`. Upstream removed this model because its scientific basis is no longer considered reliable. Existing signatures, re-exports, and numerical results remain available for compatibility; firebrand ignition remains supported.

### Added

- Native Rust backend. The Cargo workspace under `rust/` contains the standalone `behave-fire` core crate (version `0.1.0`, independent of the Python version, no runtime or build dependencies) and the unpublished `behave-py` PyO3 bindings built as `behave_fire._rust`. The core covers every public calculation: surface and crown fire with their stages, mortality, spotting, ignition, safety zones, fine dead fuel moisture, vapor pressure deficit, slope and ellipse helpers, and containment, plus typed unit conversions and embedded fuel and species tables.
- Backend selection: `BehaveRun(fuel_models, *, backend="auto" | "python" | "rust", threads=1)`, `available_backends()`, `get_backend()`, and `BehaveBackendWarning`. Explicit namespaces are `behave_fire.python` and `behave_fire.rust`; historical component imports continue to use Python. `auto` selects Python silently when the extension is absent and with a once-per-process warning when it is present but unusable; forced Rust raises.
- Native `threads` setting (`1` serial, `0` all available, positive count capped by work) using scoped threads with deterministic output order.
- Typed native calculation stages (`engine`) with a reusable fuel snapshot, stage resumption from editable intermediates, and fused overrides (`SurfaceTweaks`, `CrownTweaks`). Borrowed raster columns (`raster`) with per-cell overrides, optional native validity, and caller-driven tiles. These are Rust interfaces; the Python binding does not yet route public calls through them.
- Python/Rust parity, ownership, serialization, packaging, and workflow tests, and tools for backend comparison, platform probing, paired benchmarking, and performance acceptance (`tools/compare_backends.py`, `tools/benchmark_backends.py`, `tools/check_performance.py`).
- Contracts and measurements: `docs/RUST_BACKEND.md`, `docs/BACKEND_DIFFERENCES.md`, `docs/RUST_ENGINE.md`, `docs/RUST_BACKEND_BENCHMARKS.md`, `docs/RELEASING.md`, and `docs/CRATES_IO.md`.
- Pytest regression coverage for unit conversions, metric fuel records, static flags, and lightning compatibility, using portable fixtures without requiring a C++ checkout.

### Known limitations

- The Rust backend is not yet faster than Python for typical workloads. Measured single-host results show Python ahead for single-cell and 10,000-cell public calls, with parallel Rust ahead only on very large surface and crown rasters; mixed-dtype or masked Crown inputs are much slower on Rust. No speedup is promised. See `docs/RUST_BACKEND_BENCHMARKS.md` and `docs/RUST_ENGINE.md`.
- Neither the Python package nor the Rust crate has been published to PyPI or crates.io.

## [1.0.0b2] - 2026-04-29

### Added
- Added explicit unit parameters in `BehaveRun` for inputs that were previously fixed to internal units, including fuel moistures, canopy cover, canopy heights, canopy bulk density, crown ratio, scorch height, tree height, and DBH.

### Changed
- Reset `behave.py` facade input-unit defaults to the component-native US customary/base units where unit enums are accepted. Defaults now align with the internal modeling modules (`Fraction`, `Feet`, `PoundsPerCubicFoot`, `Inches`, etc.) instead of metric-facing defaults.
- Refactored the crown fire pipeline so `run_crown_fire()` is a driver over individually callable `calculate_*` step functions for crown surface spread, crown heat/intensity, critical thresholds, ratios, fire-type classification, crown fraction burned, passive crown behavior, and final behavior assignment.
- Updated the landscape example and test harness to pass explicit unit enums and values through the revised facade.
- Standardized package and documentation references from `behave_py` / `BEHAVE` wording to `behave_fire` / `Behave7` where applicable.

### Fixed
- Added `.gitignore` coverage for generated example raster inputs and results under `src/examples/data/` and `src/examples/results/` so large derived files are not tracked accidentally.

## [1.0.0b1] - 2026-03-29

Pre-release beta — pending code review before stable publication.

### Added
- Initial release.
- Vectorized NumPy surface fire model (Rothermel 1972).
- Crown fire model — Rothermel (1991) and Scott & Reinhardt (2001).
- Tree mortality — crown scorch and bole char equations for 300+ species.
- Spotting distances for surface fires, burning piles, and torching trees (Albini 1979).
- Firebrand and lightning ignition probability.
- Safety zone calculations (Butler & Cohen 1998).
- Containment simulation — single-resource Runge-Kutta ODE model.
- Fine dead fuel moisture lookup table (Nelson 2000).
- Slope tool and vapor pressure deficit calculator.
- 18-class unit conversion system (US customary ↔ metric).
- ~60 standard fuel models (Scott-Burgan 40 + original 13).
- Full landscape example over a synthetic 10 km × 10 km raster.

