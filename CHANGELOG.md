# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Add machine-checked local C++ comparison receipts (`tools/cpp_receipt.py`) for use when the
  pinned upstream C++ reference changes. A receipt binds one platform's full local comparison
  to the tested commit, sources, fixture inventory, extension, toolchains, and host;
  verification reads the archived reports and re-applies the Surface caps and exact rules.
  The receipt stays local and is not committed, and the qualification stays
  withdrawn. Neither pull request merges nor release publication wait on it.
- Publication no longer runs live C++ qualification: `release.yml` and `release-crate.yml`
  validate the distributions and the crate only. `release.yml` grants its CI call the
  permissions the `plan` job needs, and a
  missing, unreadable, or invalid reuse record on `master` falls back to the full tier.
- Enforce the Surface caps and the exact binary64 rules for FireSize, Ignite, and
  Mortality in the pinned C++ comparisons whatever the approval state, and
  add deliberate-failure tests that perturb temporary fixture copies.
- Run the full platform and packaging CI once per promotion. Pull requests into
  `pre_release` and merges into it run the quick tier; promotion and hotfix pull
  requests into `master` run the full tier, and the following `master` push reuses
  that result only when its Git tree equals the tested merge tree (otherwise it runs
  the full tier). A `plan` job (`tools/ci_plan.py`) chooses the tier, and manual
  dispatches can target `musl`, `sdist`, or `musl-and-sdist`. C++ calibration in a
  manual run now needs `cpp_qualification=true`.
- Move comparisons of Python and Rust outputs with pinned C++ outputs out of CI and
  routine runs: the Python tests carry the `cpp_comparison` marker, which the default
  pytest run and CI deselect, and the Rust comparison tests are `#[ignore]`d and run
  with `--include-ignored`. Run them when the pinned C++ reference changes. CI keeps
  Python/Rust parity, reference-data integrity, packaging, and installation checks.
- Correct the README and landscape example Crown runs: `do_crown_run` reads surface
  spread rate, fireline intensity, and heat per unit area in base units, so the
  examples now pass a base-unit surface run instead of metric display output, which
  understated crown-stage spread (5.34 instead of about 30.5 ft/min in the README
  case). The README also documents the canopy unit selectors, labels model 124 as
  GS4, and states the shipped mortality coverage (24 species with crown scorch and
  bole char equations). Calculations are unchanged.
- Publish to PyPI only from a tag push; a manual run on a tag ref can no longer
  reach the production publishing job.
- Link the GCC runtime statically into the musl wheel, which no longer bundles a
  separate `libgcc_s` shared library; the build job rejects any `libgcc_s` member or
  dependency.
- Ship the Python/Rust acceptance fixtures (`tests/data/rust_port`) in the source
  distribution, verify them against the release commit, and run the shipped fixture
  test and full parity comparison from the extracted archive.
  `tools/compare_backends.py` now fails when it finds no acceptance scenario.
- Align Python and Rust with the pinned Behave fuel records for SCAL16 (152) and SCAL14 (153).
  SCAL16 dead 1-hour load changes from 2.20 to 2.25 US tons/acre. SCAL14 dead 100-hour load
  changes from 1.10 to 1.05 and live herbaceous load from 1.40 to 1.45 US tons/acre.
  Calculations using these built-in models now use the corrected loads, resolving F-019.

- Restart version numbering at `0.1.0` under `MAJOR.MINOR.PATCH`; the package
  remains beta. Versions `1.0.0b1` and `1.0.0b2` were never published to a
  package registry. The current candidate is `0.1.0rc1` (Cargo `0.1.0-rc.1`), and
  TestPyPI rehearsals use `MAJOR.MINOR.PATCHrcN`.
- Leave the first-party version fields of `rust/crates/behave-py/Cargo.toml`,
  `rust/crates/behave-fire/Cargo.toml`, and `rust/Cargo.lock` out of the C++
  qualification source digests, so version-only changes need no new
  qualification. Every other manifest and lock byte stays bound, and a missing,
  duplicated, or malformed version field fails closed.
- Remove the development-era timing apparatus from the repository: the
  `docs/performance/` evidence, `docs/RUST_BACKEND_BENCHMARKS.md`, the benchmark,
  allocation, and acceptance tools with their tests, the `tools/perf_probe` crate,
  and the private `allocation-probe` Cargo feature of the bindings. Git history
  retains them. Correctness checks from those tests moved to retained tests,
  including the Rust raster parity check in the renamed
  `tools/binding_field_tests` crate, and `tools/performance_cases.py` became
  `tools/raster_cases.py`.
- Add `tools/time_standard_runs.py`, which prints Surface and Crown run times for
  each available backend on one machine. Timing is reported only and never used
  as a pass or fail check.

- Update the pinned Behave reference to `3ca33d5b919d4253b411bfb6464b4c872dda6fc9`, replay the reviewed inputs, and retire the corrected moisture-of-extinction discrepancy. The Surface caps remain `1e-12` relative and absolute.

- Enforce the approved Surface fixture ceilings: 806 cases, 6 MiB raw and
  512 KiB gzip for canonical fixtures, and 512 KiB raw and 64 KiB gzip for
  standalone copies. Bind inventory evidence to the selected snapshot and
  require six exact Surface function witnesses plus active qualification for
  readiness, without claiming coverage of other assigned methods.

- Match the pinned C++ moisture-damping multiplication order in Python and Rust
  without changing coefficients or extinction guards. Refresh the reviewed
  last-bit reference differences, including derived fixture intermediates.
- Preserve fractional extinction values for the 23 inconsistent upstream fuel
  records and add separately declared Percent-unit custom controls. The original
  numbered C++ observations remain unchanged. Manual GNU/MSVC and seven-platform
  calibration passed the unchanged owner-approved `1e-12` relative and absolute
  caps, activating Surface comparisons with documented original-unit differences.
- Keep native extension binaries in wheels while excluding compiled files from
  source archives. Global binary exclusions previously stripped the extension
  from wheels carrying a native platform tag.

- Align Python and Rust Surface boundaries with official Behave C++ at
  `f3b2943`: retain effective wind and elliptical shape for positive baseline
  spread at or below `1e-7 ft/min`, and return zero Surface flame length when
  fireline intensity is strictly below `1e-7 BTU/ft/s`. The threshold itself
  uses the existing flame equation. These intentionally change earlier Python
  results near extinction; zero-denominator guards remain in place.

- Remove the Python, Rust core, and native containment module pending a licensing
  review: the upstream C++ containment headers declare GNU Public License 2.
  The module is preserved locally and is expected to return as a separate project.
  Git history retains the module.

- Follow the official C++ `FuelModels::isFuelModelDefined` bounds (1–256) for fuel
  lookups
  ([source](https://github.com/firelab/behave/blob/f3b2943e69abca942e795c5b1b45276f6338abb5/src/behave/fuelModels.cpp#L863)).
  The C++ stores only indices 0–255 and runs its zero-load check with unchecked
  lookups before the definition check, so it has no defined result for IDs below
  0 or above 255 (ID 256 passes its range check but is outside its storage).
  The behave_fire result for those IDs is a project decision, not a compared
  C++ behavior.
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
  3.18-5.52x serial and 6.06-13.44x with four workers on the declared host,
  measured at source `9be8a50`.

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
- With the Rust backend, real physical inputs (including float16 and float32) are computed in float64, masked values become NaN, and physical results are float64 arrays. Result dtypes, masks, and overflow, NaN, and threshold outcomes can therefore differ from the Python backend. Select `backend="python"` for the historical dtype and mask behavior. See `docs/BACKEND_DIFFERENCES.md`.

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

### Changed

- Moved the test suite and fixtures to root `tests/`. Updated pytest discovery, package exclusions, documentation, and the standalone harness to use repository-relative or file-relative paths and canonical package imports.
- Aligned heat conversions with upstream's Btu convention: the kJ/kg-to-Btu/lb factor changes from `0.429923` to `0.429592` (about 0.077% lower), and the reverse factor changes from `1 / 0.429923` to `2.32779`. This is upstream convention alignment, not a dimensional correction; metric custom-fuel heat values and converted outputs change.

### Deprecated

- Lightning ignition probability now emits `DeprecationWarning`. Upstream removed this model because its scientific basis is no longer considered reliable. Existing signatures, re-exports, and numerical results remain available for compatibility; firebrand ignition remains supported.

### Added

- Native Rust backend. The Cargo workspace under `rust/` contains the standalone `behave-fire` core crate (version `0.1.0`, independent of the Python version, no runtime or build dependencies) and the unpublished `behave-py` PyO3 bindings built as `behave_fire._rust`. The core covers every public calculation: surface and crown fire with their stages, mortality, spotting, ignition, safety zones, fine dead fuel moisture, vapor pressure deficit, and slope and ellipse helpers, plus typed unit conversions and embedded fuel and species tables.
- Backend selection: `BehaveRun(fuel_models, *, backend="auto" | "python" | "rust", threads=1)`, `available_backends()`, `get_backend()`, and `BehaveBackendWarning`. Explicit namespaces are `behave_fire.python` and `behave_fire.rust`; historical component imports continue to use Python. `auto` prefers Rust and falls back to Python with a once-per-process `BehaveBackendWarning` when the extension is absent or unusable; forced Rust raises.
- Native `threads` setting (`1` serial, `0` all available, positive count capped by work) using scoped threads with deterministic output order.
- Typed native calculation stages (`engine`) with a reusable fuel snapshot, stage resumption from editable intermediates, and fused overrides (`SurfaceTweaks`, `CrownTweaks`). Borrowed raster columns (`raster`) with per-cell overrides, optional native validity, and caller-driven tiles. Python reaches the staged engine through `behave_fire.rust.SurfaceGrid` and `CrownGrid`, and public Rust Surface and Crown calls use the float64 raster engine (see "Float64 models and backend routing").
- Python/Rust parity, ownership, serialization, packaging, and workflow tests, and tools for backend comparison and platform probing (`tools/compare_backends.py`).
- Contracts and procedures: `docs/RUST_BACKEND.md`, `docs/BACKEND_DIFFERENCES.md`, `docs/RUST_ENGINE.md`, `docs/RELEASING.md`, and `docs/CRATES_IO.md`.
- Pytest regression coverage for unit conversions, metric fuel records, static flags, and lightning compatibility, using portable fixtures without requiring a C++ checkout.

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
