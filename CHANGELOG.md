# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

