# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0b2] - 2026-04-29

### Added
- Added explicit unit parameters in `BehaveRun` for inputs that were previously fixed to internal units, including fuel moistures, canopy cover, canopy heights, canopy bulk density, crown ratio, scorch height, tree height, and DBH.

### Changed
- Reset `behave.py` facade input-unit defaults to the component-native US customary/base units where unit enums are accepted. Defaults now align with the internal modeling modules (`Fraction`, `Feet`, `PoundsPerCubicFoot`, `Inches`, etc.) instead of metric-facing defaults.
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

