# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-29

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

