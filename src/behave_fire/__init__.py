"""
Behave Fire Behavior Model - Python and Rust implementations

This package provides vectorized (NumPy array) fire behavior modeling:
- Surface fire calculations
- Crown fire dynamics
- Tree mortality analysis
- Spot fire prediction
- Ignition potential assessment
- Safety zone calculations
- Containment and support tools

Main entry point: the automatic BehaveRun facade in auto.py, which selects the
Python reference implementation or the native Rust extension (backend="auto",
"python", or "rust"). Explicit namespaces are behave_fire.python and
behave_fire.rust. The explicit Python compatibility class remains in behave.py.
"""

from .auto import BehaveBackendWarning, BehaveRun, available_backends, get_backend
from .components.fuel_models import FuelModels
from .components.species_master_table import SpeciesMasterTable
from .components.behave_units import (
    AreaUnits, BasalAreaUnits, LengthUnits, LoadingUnits, PressureUnits,
    SurfaceAreaToVolumeUnits, SpeedUnits, FractionUnits, SlopeUnits,
    DensityUnits, HeatOfCombustionUnits, HeatSinkUnits, HeatPerUnitAreaUnits,
    HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits,
    TemperatureUnits, TimeUnits,
)

from ._version import get_version as _get_version

__version__ = _get_version()
__author__ = "Behave Development Team"
__all__ = [
    'BehaveRun',
    'BehaveBackendWarning',
    'available_backends',
    'get_backend',
    'FuelModels',
    'SpeciesMasterTable',
    'AreaUnits',
    'BasalAreaUnits',
    'LengthUnits',
    'LoadingUnits',
    'PressureUnits',
    'SurfaceAreaToVolumeUnits',
    'SpeedUnits',
    'FractionUnits',
    'SlopeUnits',
    'DensityUnits',
    'HeatOfCombustionUnits',
    'HeatSinkUnits',
    'HeatPerUnitAreaUnits',
    'HeatSourceAndReactionIntensityUnits',
    'FirelineIntensityUnits',
    'TemperatureUnits',
    'TimeUnits',
]

