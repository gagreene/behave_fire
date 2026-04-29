"""
Behave Fire Behavior Model - Python Implementation

This package provides vectorized (NumPy array) fire behavior modeling:
- Surface fire calculations
- Crown fire dynamics
- Tree mortality analysis
- Spot fire prediction
- Ignition potential assessment
- Safety zone calculations

Main entry point: BehaveRun class in behave.py
"""

from .behave import BehaveRun
from .components.fuel_models import FuelModels
from .components.species_master_table import SpeciesMasterTable
from .components.behave_units import (
    AreaUnits, BasalAreaUnits, LengthUnits, LoadingUnits, PressureUnits,
    SurfaceAreaToVolumeUnits, SpeedUnits, FractionUnits, SlopeUnits,
    DensityUnits, HeatOfCombustionUnits, HeatSinkUnits, HeatPerUnitAreaUnits,
    HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits,
    TemperatureUnits, TimeUnits,
)

__version__ = "1.0.0b2"
__author__ = "Behave Development Team"
__all__ = [
    'BehaveRun',
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

