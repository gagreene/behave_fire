"""
Behave Fire Behavior Model - Python Implementation
Complete Python port of the Behave fire behavior model with 100% C++ feature parity.

This package provides comprehensive fire behavior modeling capabilities including:
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
from .components.surface import Surface
from .components.crown import Crown
from .components.mortality import Mortality
from .components.spot import Spot
from .components.ignite import Ignite
from .components.safety import Safety
from .components.behave_units import (
    AreaUnits, BasalAreaUnits, LengthUnits, LoadingUnits, PressureUnits,
    SurfaceAreaToVolumeUnits, SpeedUnits, FractionUnits, SlopeUnits,
    DensityUnits, HeatOfCombustionUnits, HeatSinkUnits, HeatPerUnitAreaUnits,
    HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits,
    TemperatureUnits, TimeUnits
)

__version__ = "1.0.0"
__author__ = "Behave Development Team"
__all__ = [
    'BehaveRun',
    'FuelModels',
    'SpeciesMasterTable',
    'Surface',
    'Crown',
    'Mortality',
    'Spot',
    'Ignite',
    'Safety',
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

