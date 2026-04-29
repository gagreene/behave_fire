"""
Behave Fire Behavior Model - Components Package

This package contains all the core fire behavior calculation components:

- behave_units: Unit conversion system (16 unit categories, functions + enum classes)
- fuel_models: Fuel model database and access
- species_master_table: Species data and regional availability
- surface: Vectorized surface fire behavior calculations
- crown: Vectorized crown fire dynamics calculations
- mortality: Vectorized tree mortality calculations
- spot: Vectorized spotting fire prediction
- ignite: Vectorized ignition potential calculations
- safety: Vectorized safety zone calculations
- vapor_pressure_deficit_calculator: Vectorized VPD calculations
- fine_dead_fuel_moisture_tool: Vectorized FDMF table lookup
- fuel_models_array: Vectorized fuel property lookup
"""

from .behave_units import (
    AreaUnits, BasalAreaUnits, LengthUnits, LoadingUnits, PressureUnits,
    SurfaceAreaToVolumeUnits, SpeedUnits, FractionUnits, SlopeUnits,
    DensityUnits, HeatOfCombustionUnits, HeatSinkUnits, HeatPerUnitAreaUnits,
    HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits,
    TemperatureUnits, TimeUnits,
    speed_to_base, speed_from_base, length_to_base, length_from_base,
    area_to_base, area_from_base, fraction_to_base, fraction_from_base,
    temp_to_base, temp_from_base, slope_to_base, slope_from_base,
    pressure_to_base, pressure_from_base,
    fireline_intensity_to_base, fireline_intensity_from_base,
    hpua_to_base, hpua_from_base,
    loading_to_base, loading_from_base,
    density_to_base, density_from_base,
    time_to_base, time_from_base,
)
from .fuel_models import FuelModels
from .species_master_table import SpeciesMasterTable, SpeciesMasterTableRecord
from .surface import (
    build_particle_arrays, calculate_fuelbed_intermediates,
    calculate_reaction_intensity, calculate_spread_rate,
    calculate_fire_area, calculate_fire_perimeter,
    calculate_fire_length, calculate_fire_width,
)
from .crown import (
    assign_final_crown_fire_behavior,
    calculate_critical_crown_fire_spread_rate,
    calculate_critical_surface_fireline_intensity,
    calculate_crown_fire_ratios,
    calculate_crown_fraction_burned,
    calculate_crown_heat_and_intensity,
    calculate_crown_surface_fire,
    calculate_crowning_surface_fire_spread_rate,
    calculate_passive_crown_fire_behavior,
    calculate_surface_fire_critical_spread_rate,
    classify_crown_fire_type,
    coerce_crown_fire_inputs,
    get_surface_fire_inputs,
    run_crown_fire,
)
from .mortality import (
    calculate_scorch_height, build_mortality_lookup,
    calculate_crown_scorch_mortality,
)
from .spot import (
    calculate_spotting_from_surface_fire,
    calculate_spotting_from_burning_pile,
    calculate_spotting_from_torching_trees,
)
from .ignite import (
    calculate_firebrand_ignition_probability,
    calculate_lightning_ignition_probability,
)
from .safety import calculate_safety_zone
from .vapor_pressure_deficit_calculator import calculate_vpd
from .fine_dead_fuel_moisture_tool import calculate_fine_dead_fuel_moisture
from .fuel_models_array import build_fuel_lookup_arrays

__all__ = [
    # Unit enum classes
    'AreaUnits', 'BasalAreaUnits', 'LengthUnits', 'LoadingUnits', 'PressureUnits',
    'SurfaceAreaToVolumeUnits', 'SpeedUnits', 'FractionUnits', 'SlopeUnits',
    'DensityUnits', 'HeatOfCombustionUnits', 'HeatSinkUnits', 'HeatPerUnitAreaUnits',
    'HeatSourceAndReactionIntensityUnits', 'FirelineIntensityUnits',
    'TemperatureUnits', 'TimeUnits',
    # Unit conversion functions
    'speed_to_base', 'speed_from_base', 'length_to_base', 'length_from_base',
    'area_to_base', 'area_from_base', 'fraction_to_base', 'fraction_from_base',
    'temp_to_base', 'temp_from_base', 'slope_to_base', 'slope_from_base',
    'pressure_to_base', 'pressure_from_base',
    'fireline_intensity_to_base', 'fireline_intensity_from_base',
    'hpua_to_base', 'hpua_from_base',
    'loading_to_base', 'loading_from_base',
    'density_to_base', 'density_from_base',
    'time_to_base', 'time_from_base',
    # Data tables
    'FuelModels', 'SpeciesMasterTable', 'SpeciesMasterTableRecord',
    'build_fuel_lookup_arrays',
    # Calculation functions
    'build_particle_arrays', 'calculate_fuelbed_intermediates',
    'calculate_reaction_intensity', 'calculate_spread_rate',
    'calculate_fire_area', 'calculate_fire_perimeter',
    'calculate_fire_length', 'calculate_fire_width',
    'assign_final_crown_fire_behavior',
    'calculate_critical_crown_fire_spread_rate',
    'calculate_critical_surface_fireline_intensity',
    'calculate_crown_fire_ratios',
    'calculate_crown_fraction_burned',
    'calculate_crown_heat_and_intensity',
    'calculate_crown_surface_fire',
    'calculate_crowning_surface_fire_spread_rate',
    'calculate_passive_crown_fire_behavior',
    'calculate_surface_fire_critical_spread_rate',
    'classify_crown_fire_type',
    'coerce_crown_fire_inputs',
    'get_surface_fire_inputs',
    'run_crown_fire',
    'calculate_scorch_height', 'build_mortality_lookup',
    'calculate_crown_scorch_mortality',
    'calculate_spotting_from_surface_fire', 'calculate_spotting_from_burning_pile',
    'calculate_spotting_from_torching_trees',
    'calculate_firebrand_ignition_probability',
    'calculate_lightning_ignition_probability',
    'calculate_safety_zone',
    'calculate_vpd',
    'calculate_fine_dead_fuel_moisture',
]

