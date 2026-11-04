"""Explicit native crown adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def assign_final_crown_fire_behavior(
        is_surface_fire: Union[bool, np.ndarray],
        is_passive_crown: Union[bool, np.ndarray],
        surface_ros: Union[float, np.ndarray],
        surface_hpua: Union[float, np.ndarray],
        surface_fli: Union[float, np.ndarray],
        passive_ros: Union[float, np.ndarray],
        passive_hpua: Union[float, np.ndarray],
        passive_fli: Union[float, np.ndarray],
        passive_fl: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        crown_hpua: Union[float, np.ndarray],
        crown_fli: Union[float, np.ndarray],
        crown_fl: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param is_surface_fire: Input matching the reference API.
    :param is_passive_crown: Input matching the reference API.
    :param surface_ros: Input matching the reference API.
    :param surface_hpua: Input matching the reference API.
    :param surface_fli: Input matching the reference API.
    :param passive_ros: Input matching the reference API.
    :param passive_hpua: Input matching the reference API.
    :param passive_fli: Input matching the reference API.
    :param passive_fl: Input matching the reference API.
    :param crown_ros: Input matching the reference API.
    :param crown_hpua: Input matching the reference API.
    :param crown_fli: Input matching the reference API.
    :param crown_fl: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.assign_final_crown_fire_behavior', locals())

def calculate_critical_crown_fire_spread_rate(
        canopy_bulk_density: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param canopy_bulk_density: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_critical_crown_fire_spread_rate', locals())

def calculate_critical_surface_fireline_intensity(
        canopy_base_height: Union[float, np.ndarray],
        moisture_foliar: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param canopy_base_height: Input matching the reference API.
    :param moisture_foliar: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_critical_surface_fireline_intensity', locals())

def calculate_crown_fire_ratios(
        surface_fli: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        critical_surface_fli: Union[float, np.ndarray],
        critical_crown_ros: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param surface_fli: Input matching the reference API.
    :param crown_ros: Input matching the reference API.
    :param critical_surface_fli: Input matching the reference API.
    :param critical_crown_ros: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_crown_fire_ratios', locals())

def calculate_crown_fraction_burned(
        surface_ros: Union[float, np.ndarray],
        surface_critical_ros: Union[float, np.ndarray],
        crowning_surface_ros: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param surface_ros: Input matching the reference API.
    :param surface_critical_ros: Input matching the reference API.
    :param crowning_surface_ros: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_crown_fraction_burned', locals())

def calculate_crown_heat_and_intensity(
        surface_hpua: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        canopy_base_height: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
        canopy_bulk_density: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param surface_hpua: Input matching the reference API.
    :param crown_ros: Input matching the reference API.
    :param canopy_base_height: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :param canopy_bulk_density: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_crown_heat_and_intensity', locals())

def calculate_crown_surface_fire(
        lut: dict,
        S: tuple,
        m1h: Union[float, np.ndarray],
        m10h: Union[float, np.ndarray],
        m100h: Union[float, np.ndarray],
        mlh: Union[float, np.ndarray],
        mlw: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        slope_deg: Union[float, np.ndarray],
        aspect: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param lut: Input matching the reference API.
    :param S: Input matching the reference API.
    :param m1h: Input matching the reference API.
    :param m10h: Input matching the reference API.
    :param m100h: Input matching the reference API.
    :param mlh: Input matching the reference API.
    :param mlw: Input matching the reference API.
    :param wind_speed: Input matching the reference API.
    :param wind_speed_units: Input matching the reference API.
    :param wind_direction: Input matching the reference API.
    :param wind_orientation_mode: Input matching the reference API.
    :param slope_deg: Input matching the reference API.
    :param aspect: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_crown_surface_fire', locals())

def calculate_crowning_surface_fire_spread_rate(
        ri_crown: Union[float, np.ndarray],
        ib_crown: dict,
        critical_crown_ros: Union[float, np.ndarray],
        S: tuple,
        canopy_height: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param ri_crown: Input matching the reference API.
    :param ib_crown: Input matching the reference API.
    :param critical_crown_ros: Input matching the reference API.
    :param S: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_crowning_surface_fire_spread_rate', locals())

def calculate_passive_crown_fire_behavior(
        surface_ros: Union[float, np.ndarray],
        surface_hpua: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        canopy_hpua: Union[float, np.ndarray],
        cfb: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param surface_ros: Input matching the reference API.
    :param surface_hpua: Input matching the reference API.
    :param crown_ros: Input matching the reference API.
    :param canopy_hpua: Input matching the reference API.
    :param cfb: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_passive_crown_fire_behavior', locals())

def calculate_surface_fire_critical_spread_rate(
        critical_surface_fli: Union[float, np.ndarray],
        surface_hpua: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param critical_surface_fli: Input matching the reference API.
    :param surface_hpua: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.calculate_surface_fire_critical_spread_rate', locals())

def classify_crown_fire_type(
        transition_ratio: Union[float, np.ndarray],
        active_ratio: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param transition_ratio: Input matching the reference API.
    :param active_ratio: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.classify_crown_fire_type', locals())

def coerce_crown_fire_inputs(
        fuel_model_grid: Union[int, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        wind_direction: Union[float, np.ndarray],
        slope: Union[float, np.ndarray],
        slope_units: int=0,
        aspect: Union[float, np.ndarray]=0.0,
        canopy_base_height: Union[float, np.ndarray]=0.0,
        canopy_height: Union[float, np.ndarray]=0.0,
        canopy_bulk_density: Union[float, np.ndarray]=0.0,
        moisture_foliar: Union[float, np.ndarray]=100.0,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param fuel_model_grid: Input matching the reference API.
    :param wind_speed: Input matching the reference API.
    :param wind_direction: Input matching the reference API.
    :param slope: Input matching the reference API.
    :param slope_units: Input matching the reference API.
    :param aspect: Input matching the reference API.
    :param canopy_base_height: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :param canopy_bulk_density: Input matching the reference API.
    :param moisture_foliar: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.coerce_crown_fire_inputs', locals())

def get_surface_fire_inputs(surface_results: dict) -> dict:
    """Dispatch to the standalone native implementation.

    :param surface_results: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.get_surface_fire_inputs', locals())

def run_crown_fire(
        surface_results: dict,
        lut: dict,
        fuel_model_grid: Union[int, np.ndarray],
        m1h: Union[float, np.ndarray],
        m10h: Union[float, np.ndarray],
        m100h: Union[float, np.ndarray],
        mlh: Union[float, np.ndarray],
        mlw: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        slope: Union[float, np.ndarray],
        slope_units: int=0,
        aspect: Union[float, np.ndarray]=0.0,
        canopy_base_height: Union[float, np.ndarray]=0.0,
        canopy_height: Union[float, np.ndarray]=0.0,
        canopy_bulk_density: Union[float, np.ndarray]=0.0,
        moisture_foliar: Union[float, np.ndarray]=100.0,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param surface_results: Input matching the reference API.
    :param lut: Input matching the reference API.
    :param fuel_model_grid: Input matching the reference API.
    :param m1h: Input matching the reference API.
    :param m10h: Input matching the reference API.
    :param m100h: Input matching the reference API.
    :param mlh: Input matching the reference API.
    :param mlw: Input matching the reference API.
    :param wind_speed: Input matching the reference API.
    :param wind_speed_units: Input matching the reference API.
    :param wind_direction: Input matching the reference API.
    :param wind_orientation_mode: Input matching the reference API.
    :param slope: Input matching the reference API.
    :param slope_units: Input matching the reference API.
    :param aspect: Input matching the reference API.
    :param canopy_base_height: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :param canopy_bulk_density: Input matching the reference API.
    :param moisture_foliar: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('crown.run_crown_fire', locals())
