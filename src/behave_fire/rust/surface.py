"""Explicit native surface adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def build_particle_arrays(
        lut: dict,
        fuel_model_grid: Union[int, np.ndarray],
        m1h: Union[float, np.ndarray],
        m10h: Union[float, np.ndarray],
        m100h: Union[float, np.ndarray],
        mlh: Union[float, np.ndarray],
        mlw: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param lut: Input matching the reference API.
    :param fuel_model_grid: Input matching the reference API.
    :param m1h: Input matching the reference API.
    :param m10h: Input matching the reference API.
    :param m100h: Input matching the reference API.
    :param mlh: Input matching the reference API.
    :param mlw: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.build_particle_arrays', locals())

def calculate_backing_spread_rate(
        spread_rate: Union[float, np.ndarray],
        eccentricity: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param spread_rate: Input matching the reference API.
    :param eccentricity: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_backing_spread_rate', locals())

def calculate_direction_of_max_spread(
        x_component: Union[float, np.ndarray],
        y_component: Union[float, np.ndarray],
        rate_vector: Union[float, np.ndarray],
        wind_orientation_mode: str,
        aspect: Union[float, np.ndarray]=0.0,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param x_component: Input matching the reference API.
    :param y_component: Input matching the reference API.
    :param rate_vector: Input matching the reference API.
    :param wind_orientation_mode: Input matching the reference API.
    :param aspect: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_direction_of_max_spread', locals())

def calculate_eccentricity(fire_length_to_width_ratio: Union[float, np.ndarray]) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param fire_length_to_width_ratio: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_eccentricity', locals())

def calculate_effective_wind_speed(
        spread_rate: Union[float, np.ndarray],
        no_wind_no_slope_spread_rate: Union[float, np.ndarray],
        wind_b: Union[float, np.ndarray],
        wind_c: Union[float, np.ndarray],
        relative_packing_ratio: Union[float, np.ndarray],
        wind_e: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param spread_rate: Input matching the reference API.
    :param no_wind_no_slope_spread_rate: Input matching the reference API.
    :param wind_b: Input matching the reference API.
    :param wind_c: Input matching the reference API.
    :param relative_packing_ratio: Input matching the reference API.
    :param wind_e: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_effective_wind_speed', locals())

def calculate_fire_area(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        lwr: Union[float, np.ndarray],
        elapsed_min: float,
        is_crown: bool=False,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param forward_ros: Input matching the reference API.
    :param backing_ros: Input matching the reference API.
    :param lwr: Input matching the reference API.
    :param elapsed_min: Input matching the reference API.
    :param is_crown: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fire_area', locals())

def calculate_fire_length(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        elapsed_min: float,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param forward_ros: Input matching the reference API.
    :param backing_ros: Input matching the reference API.
    :param elapsed_min: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fire_length', locals())

def calculate_fire_length_to_width_ratio(
        effective_wind_speed: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param effective_wind_speed: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fire_length_to_width_ratio', locals())

def calculate_fire_perimeter(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        lwr: Union[float, np.ndarray],
        elapsed_min: float,
        is_crown: bool=False,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param forward_ros: Input matching the reference API.
    :param backing_ros: Input matching the reference API.
    :param lwr: Input matching the reference API.
    :param elapsed_min: Input matching the reference API.
    :param is_crown: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fire_perimeter', locals())

def calculate_fire_width(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        lwr: Union[float, np.ndarray],
        elapsed_min: float,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param forward_ros: Input matching the reference API.
    :param backing_ros: Input matching the reference API.
    :param lwr: Input matching the reference API.
    :param elapsed_min: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fire_width', locals())

def calculate_fireline_intensity(
        spread_rate: Union[float, np.ndarray],
        heat_per_unit_area: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param spread_rate: Input matching the reference API.
    :param heat_per_unit_area: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fireline_intensity', locals())

def calculate_flame_length(fireline_intensity: Union[float, np.ndarray]) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param fireline_intensity: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_flame_length', locals())

def calculate_flanking_spread_rate(
        spread_rate: Union[float, np.ndarray],
        backing_spread_rate: Union[float, np.ndarray],
        fire_length_to_width_ratio: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param spread_rate: Input matching the reference API.
    :param backing_spread_rate: Input matching the reference API.
    :param fire_length_to_width_ratio: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_flanking_spread_rate', locals())

def calculate_forward_spread_rate(
        no_wind_no_slope_spread_rate: Union[float, np.ndarray],
        rate_vector: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param no_wind_no_slope_spread_rate: Input matching the reference API.
    :param rate_vector: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_forward_spread_rate', locals())

def calculate_fuelbed_intermediates(p: dict) -> dict:
    """Dispatch to the standalone native implementation.

    :param p: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_fuelbed_intermediates', locals())

def calculate_heat_per_unit_area(
        reaction_intensity: Union[float, np.ndarray],
        residence_time: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param reaction_intensity: Input matching the reference API.
    :param residence_time: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_heat_per_unit_area', locals())

def calculate_midflame_wind_speed(
        waf: Union[float, np.ndarray],
        twenty_foot_wind_speed: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param waf: Input matching the reference API.
    :param twenty_foot_wind_speed: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_midflame_wind_speed', locals())

def calculate_no_wind_no_slope_spread_rate(
        ri: Union[float, np.ndarray],
        propagating_flux: Union[float, np.ndarray],
        heat_sink: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param ri: Input matching the reference API.
    :param propagating_flux: Input matching the reference API.
    :param heat_sink: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_no_wind_no_slope_spread_rate', locals())

def calculate_reaction_intensity(ib: dict) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param ib: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_reaction_intensity', locals())

def calculate_reaction_intensity_output(reaction_intensity: Union[float, np.ndarray]) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param reaction_intensity: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_reaction_intensity_output', locals())

def calculate_residence_time(sigma: Union[float, np.ndarray]) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param sigma: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_residence_time', locals())

def calculate_slope_phi(
        slope: Union[float, np.ndarray],
        slope_units: int,
        packing_ratio: Union[float, np.ndarray],
        reaction_intensity: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param slope: Input matching the reference API.
    :param slope_units: Input matching the reference API.
    :param packing_ratio: Input matching the reference API.
    :param reaction_intensity: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_slope_phi', locals())

def calculate_surface_rate_vector(
        no_wind_no_slope_spread_rate: Union[float, np.ndarray],
        wind_phi: Union[float, np.ndarray],
        slope_phi: Union[float, np.ndarray],
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        aspect: Union[float, np.ndarray]=0.0,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param no_wind_no_slope_spread_rate: Input matching the reference API.
    :param wind_phi: Input matching the reference API.
    :param slope_phi: Input matching the reference API.
    :param wind_direction: Input matching the reference API.
    :param wind_orientation_mode: Input matching the reference API.
    :param aspect: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_surface_rate_vector', locals())

def calculate_surface_wind_adjustment_factor(
        canopy_cover: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
        crown_ratio: Union[float, np.ndarray],
        depth: Union[float, np.ndarray],
        waf_method: str='UseCrownRatio',
        user_waf: Union[float, np.ndarray, None]=None,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param canopy_cover: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :param crown_ratio: Input matching the reference API.
    :param depth: Input matching the reference API.
    :param waf_method: Input matching the reference API.
    :param user_waf: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_surface_wind_adjustment_factor', locals())

def calculate_surface_wind_coefficients(sigma: Union[float, np.ndarray]) -> dict:
    """Dispatch to the standalone native implementation.

    :param sigma: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_surface_wind_coefficients', locals())

def calculate_twenty_foot_wind_speed(
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_height_mode: str='TwentyFoot',
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param wind_speed: Input matching the reference API.
    :param wind_speed_units: Input matching the reference API.
    :param wind_height_mode: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_twenty_foot_wind_speed', locals())

def calculate_wind_adjustment_factor(
        canopy_cover: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
        crown_ratio: Union[float, np.ndarray],
        depth: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param canopy_cover: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :param crown_ratio: Input matching the reference API.
    :param depth: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_wind_adjustment_factor', locals())

def calculate_wind_phi(
        sigma: Union[float, np.ndarray],
        midflame_wind_speed: Union[float, np.ndarray],
        wind_b: Union[float, np.ndarray],
        wind_c: Union[float, np.ndarray],
        relative_packing_ratio: Union[float, np.ndarray],
        wind_e: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param sigma: Input matching the reference API.
    :param midflame_wind_speed: Input matching the reference API.
    :param wind_b: Input matching the reference API.
    :param wind_c: Input matching the reference API.
    :param relative_packing_ratio: Input matching the reference API.
    :param wind_e: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.calculate_wind_phi', locals())

def run_surface_fire(
        ri: Union[float, np.ndarray],
        ib: dict,
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        slope: Union[float, np.ndarray],
        slope_units: int=0,
        aspect: Union[float, np.ndarray]=0.0,
        canopy_cover: Union[float, np.ndarray]=0.0,
        canopy_height: Union[float, np.ndarray]=0.0,
        crown_ratio: Union[float, np.ndarray]=0.0,
        wind_height_mode: str='TwentyFoot',
        waf_method: str='UseCrownRatio',
        user_waf: Union[float, np.ndarray, None]=None,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param ri: Input matching the reference API.
    :param ib: Input matching the reference API.
    :param wind_speed: Input matching the reference API.
    :param wind_speed_units: Input matching the reference API.
    :param wind_direction: Input matching the reference API.
    :param wind_orientation_mode: Input matching the reference API.
    :param slope: Input matching the reference API.
    :param slope_units: Input matching the reference API.
    :param aspect: Input matching the reference API.
    :param canopy_cover: Input matching the reference API.
    :param canopy_height: Input matching the reference API.
    :param crown_ratio: Input matching the reference API.
    :param wind_height_mode: Input matching the reference API.
    :param waf_method: Input matching the reference API.
    :param user_waf: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('surface.run_surface_fire', locals())
