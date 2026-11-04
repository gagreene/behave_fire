"""Explicit native ignite adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def calculate_firebrand_ignition_probability(
        air_temp_f: Union[float, np.ndarray],
        sun_shade_fraction: Union[float, np.ndarray],
        moisture_1h_fraction: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param air_temp_f: Input matching the reference API.
    :param sun_shade_fraction: Input matching the reference API.
    :param moisture_1h_fraction: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('ignite.calculate_firebrand_ignition_probability', locals())

def calculate_fuel_temperature(
        air_temp_f: Union[float, np.ndarray],
        sun_shade_fraction: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param air_temp_f: Input matching the reference API.
    :param sun_shade_fraction: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('ignite.calculate_fuel_temperature', locals())

def calculate_lightning_ignition_probability(
        fuel_bed_type_grid: Union[int, np.ndarray],
        moisture_100h_fraction: Union[float, np.ndarray],
        duff_depth_inches: Union[float, np.ndarray],
        charge_type: int,
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param fuel_bed_type_grid: Input matching the reference API.
    :param moisture_100h_fraction: Input matching the reference API.
    :param duff_depth_inches: Input matching the reference API.
    :param charge_type: Input matching the reference API.
    :return: Owned native calculation result.
    """
    warnings.warn('Lightning ignition probability is deprecated: upstream Behave removed this model because its scientific basis is no longer considered reliable. Retained for compatibility only.', DeprecationWarning, stacklevel=2)
    return _call('ignite.calculate_lightning_ignition_probability', locals())
