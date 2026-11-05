"""Explicit native fine_dead_fuel_moisture_tool adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def calculate_fine_dead_fuel_moisture(
        dry_bulb_i: Union[int, np.ndarray],
        rh_i: Union[int, np.ndarray],
        slope_i: Union[int, np.ndarray],
        aspect_i: Union[int, np.ndarray],
        shading_i: Union[int, np.ndarray],
        month_i: Union[int, np.ndarray],
        elev_i: Union[int, np.ndarray],
        time_i: Union[int, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param dry_bulb_i: Input matching the reference API.
    :param rh_i: Input matching the reference API.
    :param slope_i: Input matching the reference API.
    :param aspect_i: Input matching the reference API.
    :param shading_i: Input matching the reference API.
    :param month_i: Input matching the reference API.
    :param elev_i: Input matching the reference API.
    :param time_i: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture', locals())
