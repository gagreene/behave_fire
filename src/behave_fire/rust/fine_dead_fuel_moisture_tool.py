"""Explicit native fine_dead_fuel_moisture_tool adapters."""
from __future__ import annotations

from typing import Union

import numpy as np

from ._bridge import _call


def calculate_fine_dead_fuel_moisture(
        dry_bulb_i: int | np.ndarray,
        rh_i: int | np.ndarray,
        slope_i: int | np.ndarray,
        aspect_i: int | np.ndarray,
        shading_i: int | np.ndarray,
        month_i: int | np.ndarray,
        elev_i: int | np.ndarray,
        time_i: int | np.ndarray,
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
