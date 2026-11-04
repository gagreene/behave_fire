"""Explicit native vapor_pressure_deficit_calculator adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def calculate_vpd(
        temperature: Union[float, np.ndarray],
        temp_units: int,
        relative_humidity: Union[float, np.ndarray],
        rh_units: int,
        output_units: int=0,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param temperature: Input matching the reference API.
    :param temp_units: Input matching the reference API.
    :param relative_humidity: Input matching the reference API.
    :param rh_units: Input matching the reference API.
    :param output_units: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('vapor_pressure_deficit_calculator.calculate_vpd', locals())
