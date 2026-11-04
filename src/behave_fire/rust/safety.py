"""Explicit native safety adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def calculate_safety_zone(
        flame_height_ft: Union[float, np.ndarray],
        number_of_personnel: Union[int, float, np.ndarray],
        area_per_person_sqft: Union[float, np.ndarray],
        number_of_equipment: Union[int, float, np.ndarray],
        area_per_equipment_sqft: Union[float, np.ndarray],
) -> dict:
    """Dispatch to the standalone native implementation.

    :param flame_height_ft: Input matching the reference API.
    :param number_of_personnel: Input matching the reference API.
    :param area_per_person_sqft: Input matching the reference API.
    :param number_of_equipment: Input matching the reference API.
    :param area_per_equipment_sqft: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('safety.calculate_safety_zone', locals())
