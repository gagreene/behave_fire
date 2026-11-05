"""Explicit native spot adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native

def calculate_spotting_from_burning_pile(
        flame_height_ft: Union[float, np.ndarray],
        wind_mph: Union[float, np.ndarray],
        cover_height_ft: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param flame_height_ft: Input matching the reference API.
    :param wind_mph: Input matching the reference API.
    :param cover_height_ft: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('spot.calculate_spotting_from_burning_pile', locals())

def calculate_spotting_from_surface_fire(
        flame_length_ft: Union[float, np.ndarray],
        wind_mph: Union[float, np.ndarray],
        cover_height_ft: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param flame_length_ft: Input matching the reference API.
    :param wind_mph: Input matching the reference API.
    :param cover_height_ft: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('spot.calculate_spotting_from_surface_fire', locals())

def calculate_spotting_from_torching_trees(
        dbh_in: Union[float, np.ndarray],
        height_ft: Union[float, np.ndarray],
        count: Union[int, float, np.ndarray],
        wind_mph: Union[float, np.ndarray],
        cover_height_ft: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param dbh_in: Input matching the reference API.
    :param height_ft: Input matching the reference API.
    :param count: Input matching the reference API.
    :param wind_mph: Input matching the reference API.
    :param cover_height_ft: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('spot.calculate_spotting_from_torching_trees', locals())
