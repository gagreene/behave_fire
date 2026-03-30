"""
safety.py — Vectorized Safety Zone Calculations (§13.4)

Public API
----------
calculate_safety_zone(flame_height_ft, number_of_personnel, area_per_person_sqft,
                       number_of_equipment, area_per_equipment_sqft)
    → dict with keys 'separation_distance', 'radius', 'area'

Reference: Butler & Cohen 1998 — separation distance = 4 × flame height.
The safety zone radius adds a core area for personnel and equipment.
"""

import numpy as np
from typing import Union


def calculate_safety_zone(
        flame_height_ft: Union[float, np.ndarray],
        number_of_personnel: Union[int, float, np.ndarray],
        area_per_person_sqft: Union[float, np.ndarray],
        number_of_equipment: Union[int, float, np.ndarray],
        area_per_equipment_sqft: Union[float, np.ndarray]
) -> dict:
    """
    Calculate safety zone dimensions for firefighter deployment.

    The minimum separation distance is 4× the flame height (Butler & Cohen 1998).
    The total required radius extends beyond that to accommodate a circular core
    area sized for all personnel and equipment.

    :param flame_height_ft: Active flame height (ft) (*S) or scalar.
    :param number_of_personnel: Number of people requiring protection
        (*S) or scalar.
    :param area_per_person_sqft: Required area per person (ft²) (*S) or scalar.
        Typically 30–50 ft² per person.
    :param number_of_equipment: Number of equipment items (e.g. vehicles)
        (*S) or scalar.
    :param area_per_equipment_sqft: Required area per equipment item (ft²)
        (*S) or scalar.
    :return: dict with keys:
        ``'separation_distance'`` (*S) ndarray — minimum safe separation (ft),
        equal to 4 × ``flame_height_ft``;
        ``'radius'`` (*S) ndarray — total safety zone radius (ft),
        including the core area;
        ``'area'`` (*S) ndarray — total safety zone area (ft²),
        equal to π × ``radius``².
    """
    fh  = np.asarray(flame_height_ft,         dtype=float)
    np_ = np.asarray(number_of_personnel,     dtype=float)
    app = np.asarray(area_per_person_sqft,    dtype=float)
    ne  = np.asarray(number_of_equipment,     dtype=float)
    ape = np.asarray(area_per_equipment_sqft, dtype=float)

    # Minimum separation = 4× flame height (Butler & Cohen 1998)
    sep = 4.0 * fh

    # Core area radius for personnel and equipment
    core   = app * np_ + ape * ne
    core_r = np.where(core > 1e-7, np.sqrt(core / np.pi), 0.0)

    # Total radius = separation distance + core radius
    r = sep + core_r

    return {
        'separation_distance': sep,
        'radius':              r,
        'area':                np.pi * r ** 2,
    }
