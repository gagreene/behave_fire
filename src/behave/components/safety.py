"""
safety_array.py — Vectorized Safety Zone Calculations (§13.4)

Public API
----------
calculate_safety_zone(flame_height_ft, number_of_personnel, area_per_person_sqft,
                       number_of_equipment, area_per_equipment_sqft)
    → dict with keys 'separation_distance', 'radius', 'area'

Reference: Butler & Cohen 1998 — separation distance = 4 × flame height.
The safety zone radius adds a core area for personnel and equipment.
"""

import numpy as np


def calculate_safety_zone(flame_height_ft,
                          number_of_personnel,
                          area_per_person_sqft,
                          number_of_equipment,
                          area_per_equipment_sqft):
    """
    Calculate safety zone dimensions.

    The separation distance is 4× flame height (Butler & Cohen 1998).
    The total required radius adds a core area for personnel and equipment.

    Parameters
    ----------
    flame_height_ft         : (*S) or scalar — flame height in feet
    number_of_personnel     : (*S) or scalar — number of people
    area_per_person_sqft    : (*S) or scalar — ft² per person
    number_of_equipment     : (*S) or scalar — number of equipment items
    area_per_equipment_sqft : (*S) or scalar — ft² per equipment item

    Returns
    -------
    dict with keys:
        'separation_distance' : (*S) ndarray — minimum separation distance (feet)
        'radius'              : (*S) ndarray — total safety zone radius (feet)
        'area'                : (*S) ndarray — total safety zone area (sq ft)
    """
    fh = np.asarray(flame_height_ft, dtype=float)
    np_ = np.asarray(number_of_personnel, dtype=float)
    app = np.asarray(area_per_person_sqft, dtype=float)
    ne = np.asarray(number_of_equipment, dtype=float)
    ape = np.asarray(area_per_equipment_sqft, dtype=float)

    sep = 4.0 * fh
    core = app * np_ + ape * ne
    core_r = np.where(core > 1e-7, np.sqrt(core / np.pi), 0.0)
    r = sep + core_r

    return {
        'separation_distance': sep,
        'radius': r,
        'area': np.pi * r ** 2,
    }
