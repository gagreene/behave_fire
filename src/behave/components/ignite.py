"""
ignite.py — Vectorized Ignition Probability (§13.3)

Implements fuel temperature adjustment and probability-of-ignition calculations
for both firebrand and lightning ignition sources.

Public API
----------
calculate_fuel_temperature(air_temp_f, sun_shade_fraction)
    → (*S) ndarray — fuel temperature (°F)

calculate_firebrand_ignition_probability(air_temp_f, sun_shade_fraction,
                                          moisture_1h_fraction)
    → (*S) ndarray — probability [0, 1]

calculate_lightning_ignition_probability(fuel_bed_type_grid,
                                          moisture_100h_fraction,
                                          duff_depth_inches, charge_type)
    → (*S) ndarray — probability [0, 1]

Fuel bed types (0-based integer):
    0: PONDEROSA_PINE_LITTER
    1: PUNKY_WOOD_ROTTEN_CHUNKY
    2: PUNKY_WOOD_POWDER_DEEP
    3: PUNK_WOOD_POWDER_SHALLOW
    4: LODGEPOLE_PINE_DUFF
    5: DOUGLAS_FIR_DUFF
    6: HIGH_ALTITUDE_MIXED
    7: PEAT_MOSS

Charge types:
    0: negative
    1: positive
    2: unknown (weighted average of negative and positive)
"""

import numpy as np
from typing import Union

# Lightning coefficient table — shape (8, 4): [a_pos, b_pos, a_neg, b_neg]
# Each row corresponds to one of the 8 fuel bed types (index 0–7).
_LIG_COEFFS = np.array([
    [0.92, -0.087, 1.04, -0.054],   # 0: PONDEROSA_PINE_LITTER
    [0.44, -0.110, 0.59, -0.094],   # 1: PUNKY_WOOD_ROTTEN_CHUNKY
    [0.86, -0.060, 0.90, -0.056],   # 2: PUNKY_WOOD_POWDER_DEEP
    [0.60, -0.011, 0.73, -0.011],   # 3: PUNK_WOOD_POWDER_SHALLOW (linear formula)
    [5.13,  0.68,  3.84,  0.60 ],   # 4: LODGEPOLE_PINE_DUFF      (logistic-duff)
    [6.69,  1.39,  5.48,  1.28 ],   # 5: DOUGLAS_FIR_DUFF         (logistic-duff)
    [0.62, -0.050, 0.80, -0.014],   # 6: HIGH_ALTITUDE_MIXED      (mixed formula)
    [0.71, -0.070, 0.84, -0.060],   # 7: PEAT_MOSS
])

# Conditional ignition probability constants (Latham 1991)
_CC_NEG,   _CC_POS   = 0.2, 0.9     # conditional prob given neg / pos charge
_FREQ_NEG, _FREQ_POS = 0.723, 0.277  # historical frequency of neg / pos strikes


def calculate_fuel_temperature(
        air_temp_f: Union[float, np.ndarray],
        sun_shade_fraction: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Fuel temperature adjustment (Rothermel 1983).

    Fuel exposed to direct sun is warmer than air temperature; shaded fuel
    approaches air temperature.  The adjustment ranges from +25°F (full sun)
    to +5°F (full shade).

    :param air_temp_f: Air temperature (°F) (*S) or scalar.
    :param sun_shade_fraction: Shade fraction (0 = full sun, 1 = full shade)
        (*S) or scalar.
    :return: (*S) ndarray — fuel temperature (°F).
    """
    t = np.asarray(air_temp_f,         dtype=float)
    s = np.asarray(sun_shade_fraction, dtype=float)
    # Adjustment: +25°F at full sun, decreasing linearly to +5°F at full shade
    return t + (25.0 - 20.0 * s)


def calculate_firebrand_ignition_probability(
        air_temp_f: Union[float, np.ndarray],
        sun_shade_fraction: Union[float, np.ndarray],
        moisture_1h_fraction: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Probability of ignition from a glowing firebrand (Rothermel & Rinehart 1983).

    :param air_temp_f: Air temperature (°F) (*S) or scalar.
    :param sun_shade_fraction: Shade fraction (0 = full sun, 1 = full shade)
        (*S) or scalar.
    :param moisture_1h_fraction: 1-hr dead fuel moisture as fraction
        (e.g. 0.06 = 6%) (*S) or scalar.
    :return: (*S) ndarray — probability of ignition [0, 1].
    """
    # Convert air temperature to fuel temperature, then to Celsius
    ft_f = calculate_fuel_temperature(air_temp_f, sun_shade_fraction)
    ft_c = (ft_f - 32.0) * 5.0 / 9.0   # °F → °C
    m    = np.asarray(moisture_1h_fraction, dtype=float)

    # Heat of ignition index (lower = higher ignition probability)
    hoi = (
        144.51
        - 0.26600 * ft_c
        - 0.00058 * ft_c ** 2
        - ft_c * m
        + 18.54 * (1.0 - np.exp(-15.1 * m))
        + 640.0 * m
    )
    # Map HOI to probability (capped at x=40 for HOI ≤ 360)
    x = 0.1 * (400.0 - np.minimum(hoi, 400.0))
    return np.clip(
        (0.000048 * np.power(np.maximum(x, 0.0), 4.3)) / 50.0,
        0.0, 1.0
    )


def calculate_lightning_ignition_probability(
        fuel_bed_type_grid: Union[int, np.ndarray],
        moisture_100h_fraction: Union[float, np.ndarray],
        duff_depth_inches: Union[float, np.ndarray],
        charge_type: int
) -> np.ndarray:
    """
    Probability of ignition from a lightning strike (Latham 1991).

    Different equation forms are used for each fuel bed type:
    - Types 0, 1, 2, 7: exponential in moisture
    - Type 3: linear in moisture
    - Types 4, 5: logistic in duff depth
    - Type 6: mixed (exponential negative, linear positive)

    :param fuel_bed_type_grid: Fuel bed type per cell (*S) int, 0–7.
        See module docstring for type definitions.
    :param moisture_100h_fraction: 100-hr dead fuel moisture as fraction
        (e.g. 0.20 = 20%) (*S) or scalar. Internally capped at 40%.
    :param duff_depth_inches: Duff layer depth (inches) (*S) or scalar.
        Internally capped at ~3.94 inches (10 cm).
    :param charge_type: Lightning charge type — scalar int.
        0 = negative, 1 = positive, 2 = unknown (frequency-weighted average).
    :return: (*S) ndarray — probability of ignition [0, 1].
    """
    ft  = np.clip(np.asarray(fuel_bed_type_grid, dtype=np.int32), 0, 7)
    # Convert fraction → percent, cap at 40%
    m   = np.minimum(np.asarray(moisture_100h_fraction, dtype=float) * 100.0, 40.0)
    # Convert inches → cm, cap at 10 cm
    d   = np.minimum(np.asarray(duff_depth_inches, dtype=float) * 2.54, 10.0)

    ap = _LIG_COEFFS[ft, 0]
    bp = _LIG_COEFFS[ft, 1]
    an = _LIG_COEFFS[ft, 2]
    bn = _LIG_COEFFS[ft, 3]

    # Positive-charge probability: logistic (duff-depth) for types 4,5; exponential otherwise
    p_pos = np.where(
        (ft == 4) | (ft == 5),
        1.0 / (1.0 + np.exp(ap - np.abs(bp) * d)),
        ap * np.exp(bp * m)
    )

    # Negative-charge probability: logistic for 4,5; linear for 3,6; exponential for rest
    p_neg = np.where(
        (ft == 4) | (ft == 5),
        1.0 / (1.0 + np.exp(an - np.abs(bn) * d)),
        np.where(
            (ft == 3) | (ft == 6),
            an + bn * m,
            an * np.exp(bn * m)
        )
    )

    # Combine according to charge type
    if charge_type == 0:
        prob = _CC_NEG * p_neg
    elif charge_type == 1:
        prob = _CC_POS * p_pos
    else:
        # Unknown: frequency-weighted average of negative and positive
        prob = _FREQ_NEG * _CC_NEG * p_neg + _FREQ_POS * _CC_POS * p_pos

    return np.clip(prob, 0.0, 1.0)

