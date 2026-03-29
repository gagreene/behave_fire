"""
ignite_array.py — Vectorized Ignition Probability (§13.3)

Public API
----------
calculate_fuel_temperature(air_temp_f, sun_shade_fraction)
calculate_firebrand_ignition_probability(air_temp_f, sun_shade_fraction,
                                          moisture_1h_fraction)
calculate_lightning_ignition_probability(fuel_bed_type_grid,
                                          moisture_100h_fraction,
                                          duff_depth_inches, charge_type)

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
    2: unknown (weighted average)
"""

import numpy as np

# Lightning coefficient table — shape (8, 4): [a_pos, b_pos, a_neg, b_neg]
_LIG_COEFFS = np.array([
    [0.92, -0.087, 1.04, -0.054],   # 0: PONDEROSA_PINE_LITTER
    [0.44, -0.110, 0.59, -0.094],   # 1: PUNKY_WOOD_ROTTEN_CHUNKY
    [0.86, -0.060, 0.90, -0.056],   # 2: PUNKY_WOOD_POWDER_DEEP
    [0.60, -0.011, 0.73, -0.011],   # 3: PUNK_WOOD_POWDER_SHALLOW (linear)
    [5.13,  0.68,  3.84,  0.60 ],   # 4: LODGEPOLE_PINE_DUFF      (logistic-duff)
    [6.69,  1.39,  5.48,  1.28 ],   # 5: DOUGLAS_FIR_DUFF         (logistic-duff)
    [0.62, -0.050, 0.80, -0.014],   # 6: HIGH_ALTITUDE_MIXED      (mixed)
    [0.71, -0.070, 0.84, -0.060],   # 7: PEAT_MOSS
])

_CC_NEG,   _CC_POS   = 0.2, 0.9
_FREQ_NEG, _FREQ_POS = 0.723, 0.277


def calculate_fuel_temperature(air_temp_f, sun_shade_fraction):
    """
    Fuel temperature adjustment (Rothermel 1983).

    Parameters
    ----------
    air_temp_f         : (*S) or scalar — air temperature in °F
    sun_shade_fraction : (*S) or scalar — 0 (full sun) to 1 (full shade)

    Returns
    -------
    (*S) ndarray — fuel temperature in °F
    """
    t = np.asarray(air_temp_f,         dtype=float)
    s = np.asarray(sun_shade_fraction, dtype=float)
    return t + (25.0 - 20.0 * s)


def calculate_firebrand_ignition_probability(air_temp_f, sun_shade_fraction,
                                              moisture_1h_fraction):
    """
    Probability of ignition from a firebrand (Rothermel & Rinehart 1983).

    Parameters
    ----------
    air_temp_f         : (*S) or scalar — °F
    sun_shade_fraction : (*S) or scalar — 0–1
    moisture_1h_fraction: (*S) or scalar — 1-hr dead fuel moisture (fraction)

    Returns
    -------
    (*S) ndarray — probability [0, 1]
    """
    ft_f = calculate_fuel_temperature(air_temp_f, sun_shade_fraction)
    ft_c = (ft_f - 32.0) * 5.0 / 9.0   # → Celsius
    m    = np.asarray(moisture_1h_fraction, dtype=float)

    hoi = (
        144.51
        - 0.26600 * ft_c
        - 0.00058 * ft_c ** 2
        - ft_c * m
        + 18.54 * (1.0 - np.exp(-15.1 * m))
        + 640.0 * m
    )
    x = 0.1 * (400.0 - np.minimum(hoi, 400.0))
    return np.clip(
        (0.000048 * np.power(np.maximum(x, 0.0), 4.3)) / 50.0,
        0.0, 1.0
    )


def calculate_lightning_ignition_probability(fuel_bed_type_grid,
                                               moisture_100h_fraction,
                                               duff_depth_inches,
                                               charge_type):
    """
    Probability of ignition from a lightning strike (Latham 1991).

    Parameters
    ----------
    fuel_bed_type_grid    : (*S) int 0–7 — fuel bed type per cell
    moisture_100h_fraction: (*S) or scalar — 100-hr dead fuel moisture (fraction)
    duff_depth_inches     : (*S) or scalar — duff depth in inches
    charge_type           : scalar int — 0=negative, 1=positive, 2=unknown

    Returns
    -------
    (*S) ndarray — probability [0, 1]
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

    # p_pos: exp for 0,1,2,3,6,7; logistic-duff for 4,5
    p_pos = np.where(
        (ft == 4) | (ft == 5),
        1.0 / (1.0 + np.exp(ap - np.abs(bp) * d)),
        ap * np.exp(bp * m)
    )

    # p_neg: exp for 0,1,2,7; linear for 3,6; logistic-duff for 4,5
    p_neg = np.where(
        (ft == 4) | (ft == 5),
        1.0 / (1.0 + np.exp(an - np.abs(bn) * d)),
        np.where(
            (ft == 3) | (ft == 6),
            an + bn * m,
            an * np.exp(bn * m)
        )
    )

    if charge_type == 0:
        prob = _CC_NEG * p_neg
    elif charge_type == 1:
        prob = _CC_POS * p_pos
    else:
        prob = _FREQ_NEG * _CC_NEG * p_neg + _FREQ_POS * _CC_POS * p_pos

    return np.clip(prob, 0.0, 1.0)

