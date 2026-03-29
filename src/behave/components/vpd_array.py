"""
vpd_array.py — Vectorized Vapor Pressure Deficit (§13.6)

Public API
----------
calculate_vpd(temperature, temp_units, relative_humidity, rh_units,
               output_units=0)
    → dict with 'vpd', 'actual_vp', 'saturated_vp'

Uses the Tetens formula for saturation vapor pressure.
Requires behave_units_array for unit conversion.

Unit enums (scalar integers, same values as behave_units.py):
    temp_units   : 0=Fahrenheit, 1=Celsius, 2=Kelvin
    rh_units     : 0=Fraction,   1=Percent
    output_units : 0=Pascal, 1=HectoPascal, 2=KiloPascal, ...
                   (PressureUnitsEnum from behave_units.py)
"""

import numpy as np


def calculate_vpd(temperature, temp_units,
                   relative_humidity, rh_units,
                   output_units=0):
    """
    Vectorized vapor pressure deficit.

    Parameters
    ----------
    temperature       : (*S) or scalar
    temp_units        : scalar int — TemperatureUnitsEnum
    relative_humidity : (*S) or scalar
    rh_units          : scalar int — FractionUnitsEnum (0=fraction, 1=percent)
    output_units      : scalar int — PressureUnitsEnum (default 0 = Pascal)

    Returns
    -------
    dict with keys:
        'vpd'           : (*S) ndarray in output_units
        'actual_vp'     : (*S) ndarray in output_units
        'saturated_vp'  : (*S) ndarray in output_units
    """
    try:
        from .behave_units_array import temp_to_base, fraction_to_base, pressure_from_base
    except ImportError:
        from behave_units_array import temp_to_base, fraction_to_base, pressure_from_base

    # Convert to base units
    temp_f = temp_to_base(temperature, temp_units)       # (*S) °F
    rh     = fraction_to_base(relative_humidity, rh_units)  # (*S) fraction

    # °F → °C for Tetens formula
    temp_c = (np.asarray(temp_f) - 32.0) * 5.0 / 9.0

    # Tetens formula: es(T) = 6.11 * 10^(7.5 * T / (T + 237.3))  [hPa]
    temp_k = temp_c + 237.3
    safe_tk = np.where(np.abs(temp_k) > 1e-7, temp_k, 1.0)
    sat_hpa = 6.11 * np.power(10.0, (7.5 * temp_c) / safe_tk)
    act_hpa = np.asarray(rh) * sat_hpa
    vpd_hpa = sat_hpa - act_hpa

    # hPa → Pa (base), then convert to output_units
    sat_pa = sat_hpa * 100.0
    act_pa = act_hpa * 100.0
    vpd_pa = vpd_hpa * 100.0

    return {
        'vpd':          pressure_from_base(vpd_pa,  output_units),
        'actual_vp':    pressure_from_base(act_pa,  output_units),
        'saturated_vp': pressure_from_base(sat_pa,  output_units),
    }

