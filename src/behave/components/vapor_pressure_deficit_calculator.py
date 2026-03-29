"""
vapor_pressure_deficit_calculator.py — Vectorized Vapor Pressure Deficit (§13.6)

Public API
----------
calculate_vpd(temperature, temp_units, relative_humidity, rh_units,
               output_units=0)
    → dict with 'vpd', 'actual_vp', 'saturated_vp'

Uses the Tetens formula for saturation vapor pressure.
Requires behave_units for unit conversion.

Unit enums (scalar integers, same values as behave_units.py):
    temp_units   : 0=Fahrenheit, 1=Celsius, 2=Kelvin
    rh_units     : 0=Fraction,   1=Percent
    output_units : 0=Pascal, 1=HectoPascal, 2=KiloPascal, ...
                   (PressureUnitsEnum from behave_units.py)
"""

import numpy as np
from typing import Union


def calculate_vpd(
        temperature: Union[float, np.ndarray],
        temp_units: int,
        relative_humidity: Union[float, np.ndarray],
        rh_units: int,
        output_units: int = 0
) -> dict:
    """
    Vectorized vapor pressure deficit using the Tetens formula.

    Computes saturated vapor pressure (es), actual vapor pressure (ea), and
    the deficit (VPD = es − ea).

    :param temperature: Air temperature (*S) or scalar, in ``temp_units``.
    :param temp_units: Scalar integer ``TemperatureUnitsEnum`` value
        (0=Fahrenheit, 1=Celsius, 2=Kelvin).
    :param relative_humidity: Relative humidity (*S) or scalar, in ``rh_units``.
    :param rh_units: Scalar integer ``FractionUnitsEnum`` value
        (0=fraction [0–1], 1=percent [0–100]).
    :param output_units: Scalar integer ``PressureUnitsEnum`` value for the
        output pressure (default 0 = Pascal).
    :return: dict with keys:
        ``'vpd'`` (*S) ndarray — vapor pressure deficit in ``output_units``;
        ``'actual_vp'`` (*S) ndarray — actual vapor pressure in ``output_units``;
        ``'saturated_vp'`` (*S) ndarray — saturated vapor pressure in
        ``output_units``.
    """
    try:
        from .behave_units import temp_to_base, fraction_to_base, pressure_from_base
    except ImportError:
        from behave_units import temp_to_base, fraction_to_base, pressure_from_base

    # Convert inputs to base units: °F and fraction
    temp_f = temp_to_base(value=temperature, units=temp_units)
    rh     = fraction_to_base(value=relative_humidity, units=rh_units)

    # °F → °C required for Tetens formula
    temp_c = (np.asarray(temp_f) - 32.0) * 5.0 / 9.0

    # Tetens formula: es(T) = 6.11 × 10^(7.5 × T / (T + 237.3))  [hPa]
    temp_k = temp_c + 237.3
    safe_tk = np.where(np.abs(temp_k) > 1e-7, temp_k, 1.0)
    sat_hpa = 6.11 * np.power(10.0, (7.5 * temp_c) / safe_tk)   # hPa
    act_hpa = np.asarray(rh) * sat_hpa                            # hPa
    vpd_hpa = sat_hpa - act_hpa                                    # hPa

    # hPa → Pa (base pressure unit), then convert to requested output units
    sat_pa = sat_hpa * 100.0
    act_pa = act_hpa * 100.0
    vpd_pa = vpd_hpa * 100.0

    return {
        'vpd':          pressure_from_base(value=vpd_pa,  units=output_units),
        'actual_vp':    pressure_from_base(value=act_pa,  units=output_units),
        'saturated_vp': pressure_from_base(value=sat_pa,  units=output_units),
    }

