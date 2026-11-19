"""Boundary coverage for the Python vapor pressure deficit calculator."""

import numpy as np

from behave_fire.components.behave_units import FractionUnits, PressureUnits, TemperatureUnits
from behave_fire.components.vapor_pressure_deficit_calculator import calculate_vpd

# ---- humidity broadcast tests ----

def test_vpd_humidity_endpoints_broadcast_in_kilopascals():
    result = calculate_vpd(
        temperature=np.array([[273.15], [293.15]]),
        temp_units=TemperatureUnits.TemperatureUnitsEnum.Kelvin,
        relative_humidity=np.array([0.0, 50.0, 100.0]),
        rh_units=FractionUnits.FractionUnitsEnum.Percent,
        output_units=PressureUnits.PressureUnitsEnum.KiloPascal,
    )

    saturated = result["saturated_vp"]
    actual = result["actual_vp"]
    deficit = result["vpd"]
    assert saturated.shape == (2, 1)
    assert actual.shape == deficit.shape == (2, 3)

    # At freezing, the Tetens exponent is zero and saturation is 6.11 hPa.
    assert saturated[0, 0] == 0.611
    np.testing.assert_array_equal(actual[0], [0.0, 0.3055, 0.611])
    np.testing.assert_array_equal(deficit[0], [0.611, 0.3055, 0.0])
    assert saturated[1, 0] > saturated[0, 0]

    np.testing.assert_array_equal(actual[:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(deficit[:, 0], saturated[:, 0])
    np.testing.assert_array_equal(actual[:, 2], saturated[:, 0])
    np.testing.assert_array_equal(deficit[:, 2], [0.0, 0.0])
    np.testing.assert_array_equal(actual[:, 1], saturated[:, 0] * 0.5)
    np.testing.assert_array_equal(deficit[:, 1], actual[:, 1])
