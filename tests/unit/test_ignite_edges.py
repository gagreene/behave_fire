"""Fuel temperature and firebrand probability boundaries for the Python Ignite component."""

import numpy as np

from behave_fire.components.ignite import (
    calculate_firebrand_ignition_probability,
    calculate_fuel_temperature,
)

# ---- firebrand ignition tests ----

def test_firebrand_probability_decreases_with_moisture_between_saturation_limits():
    moisture = np.array([0.0, 0.125, 0.25, 0.5])

    # At 71°F and half shade, the fuel is 86°F (30°C).
    dry, lower_moisture, higher_moisture, wet = calculate_firebrand_ignition_probability(71.0, 0.5, moisture)

    assert dry == 1.0
    assert 1.0 > lower_moisture > higher_moisture > 0.0
    assert wet == 0.0


# ---- fuel temperature tests ----

def test_fuel_temperature_broadcasts_air_temperature_and_shade():
    air_temp_f = np.array([[0.0], [80.0]])
    shade_fraction = np.array([[0.0, 0.25, 1.0]])

    actual = calculate_fuel_temperature(air_temp_f, shade_fraction)

    # The warming is 25°F in full sun, 20°F at quarter shade, and 5°F in full shade.
    np.testing.assert_array_equal(actual, [[25.0, 20.0, 5.0], [105.0, 100.0, 85.0]])
