"""Scorch height boundaries for the Python Mortality component."""

import numpy as np

from behave_fire.components.mortality import calculate_scorch_height

# ---- scorch height tests ----

def test_scorch_height_for_zero_intensity_and_calm_wind():
    fireline_intensity_btu_ft_s = np.array([[0.0], [1.0]])
    midflame_wind_mph = np.array([0.0, 2.0, 0.0])
    air_temp_f = np.array([77.0, 77.0, 14.0])

    actual = calculate_scorch_height(fireline_intensity_btu_ft_s, midflame_wind_mph, air_temp_f)

    # At unit intensity, calm air at 77°F gives one foot. A 2 mph wind
    # cuts that to one-third, while cooling calm air to 14°F halves it.
    np.testing.assert_array_equal(actual, [[0.0, 0.0, 0.0], [1.0, 1.0 / 3.0, 0.5]])
