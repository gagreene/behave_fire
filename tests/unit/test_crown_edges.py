"""Crown thresholds and zero-canopy behavior in the Python components."""

import numpy as np

from behave_fire.components import crown

# ---- canopy heat tests ----

def test_zero_canopy_fuel_leaves_surface_heat_intact():
    result = crown.calculate_crown_heat_and_intensity(
        surface_hpua=2000.0,
        crown_ros=60.0,
        canopy_base_height=4.0,
        canopy_height=np.array([12.0, 4.0, 12.0]),
        canopy_bulk_density=np.array([0.0, 0.125, 0.125]),
    )

    np.testing.assert_array_equal(result['crown_fuel_load'], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result['canopy_heat_per_unit_area'], [0.0, 0.0, 8000.0])
    np.testing.assert_array_equal(result['crown_fire_heat_per_unit_area'], [2000.0, 2000.0, 10000.0])
    np.testing.assert_array_equal(result['crown_fire_line_intensity'], [2000.0, 2000.0, 10000.0])


# ---- classification tests ----

def test_fire_type_at_transition_and_active_thresholds():
    ratios = np.array([np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0)])
    result = crown.classify_crown_fire_type(ratios[:, None], ratios[None, :])

    np.testing.assert_array_equal(result['fire_type'], [[0, 2, 2], [1, 3, 3], [1, 3, 3]])
    np.testing.assert_array_equal(
        result['is_surface_fire'], [[True, True, True], [False, False, False], [False, False, False]]
    )
    np.testing.assert_array_equal(
        result['is_passive_crown'], [[False, False, False], [True, False, False], [True, False, False]]
    )
    np.testing.assert_array_equal(
        result['is_active_crown'], [[False, False, False], [False, True, True], [False, True, True]]
    )


# ---- critical spread and intensity tests ----

def test_critical_surface_intensity_floors_base_height_and_foliar_moisture():
    # The 0.1 m height floor lies between 0.25 ft and 0.5 ft.
    result = crown.calculate_critical_surface_fireline_intensity(
        canopy_base_height=np.array([0.0, 0.25, 0.5, 0.0, 0.0]),
        moisture_foliar=np.array([30.0, 30.0, 30.0, 0.0, 100.0]),
    )

    assert 0.0 < result[0] == result[1] == result[3]
    assert result[2] > result[0]
    assert result[4] > result[0]


def test_zero_critical_crown_spread_rate_disables_only_active_ratio():
    result = crown.calculate_crown_fire_ratios(
        surface_fli=60.0,
        crown_ros=32.0,
        critical_surface_fli=30.0,
        critical_crown_ros=np.array([0.0, 16.0, 32.0]),
    )

    assert result['crown_fire_transition_ratio'] == 2.0
    np.testing.assert_array_equal(result['crown_fire_active_ratio'], [0.0, 2.0, 1.0])


# ---- crown fraction burned tests ----

def test_crown_fraction_burned_at_spread_thresholds():
    # The last input reverses the transition and crowning thresholds.
    result = crown.calculate_crown_fraction_burned(
        surface_ros=np.array([2.0, 4.0, 6.0, 8.0, 10.0, 2.0]),
        surface_critical_ros=4.0,
        crowning_surface_ros=np.array([8.0, 8.0, 8.0, 8.0, 8.0, 2.0]),
    )

    np.testing.assert_array_equal(result, [0.0, 0.0, 0.5, 1.0, 1.0, 0.0])
