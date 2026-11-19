"""Flame and occupancy boundaries for the Python Safety component."""

import numpy as np

from behave_fire.components.safety import calculate_safety_zone

# ---- safety zone tests ----

def test_safety_zone_for_zero_flame_and_occupancy_boundaries():
    flame_height_ft = np.array([[0.0], [1.0]])
    personnel = np.array([0, 1, 0, 1])
    equipment = np.array([0, 0, 1, 1])

    zone = calculate_safety_zone(flame_height_ft, personnel, 9 * np.pi, equipment, 16 * np.pi)

    np.testing.assert_array_equal(zone["separation_distance"], [[0.0], [4.0]])
    # Personnel and equipment need 3 ft and 4 ft core radii, giving 5 ft together.
    np.testing.assert_array_equal(zone["radius"], [[0.0, 3.0, 4.0, 5.0], [4.0, 7.0, 8.0, 9.0]])
    squared_radii = np.array([[0, 9, 16, 25], [16, 49, 64, 81]])
    area_sqft = zone["area"]
    assert area_sqft.shape == squared_radii.shape
    assert area_sqft[0, 0] == 0.0
    # Each positive circle lies between its inscribed and circumscribed squares.
    positive = squared_radii > 0
    np.testing.assert_array_less(2 * squared_radii[positive], area_sqft[positive])
    np.testing.assert_array_less(area_sqft[positive], 4 * squared_radii[positive])
