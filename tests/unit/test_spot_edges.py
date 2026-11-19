"""Flat-terrain boundaries for the Python Spot component."""

import numpy as np

from behave_fire.components.spot import (
    calculate_spotting_from_surface_fire,
    calculate_spotting_from_torching_trees,
)

# ---- surface fire tests ----

def test_surface_fire_spotting_at_calm_wind_and_zero_cover():
    wind_mph = np.array([[0.0], [5.0]])
    cover_height_ft = np.array([0.0, 10.0])

    distances = calculate_spotting_from_surface_fire(5.0, wind_mph, cover_height_ft)

    assert distances.shape == (2, 2)
    np.testing.assert_array_equal(distances[0], [0.0, 0.0])
    # With wind, the critical cover floor allows spotting even at zero input cover.
    assert np.all(np.isfinite(distances[1]) & (distances[1] > 0.0)), distances


# ---- torching trees tests ----

def test_torching_tree_spotting_at_zero_count_and_calm_wind():
    count = np.array([[0], [1]])
    wind_mph = np.array([0.0, 5.0])

    distances = calculate_spotting_from_torching_trees(12.0, 60.0, count, wind_mph, 10.0)

    assert distances.shape == (2, 2)
    np.testing.assert_array_equal(distances[0], [0.0, 0.0])
    assert distances[1, 0] == 0.0
    assert np.isfinite(distances[1, 1]) and distances[1, 1] > 0.0
