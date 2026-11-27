"""Exact and qualitative boundaries for isolated Python Surface calculations."""

import numpy as np
import pytest

from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays
from behave_fire.components.surface import (
    build_particle_arrays,
    calculate_direction_of_max_spread,
    calculate_reaction_intensity,
)

# ---- direction tests ----

def test_max_spread_direction_for_cardinal_vectors():
    """The positive x-axis points upslope, and bearings increase clockwise."""
    x = np.array([1.0, 0.0, -1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, -1.0])

    direction = calculate_direction_of_max_spread(x, y, 1.0, 'RelativeToUpslope')

    np.testing.assert_array_equal(direction, [0.0, 90.0, 180.0, 270.0])


def test_max_spread_direction_wraps_after_rotating_to_north():
    """A 90-degree upslope bearing rotates by the downslope aspect plus a half turn."""
    aspects = np.array([-90.0, 90.0, 360.0, 810.0])

    direction = calculate_direction_of_max_spread(0.0, 1.0, 1.0, 'RelativeToNorth', aspects)

    np.testing.assert_array_equal(direction, [180.0, 0.0, 270.0, 0.0])


# ---- dynamic curing tests ----

def test_dynamic_herb_load_partition_at_curing_boundaries():
    """GR1 transfers all herb below 30%, keeps transferring at 120%, and stops above 120%."""
    lookup = build_fuel_lookup_arrays(FuelModels())
    model = 101
    herb_load = lookup['live_herb'][model]
    live_herb_moisture = np.array([
        np.nextafter(0.30, 0.0), 0.30, np.nextafter(0.30, np.inf),
        np.nextafter(1.20, 0.0), 1.20, np.nextafter(1.20, np.inf),
    ])
    fuel_models = np.full(live_herb_moisture.shape, model)
    particles = build_particle_arrays(
        lookup, fuel_models, np.full(6, 0.06), np.full(6, 0.07),
        np.full(6, 0.08), live_herb_moisture, np.full(6, 0.90),
    )
    transferred = particles['load_dead'][3]
    remaining = particles['load_live'][0]

    assert transferred[0] == herb_load
    assert 0.0 < transferred[1] <= herb_load
    for index in (2, 3, 4):
        assert 0.0 < transferred[index] < herb_load, live_herb_moisture[index]
    assert transferred[5] == 0.0
    np.testing.assert_array_equal(remaining, herb_load - transferred)


# ---- reaction intensity tests ----

@pytest.mark.parametrize('life_state', ['dead', 'live'])
def test_reaction_intensity_extinguishes_each_life_state_at_its_threshold(life_state):
    """Each fuel class has positive reaction below extinction and zero at and above it."""
    fuelbed = {
        'sigma': 1500.0,
        'relative_packing_ratio': 1.0,
        'moe_dead': 0.25,
        'moe_live': 0.50,
        'w_moist_dead': 0.75,
        'w_moist_live': 0.75,
        'w_silica_dead': 0.01,
        'w_silica_live': 0.01,
        'w_heat_dead': 8000.0,
        'w_heat_live': 8000.0,
        'wfl_dead': 0.0,
        'wfl_live': 0.0,
    }
    threshold = fuelbed[f'moe_{life_state}']
    fuelbed[f'wfl_{life_state}'] = 1.0
    fuelbed[f'w_moist_{life_state}'] = np.array([
        np.nextafter(threshold, 0.0), threshold, np.nextafter(threshold, np.inf),
    ])

    intensity = calculate_reaction_intensity(fuelbed)

    assert intensity[0] > 0.0
    np.testing.assert_array_equal(intensity[1:], [0.0, 0.0])
