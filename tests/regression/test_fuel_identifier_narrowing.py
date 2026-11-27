"""A fuel identifier outside the lookup table must never select a real model through integer narrowing."""

import numpy as np
import pytest

from behave_fire import BehaveRun, FuelModels
from behave_fire.components.fuel_models_array import _fuel_model_indices


def _cell(grid):
    return dict(fuel_model_grid=grid, m1h=np.array([[.06]]), m10h=np.array([[.07]]), m100h=np.array([[.08]]),
                mlh=np.array([[.6]]), mlw=np.array([[.9]]), wind_speed=np.array([[10.]]), wind_speed_units=6,
                wind_direction=np.array([[225.]]), wind_orientation_mode="RelativeToNorth", slope=np.array([[10.]]),
                slope_units=1, aspect=np.array([[180.]]), canopy_cover=np.array([[.4]]),
                canopy_height=np.array([[15.]]), crown_ratio=np.array([[.55]]), wind_height_mode="TenMeter")


def _registry_with_model_256():
    registry = FuelModels()
    assert registry.set_custom_fuel_model(256, "ZZ1", "custom", 3., 0, .2, 0, 8000, 8000, 0, .1, .05, .05, .02, .02, 0,
                                          2000, 1500, 1500, 0, False)
    return registry


# ---- fuel identifier narrowing tests ----

def test_object_array_identifier_above_the_table_is_not_narrowed_onto_model_256():
    run = BehaveRun(_registry_with_model_256(), backend="python")
    real = run.do_surface_run(**_cell(np.array([[256]])))["spread_rate"]
    assert float(real[0, 0]) > 0.
    as_float = run.do_surface_run(**_cell(np.array([[256.5]])))["spread_rate"]
    as_object = run.do_surface_run(**_cell(np.array([[256.5]], dtype=object)))["spread_rate"]
    np.testing.assert_array_equal(as_object, as_float)
    assert not np.array_equal(as_object, real)


@pytest.mark.rust
def test_rust_rejects_a_fractional_identifier_instead_of_selecting_a_model():
    run = BehaveRun(_registry_with_model_256(), backend="rust")
    with pytest.raises(ValueError, match="finite integral"):
        run.do_surface_run(**_cell(np.array([[256.5]])))
    with pytest.raises(TypeError, match="real numeric"):
        run.do_surface_run(**_cell(np.array([[256.5]], dtype=object)))

def test_object_array_identifiers_are_bounds_checked_before_narrowing():
    for value in (256.5, 257, -1.2, 1e30, float("inf"), float("nan"), .5):
        assert _fuel_model_indices(np.array([value], dtype=object), 257)[0] == 0, value
    assert _fuel_model_indices(np.array([256], dtype=object), 257)[0] == 256
    assert _fuel_model_indices(np.array([124.], dtype=object), 257)[0] == 124
