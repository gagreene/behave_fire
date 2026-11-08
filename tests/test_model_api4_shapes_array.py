"""Each model output keeps its own real dependency grid."""
import numpy as np
import pytest

from behave_fire import _rust as native
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

pytestmark = pytest.mark.rust


def _arguments():
    values = dict(m1h=.06, m10h=.07, m100h=.08, mlh=.6, mlw=.9,
                  wind_speed=88., wind_direction=35., slope=12., aspect=0.,
                  canopy_cover=0., canopy_height=30., crown_ratio=0.,
                  canopy_base_height=4., canopy_bulk_density=.01, moisture_foliar=100.,
                  surface_ros=2., surface_fli=200., surface_hpua=1000.)
    return dict(values, fuel_model_grid=np.array([10], np.int32),
                lut=build_fuel_lookup_arrays(FuelModels()))


def test_crown_empty_weather_keeps_independent_heat_and_thresholds(monkeypatch):
    from behave_fire.rust._model_api4 import execute_model

    arguments = _arguments()
    arguments['wind_speed'] = np.empty((0, 1))
    arguments['canopy_height'] = np.array([[30., 40., 50.]])
    calls = []
    original = native.model_groups

    def invoke(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, 'model_groups', invoke)
    actual = execute_model('crown', arguments)
    assert calls == [True]
    assert actual['crown_fire_spread_rate'].shape == (0, 1)
    assert actual['canopy_heat_per_unit_area'].shape == (1, 3)
    np.testing.assert_array_equal(actual['canopy_heat_per_unit_area'], [[2080., 2880., 3680.]])
    assert actual['crown_critical_fire_spread_rate'].shape == (1,)


def test_nonempty_projection_selects_axes_instead_of_a_flat_prefix():
    from behave_fire.rust._model_api4 import execute_model

    arguments = _arguments()
    arguments['canopy_height'] = np.array([[30.], [40.], [50.]])
    arguments['surface_hpua'] = np.array([[1000., 2000.]])
    actual = execute_model('crown', arguments)
    assert actual['canopy_heat_per_unit_area'].shape == (3, 1)
    np.testing.assert_array_equal(actual['canopy_heat_per_unit_area'], [[2080.], [2880.], [3680.]])
    assert actual['crown_fire_heat_per_unit_area'].shape == (3, 2)


def test_surface_empty_weather_keeps_final_mask_broadcast_shapes():
    from behave_fire.rust._model_api4 import execute_model

    arguments = _arguments()
    arguments['wind_speed'] = np.empty((0,))
    actual = execute_model('surface', arguments)
    assert all(value.shape == (0,) for value in actual.values())
