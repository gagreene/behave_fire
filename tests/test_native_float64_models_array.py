"""The direct model boundary owns primitive columns and calls the configured core."""
import numpy as np
import pytest

from behave_fire import _rust as native
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

pytestmark = pytest.mark.rust


def _inputs(cells):
    return {name: np.full(cells, value, dtype=np.float64) for name, value in dict(
        m1h=.06, m10h=.07, m100h=.08, mlh=.6, mlw=.9, wind_speed=704.,
        wind_direction=35., slope=12., aspect=0., canopy_cover=0., canopy_height=30.,
        crown_ratio=0., canopy_base_height=4., canopy_bulk_density=.01, moisture_foliar=100.,
        surface_ros=2., surface_fli=200., surface_hpua=1000.).items()}


@pytest.mark.parametrize('kind', ['surface', 'crown'])
def test_direct_model_boundary_stores_all_primitive_fields_and_typed_codes(kind):
    physical = _inputs(5)
    controls = dict(fuel_model_grid=np.full(5, 10, np.int32))
    registry = build_fuel_lookup_arrays(FuelModels())
    actual = native.model_batch(kind, physical, controls, registry, [5], {})
    assert len(actual) == (14 if kind == 'surface' else 22)
    for name, value in actual.items():
        assert value.shape == (5,)
        assert value.dtype == (np.int32 if name == 'fire_type' else np.float64)
        assert not np.shares_memory(value, physical['wind_speed'])


def test_direct_model_callback_runs_only_after_all_inputs_and_registry_are_owned():
    physical = _inputs(5)
    controls = dict(fuel_model_grid=np.full(5, 10, np.int32))
    registry = build_fuel_lookup_arrays(FuelModels())
    expected = native.model_batch('surface', physical, controls, registry, [5], {})

    def mutate():
        physical['wind_speed'][:] = 0.
        controls['fuel_model_grid'][:] = 0
        registry['depth'][:] = 0.

    actual = native.model_batch('surface', physical, controls, registry, [5], {}, _owned_callback=mutate)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_direct_model_invalid_controls_and_shapes_fail_before_callback():
    called = []
    physical = _inputs(5)
    registry = build_fuel_lookup_arrays(FuelModels())
    for controls in [dict(fuel_model_grid=np.array([10.], dtype=np.float64)),
                     dict(fuel_model_grid=np.array([10, 10], dtype=np.int32))]:
        with pytest.raises((TypeError, ValueError)):
            native.model_batch('surface', physical, controls, registry, [5], {},
                               _owned_callback=lambda: called.append(True))
    assert not called


def test_direct_model_missing_control_masks_fail_before_callback():
    called = []
    controls = dict(fuel_model_grid=np.ma.array(np.full(5, 10, np.int32), mask=True))
    with pytest.raises((TypeError, ValueError)):
        native.model_batch('surface', _inputs(5), controls, build_fuel_lookup_arrays(FuelModels()),
                           [5], {}, _owned_callback=lambda: called.append(True))
    assert not called


def test_empty_model_shape_extent_overflow_fails_before_callback():
    called = []
    with pytest.raises((ValueError, OverflowError)):
        native.model_batch('surface', _inputs(1), dict(fuel_model_grid=np.array([10], np.int32)),
                           build_fuel_lookup_arrays(FuelModels()), [np.iinfo(np.intp).max, 0], {},
                           _owned_callback=lambda: called.append(True))
    assert not called


def test_fixed_waf_and_upslope_config_do_not_require_unused_geometry_columns():
    physical = _inputs(5)
    physical['user_waf'] = np.array([.4])
    controls = dict(fuel_model_grid=np.full(5,10,np.int32))
    registry = build_fuel_lookup_arrays(FuelModels())
    expected = native.model_batch('surface',physical,controls,registry,[5],dict(waf_method=1))
    for key in ['canopy_cover','canopy_height','crown_ratio','aspect']:
        del physical[key]
    actual = native.model_batch('surface',physical,controls,registry,[5],dict(waf_method=1))
    for key in expected:
        np.testing.assert_array_equal(actual[key],expected[key])


def test_model_unit_descriptors_are_typed_owned_and_apply_before_science():
    physical = _inputs(5)
    physical['wind_speed'][:] = 8.
    controls = dict(fuel_model_grid=np.full(5, 10, np.int32))
    registry = build_fuel_lookup_arrays(FuelModels())
    expected_inputs = {name: value.copy() for name, value in physical.items()}
    expected_inputs['wind_speed'] *= 88.
    expected = native.model_batch('surface', expected_inputs, controls, registry, [5], {})
    units = np.array([5], np.int32)

    def mutate():
        units[:] = 0

    actual = native.model_batch('surface', physical, controls, registry, [5], {},
                               conversions={'wind_speed': ('Speed', units)}, _owned_callback=mutate)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_model_unit_selector_errors_prevent_callback():
    called = []
    for codes in [np.array([999], np.int32), np.array([5., 5.]), np.array([5, 5], np.int32)]:
        with pytest.raises((TypeError, ValueError)):
            native.model_batch('surface', _inputs(5), dict(fuel_model_grid=np.full(5, 10, np.int32)),
                               build_fuel_lookup_arrays(FuelModels()), [5], {},
                               conversions={'wind_speed': ('Speed', codes)},
                               _owned_callback=lambda: called.append(True))
    assert not called


def test_native_canopy_ratio_derives_from_actual_heights_without_dummy_input():
    physical = _inputs(5)
    physical['canopy_cover'][:] = .5
    physical['crown_ratio'][:] = 26./30.
    controls = dict(fuel_model_grid=np.full(5, 10, np.int32))
    registry = build_fuel_lookup_arrays(FuelModels())
    expected = native.model_batch('surface', physical, controls, registry, [5], {})
    del physical['crown_ratio']
    actual = native.model_batch('surface', physical, controls, registry, [5], {}, derive_ratio=True)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
