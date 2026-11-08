"""Restricted model groups compute only declared real dependencies."""
import numpy as np
import pytest

native = pytest.importorskip('behave_fire._rust', exc_type=ImportError)
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

pytestmark = pytest.mark.rust


def test_canopy_heat_group_never_reads_unbound_spread_or_surface_inputs():
    actual = native.model_fields('crown', ['canopy_heat_per_unit_area'],
        dict(canopy_base_height=np.array([4., 5.]), canopy_height=np.array([30., 40.]),
             canopy_bulk_density=np.array([.01, .02])), {},
        build_fuel_lookup_arrays(FuelModels()), [2], {})
    np.testing.assert_allclose(actual['canopy_heat_per_unit_area'], [2080., 5600.])


def test_classification_does_not_read_unbound_heat_or_surface_spread():
    physical = {name: np.full(2, value) for name, value in dict(m1h=.06,m10h=.07,m100h=.08,
        mlh=.6,mlw=.9,wind_speed=88.,wind_direction=35.,slope=12.,aspect=0.,
        canopy_base_height=4.,canopy_bulk_density=.01,moisture_foliar=100.,surface_fli=200.).items()}
    actual = native.model_fields('crown', ['fire_type'], physical, {},
        build_fuel_lookup_arrays(FuelModels()), [2], {})
    assert actual['fire_type'].dtype == np.int32
    assert np.all(np.isin(actual['fire_type'], [0, 1, 2, 3]))


def test_critical_density_group_propagates_missing_without_dummy_fuel_inputs():
    actual = native.model_fields('crown', ['crown_critical_fire_spread_rate'],
        dict(canopy_bulk_density=np.array([0., np.nan, .01])), {},
        build_fuel_lookup_arrays(FuelModels()), [3], {})
    values = actual['crown_critical_fire_spread_rate']
    assert values[0] == 0. and np.isnan(values[1])
    assert values[2] > 0.


def test_every_scoped_crown_output_matches_complete_f64_model_in_all_regimes():
    physical = {name: np.full(4, value) for name, value in dict(m1h=.06,m10h=.07,m100h=.08,
        mlh=.6,mlw=.9,wind_speed=0.,wind_direction=35.,slope=12.,aspect=0.,
        canopy_base_height=4.,canopy_height=30.,canopy_bulk_density=.01,moisture_foliar=100.,
        surface_ros=2.,surface_fli=0.,surface_hpua=1000.).items()}
    physical['wind_speed'][:] = [0., 0., 5000., 5000.]
    physical['surface_fli'][:] = [32.729299106026254,130.91719642410501,32.729299106026254,130.91719642410501]
    registry = build_fuel_lookup_arrays(FuelModels())
    expected = native.model_batch('crown', physical, {}, registry, [4], {})
    actual = native.model_fields('crown', list(expected), physical, {}, registry, [4], {})
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize('missing', ['surface_hpua','surface_fli','canopy_bulk_density'])
def test_scoped_crown_missing_data_matches_complete_model(missing):
    physical = {name: np.full(4, value) for name, value in dict(m1h=.06,m10h=.07,m100h=.08,
        mlh=.6,mlw=.9,wind_speed=0.,wind_direction=35.,slope=12.,aspect=0.,
        canopy_base_height=4.,canopy_height=30.,canopy_bulk_density=.01,moisture_foliar=100.,
        surface_ros=2.,surface_fli=0.,surface_hpua=1000.).items()}
    physical['wind_speed'][:] = [0.,0.,5000.,5000.]
    physical['surface_fli'][:] = [32.729299106026254,130.91719642410501,32.729299106026254,130.91719642410501]
    physical[missing][:] = np.nan
    registry = build_fuel_lookup_arrays(FuelModels())
    expected = native.model_batch('crown',physical,{},registry,[4],{})
    actual = native.model_fields('crown',list(expected),physical,{},registry,[4],{})
    for name in expected:
        np.testing.assert_array_equal(actual[name],expected[name],err_msg=name)


def test_scoped_invalid_cells_have_initialized_missing_outputs():
    physical = dict(canopy_base_height=np.array([4.,4.]),canopy_height=np.array([30.,30.]),
                    canopy_bulk_density=np.array([.01,.01]))
    actual = native.model_fields('crown',['canopy_heat_per_unit_area'],physical,
        dict(validity=np.array([True,False])),build_fuel_lookup_arrays(FuelModels()),[2],{})
    np.testing.assert_array_equal(actual['canopy_heat_per_unit_area'],[2080.,np.nan])


def test_spread_group_uses_only_real_weather_and_moisture_columns():
    physical = {name: np.full(2, value) for name, value in dict(m1h=.06,m10h=.07,m100h=.08,
        mlh=.6,mlw=.9,wind_speed=88.,wind_direction=35.,slope=12.,aspect=0.).items()}
    actual = native.model_fields('crown', ['crown_fire_spread_rate','crown_length_to_width_ratio'],
        physical, {}, build_fuel_lookup_arrays(FuelModels()), [2], {})
    assert np.all(np.isfinite(actual['crown_fire_spread_rate']))
    assert np.all(actual['crown_length_to_width_ratio'] >= 1.)


def test_surface_fuel_group_never_reads_unbound_weather_or_canopy():
    physical = {name: np.full(2,value) for name,value in dict(m1h=.06,m10h=.07,m100h=.08,mlh=.6,mlw=.9).items()}
    fields = ['reaction_intensity','residence_time','heat_per_unit_area','no_wind_no_slope_spread_rate']
    actual = native.model_fields('surface',fields,physical,dict(fuel_model_grid=np.array([10,124],np.int32)),
        build_fuel_lookup_arrays(FuelModels()),[2],{})
    assert set(actual) == set(fields)
    assert all(np.all(np.isfinite(value)) for value in actual.values())
    assert np.all(actual['reaction_intensity'] > 0)


def test_surface_user_waf_midflame_group_has_no_canopy_or_moisture_reads():
    actual = native.model_fields('surface',['midflame_wind_speed'],
        dict(wind_speed=np.array([88.,176.]),user_waf=np.array([.4])),
        dict(fuel_model_grid=np.array([10,10],np.int32)),
        build_fuel_lookup_arrays(FuelModels()),[2],dict(waf_method=1))
    np.testing.assert_allclose(actual['midflame_wind_speed'],[35.2,70.4])


def test_unbound_required_field_is_error_before_callback():
    called = []
    with pytest.raises(ValueError, match='missing|unbound'):
        native.model_fields('crown', ['canopy_heat_per_unit_area'],
            dict(canopy_height=np.array([30.])), {}, build_fuel_lookup_arrays(FuelModels()), [1], {},
            _owned_callback=lambda: called.append(True))
    assert not called


def test_upslope_spread_group_never_reads_an_unbound_aspect():
    physical={name:np.full(2,value) for name,value in dict(m1h=.06,m10h=.07,m100h=.08,
        mlh=.6,mlw=.9,wind_speed=88.,wind_direction=35.,slope=12.).items()}
    actual=native.model_fields('crown',['crown_fire_spread_rate'],physical,{},
        build_fuel_lookup_arrays(FuelModels()),[2],{})
    assert np.all(np.isfinite(actual['crown_fire_spread_rate']))
