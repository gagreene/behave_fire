"""Remaining Rust helpers use physical float64 and separately validated controls."""
import pickle

import numpy as np
import pytest

pytestmark = pytest.mark.rust

_UNIT_CLASSES = ('Area', 'BasalArea', 'Density', 'FirelineIntensity', 'Fraction', 'HeatOfCombustion',
                 'HeatPerUnitArea', 'HeatSink', 'HeatSourceAndReactionIntensity', 'Length', 'Loading',
                 'Pressure', 'Slope', 'Speed', 'SurfaceAreaToVolume', 'Temperature', 'Time')


@pytest.mark.parametrize('family', _UNIT_CLASSES)
def test_all_unit_families_widen_once_and_preserve_missing_physical_values(family):
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units as native

    value = np.ma.array(np.array([1., 2., np.nan], np.float32), mask=[False, True, False])
    source = np.ma.asarray(value, dtype=np.float64).filled(np.nan)
    for method in ('toBaseUnits', 'fromBaseUnits'):
        actual = getattr(getattr(native, family + 'Units'), method)(value, 0)
        expected = getattr(getattr(reference, family + 'Units'), method)(source, 0)
        assert type(actual) is np.ndarray and actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=2e-14, equal_nan=True)






def test_crown_components_use_float64_and_unavailable_codes_without_mask_payloads():
    from behave_fire.rust import crown

    heat = crown.calculate_crown_heat_and_intensity(np.array([1000.], np.float32),
                                                  np.array([5.], np.float32), 3., 30., 0.02)
    assert all(value.dtype == np.float64 and type(value) is np.ndarray for value in heat.values())
    result = crown.classify_crown_fire_type(np.array([np.nan, 1.]), np.array([1., np.nan]))
    np.testing.assert_array_equal(result['fire_type'], [-1, -1])
    for name in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
        assert not result[name].any()
    assert np.isnan(crown.calculate_critical_crown_fire_spread_rate(np.nan)).all()
    assert np.isnan(crown.calculate_surface_fire_critical_spread_rate(100., np.nan)).all()
    missing = crown.calculate_crown_heat_and_intensity(np.nan, 5., 3., 30., 0.02)
    assert np.isfinite(missing['canopy_heat_per_unit_area']).all()
    assert np.isnan(missing['crown_flame_length']).all()


def test_crown_surface_records_keep_fuel_grid_when_weather_is_empty():
    from behave_fire.rust import FuelModels, crown, fuel_models_array

    arguments = dict(lut=fuel_models_array.build_fuel_lookup_arrays(FuelModels()),
        m1h=np.array([0.06]), m10h=np.array([0.07]), m100h=np.array([0.08]), mlh=np.array([0.6]), mlw=np.array([0.9]),
        S=(1,), wind_speed=np.empty(0), wind_speed_units=0, wind_direction=0.,
        wind_orientation_mode='RelativeToUpslope', slope_deg=0., aspect=0., canopy_height=30.)
    result = crown.calculate_crown_surface_fire(**arguments)
    assert result['p_crown']['depth'].shape == (1,)
    assert result['ib_crown']['sigma'].shape == (1,)
    assert result['ri_crown'].shape == (1,)
    assert result['crown_fire_spread_rate'].shape == (0,)


def test_explicit_physical_coercions_preserve_masked_missingness():
    from behave_fire.rust import FireSize, surface

    value = np.ma.array([1., 2.], mask=[False, True])
    intensity = surface.calculate_fireline_intensity(value, 60.)
    np.testing.assert_allclose(intensity, [1., np.nan], equal_nan=True)
    state = FireSize()
    assert np.isnan(state.get_fire_area(False, 0, np.ma.array(1., mask=True), 0))


@pytest.mark.parametrize('invalid', [np.nan, np.ma.array(0, mask=True), 0.5])
def test_fdfm_missing_or_nonintegral_indices_raise_before_lookup(invalid):
    from behave_fire.rust import fine_dead_fuel_moisture_tool as native

    with pytest.raises((ValueError, TypeError)):
        native.calculate_fine_dead_fuel_moisture(invalid, 0, 0, 0, 0, 0, 0, 0)
    result = native.calculate_fine_dead_fuel_moisture(np.array([-1, 0], dtype=np.int32), 0, 0, 0, 0, 0, 0, 0)
    assert result[0] == -0.02
    assert result[1] != -0.02


def test_fire_size_is_float64_state_with_ieee_domains_and_value_serialization():
    from behave_fire.rust import FireSize

    size = FireSize()
    ratio = np.float32(1.2)
    size.fire_length_to_width_ratio_ = ratio
    size.calculate_fire_eccentricity()
    expected = np.sqrt(float(ratio) ** 2 - 1.) / float(ratio)
    assert not isinstance(size.get_eccentricity(), np.float32)
    assert size.get_eccentricity() == pytest.approx(expected, rel=2e-15)
    size.effective_wind_speed_ = 1e6
    size.calculate_surface_fire_length_to_width_ratio()
    assert size.get_fire_length_to_width_ratio() == 8.
    size.fire_length_to_width_ratio_ = 0.5
    size.calculate_elliptical_dimensions()
    assert np.isnan(size.get_heading_to_backing_ratio())
    copied = pickle.loads(pickle.dumps(size))
    assert np.isnan(copied.get_heading_to_backing_ratio())
    copied.effective_wind_speed_ = np.nan
    copied.calculate_surface_fire_length_to_width_ratio()
    assert np.isnan(copied.get_fire_length_to_width_ratio())


def test_fuelbed_characteristics_keep_shape_when_only_heat_inputs_are_empty():
    from behave_fire.rust import FuelModels, fuel_models_array, surface

    lookup = fuel_models_array.build_fuel_lookup_arrays(FuelModels())
    p = surface.build_particle_arrays(lookup, np.array([124]),
        *[np.array([v]) for v in (.06, .07, .08, .6, .9)])
    p['hoc_dead'] = np.empty(0)
    result = surface.calculate_fuelbed_intermediates(p)
    assert result['sigma'].shape == (1,)
    assert result['packing_ratio'].shape == (1,)
    assert result['w_heat_dead'].shape == (0,)


def test_independent_component_outputs_survive_empty_unrelated_inputs():
    from behave_fire.rust import crown, mortality, safety, vapor_pressure_deficit_calculator as vpd

    heat = crown.calculate_crown_heat_and_intensity(np.empty(0), 5., 3., 30., 0.02)
    assert heat['canopy_heat_per_unit_area'].size == 1
    assert heat['crown_fire_line_intensity'].size == 0
    passive = crown.calculate_passive_crown_fire_behavior(5., np.empty(0), 10., 500., 0.5)
    assert passive['passive_crown_fire_spread_rate'].size == 1
    assert passive['passive_crown_fire_line_intensity'].size == 0
    pressures = vpd.calculate_vpd(np.array([25.]), 0, np.empty(0), 0)
    assert pressures['saturated_vp'].shape == (1,)
    zone = safety.calculate_safety_zone(np.array([10.]), np.empty(0), 50., 1., 300.)
    assert zone['separation_distance'].shape == (1,)
    scorched = mortality.calculate_crown_scorch_mortality(np.array([30.]), 80., 0.5,
        np.empty(0), 1, mortality.build_mortality_lookup())
    assert scorched['crown_length_scorch'].shape == (1,)
    assert scorched['probability_mortality'].shape == (0,)


def test_mortality_missing_physical_inputs_propagate_independently_of_control_storage():
    from behave_fire.rust import mortality

    coefficients = mortality.build_mortality_lookup()
    actual = mortality.calculate_crown_scorch_mortality(np.array([np.nan, 30.]), 80., 0.5, 12.,
                                                       np.array([1, 1], np.int32), coefficients)
    for value in actual.values():
        assert value.dtype == np.float64
        assert np.isnan(value[0])
        assert np.isfinite(value[1])
    with pytest.raises(ValueError):
        mortality.calculate_crown_scorch_mortality(30., 80., 0.5, 12., np.nan, coefficients)


def test_mortality_missing_threshold_conditions_are_not_zero_defaults():
    from behave_fire.rust import mortality

    coefficients = mortality.build_mortality_lookup()
    actual = mortality.calculate_crown_scorch_mortality(30., 80., np.nan, 12., 1, coefficients)
    for value in actual.values():
        assert np.isnan(value).all()
    assert np.isnan(mortality.calculate_scorch_height(np.nan, 5., 70.)).all()


def test_primitive_geometry_and_lookup_routes_do_not_enter_the_legacy_payload_decoder():
    from behave_fire import _rust
    from behave_fire.components.fine_dead_fuel_moisture_tool import calculate_fine_dead_fuel_moisture as reference
    from behave_fire.rust import crown, fine_dead_fuel_moisture_tool as fdfm, surface

    assert not hasattr(_rust, 'batch')
    assert surface.calculate_eccentricity(np.array([2.], np.float32)).dtype == np.float64
    assert surface.calculate_flame_length(np.ma.array([2.], mask=True)).dtype == np.float64
    assert surface.calculate_fire_length_to_width_ratio(5.).dtype == np.float64
    assert surface.calculate_fire_area(2., 1., 2., 1., True).dtype == np.float64
    assert crown.classify_crown_fire_type(np.nan, 1.)['fire_type'].dtype == np.int32
    np.testing.assert_allclose(fdfm.calculate_fine_dead_fuel_moisture(np.array([0, -1]), 0, 0, 0, 0, 0, 0, 0),
                               [np.asarray(reference(0, 0, 0, 0, 0, 0, 0, 0)).item(), -0.02])


def test_primitive_snapshots_own_physical_and_flag_operands_before_callbacks():
    from behave_fire import _rust

    original = np.array([2., 3.])

    def mutate_physical():
        original[:] = 9.

    result = _rust.component_batch('surface.calculate_eccentricity',
        {'fire_length_to_width_ratio': original}, {}, {}, [2], _owned_callback=mutate_physical)
    np.testing.assert_allclose(result[''], np.sqrt(np.array([3., 8.])) / np.array([2., 3.]))
    flag = np.array([True])

    def mutate_flag():
        flag[:] = False

    result = _rust.component_batch('surface.calculate_fire_area',
        {name: np.array([value]) for name, value in dict(forward_ros=2., backing_ros=1., lwr=2., elapsed_min=1.).items()},
        {}, {'is_crown': flag}, [1], _owned_callback=mutate_flag)
    np.testing.assert_allclose(result[''], [np.pi * 0.5])
    empty = _rust.component_batch('surface.calculate_eccentricity',
        {'fire_length_to_width_ratio': np.array([], np.float64)}, {}, {}, [0])
    assert empty[''].shape == (0,) and empty[''].dtype == np.float64


def test_surface_geometry_widens_inputs_and_keeps_missing_threshold_operands_missing():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native

    values = np.array([1.1, 2.3, 300.], np.float32)
    actual = native.calculate_eccentricity(values)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, reference.calculate_eccentricity(values.astype(np.float64)), rtol=2e-15)
    flame = native.calculate_flame_length(np.ma.array([np.nan, 0., -1., 1., 2.], mask=[False, False, False, False, True]))
    assert type(flame) is np.ndarray and flame.dtype == np.float64
    np.testing.assert_allclose(flame, [np.nan, 0., 0., 0.45, np.nan], equal_nan=True)
    assert np.isnan(native.calculate_fire_length_to_width_ratio(np.nan)).all()
    negative_zero = native.calculate_fire_length(np.array([-0.]), np.array([-0.]), 1.)
    assert np.signbit(negative_zero).all()


def test_vpd_missing_humidity_retains_independent_saturated_pressure():
    from behave_fire.rust import vapor_pressure_deficit_calculator as native

    result = native.calculate_vpd(np.array([25., np.nan]), 0, np.array([np.nan, 0.5]), 0)
    assert np.isfinite(result['saturated_vp'][0])
    assert np.isnan(result['saturated_vp'][1])
    assert np.isnan(result['vpd']).all()
    assert np.isnan(result['actual_vp']).all()
