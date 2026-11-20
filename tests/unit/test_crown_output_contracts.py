"""Python Crown final flame and 20-ft wind geometry contracts."""

import numpy as np
import pytest

from behave_fire import BehaveRun, FuelModels
from behave_fire.components import crown
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays


def _moisture_grid(shape):
    return {
        'm1h': np.full(shape, 0.06),
        'm10h': np.full(shape, 0.07),
        'm100h': np.full(shape, 0.08),
        'mlh': np.full(shape, 0.60),
        'mlw': np.full(shape, 0.90),
    }


# ---- final flame tests ----


def test_assign_final_flame_uses_surface_cutoff_and_preserves_dtype():
    cutoff = np.float32(1e-7)
    surface_fli = np.array(
        [np.nextafter(cutoff, np.float32(0.0)), cutoff, 60.0, 0.0, -1.0, np.nan, 60.0, 60.0],
        dtype=np.float32,
    ).reshape(2, 4)
    result = crown.assign_final_crown_fire_behavior(
        is_surface_fire=np.array([[True, True, True, True], [True, True, False, False]]),
        is_passive_crown=np.array([False, False, True, False]),
        surface_ros=6.0,
        surface_hpua=600.0,
        surface_fli=surface_fli,
        passive_ros=9.0,
        passive_hpua=1000.0,
        passive_fli=150.0,
        passive_fl=np.float32(7.0),
        crown_ros=18.0,
        crown_hpua=1200.0,
        crown_fli=360.0,
        crown_fl=np.float32(9.0),
    )

    flame_length = result['final_flame_length']
    assert flame_length.dtype == np.float32
    np.testing.assert_allclose(
        flame_length, [[0.0, 0.45 * (1e-7 ** 0.46), 0.45 * (60.0 ** 0.46), 0.0], [0.0, 0.0, 7.0, 9.0]],
        rtol=1e-6, atol=0.0,
    )


def test_crown_final_flame_follows_selected_behavior_through_facade():
    model = BehaveRun(FuelModels(), backend='python')
    inputs = {
        'fuel_model_grid': np.full(5, 10),
        **_moisture_grid((5,)),
        'wind_speed': np.array([0.0, 5.0, 15.0, 7.0, 25.0]),
        'wind_speed_units': 5,
        'wind_direction': 0.0,
        'wind_orientation_mode': 'RelativeToUpslope',
        'slope': 0.0,
        'canopy_height': 30.0,
        'canopy_base_height': np.array([15.0, 15.0, 15.0, 5.0, 15.0]),
    }
    surface = model.do_surface_run(**inputs, waf_method='UserInput', user_waf=0.4)
    minimum_surface_results = {key: surface[key] for key in ('spread_rate', 'fireline_intensity', 'heat_per_unit_area')}
    result = model.do_crown_run(surface_results=minimum_surface_results, **inputs, canopy_bulk_density=0.02)

    np.testing.assert_array_equal(result['fire_type'], [0, 0, 2, 1, 3])
    flame_length = result['final_flame_length']
    np.testing.assert_array_equal(flame_length[:3], surface['flame_length'][:3])
    for cell in (3, 4):
        assert flame_length[cell] == pytest.approx(0.2 * result['final_fireline_intensity'][cell] ** (2.0 / 3.0))


# ---- geometry tests ----


def test_crown_geometry_uses_circular_ratio_for_calm_and_missing_wind():
    with np.errstate(invalid='ignore'):
        result = crown.calculate_crown_surface_fire(
            lut=build_fuel_lookup_arrays(FuelModels()),
            S=(3,),
            **_moisture_grid((3,)),
            wind_speed=np.array([0.0, np.nan, -1.0]),
            wind_speed_units=5,
            wind_direction=0.0,
            wind_orientation_mode='RelativeToUpslope',
            slope_deg=25.0,
            aspect=0.0,
            canopy_height=30.0,
        )

    np.testing.assert_array_equal(result['crown_length_to_width_ratio'], [1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ('wind_speed_units', 'wind_speed'),
    [(0, 880.0), (1, 800.0), (2, 880.0 / 196.8503937), (3, 880.0 / 3.28084),
     (4, 880.0 / 0.0547), (5, np.float32(10.0)), (6, 880.0 / 54.680665)],
)
def test_crown_geometry_uses_twenty_foot_wind_in_each_supported_unit(wind_speed_units, wind_speed):
    slope = np.array([[0.0, 25.0], [5.0, 50.0]])
    result = crown.calculate_crown_surface_fire(
        lut=build_fuel_lookup_arrays(FuelModels()),
        S=slope.shape,
        **_moisture_grid(slope.shape),
        wind_speed=wind_speed,
        wind_speed_units=wind_speed_units,
        wind_direction=0.0,
        wind_orientation_mode='RelativeToUpslope',
        slope_deg=slope,
        aspect=0.0,
        canopy_height=30.0,
    )

    ratio = result['crown_length_to_width_ratio']
    assert ratio.shape == slope.shape
    assert ratio.dtype == np.float64
    np.testing.assert_allclose(ratio, np.full(slope.shape, 2.25), rtol=5e-10, atol=0.0)
