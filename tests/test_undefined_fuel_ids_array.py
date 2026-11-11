"""Fuel identifiers outside the C++ range use the existing undefined-row path."""

import numpy as np
import pytest

from behave_fire import python as python_backend


pytestmark = pytest.mark.rust
INVALID_IDS = [-100, -1, 0, 257, 261, 999, np.iinfo(np.int32).max, np.nan, np.inf, -np.inf]
BACKENDS = ['python', 'rust']


def _arguments(ids):
    shape = np.atleast_1d(ids).shape
    return dict(fuel_model_grid=ids,
                **{name: np.full(shape, value) for name, value in
                   [('m1h', .06), ('m10h', .07), ('m100h', .08), ('mlh', .6), ('mlw', .9),
                    ('wind_speed', 88.), ('wind_direction', 35.), ('slope', 12.)]},
                wind_speed_units=0, wind_orientation_mode='RelativeToUpslope',
                aspect=15., canopy_height=30.)


def _backend(name):
    if name == 'python':
        return python_backend
    from behave_fire import rust
    return rust


def _custom(fuels, number):
    return fuels.set_custom_fuel_model(
        number, 'CST', 'Upper endpoint control', 1., 0, .25, 0, 8000., 8000., 0,
        .04, .01, .005, .02, .01, 0, 1800., 1500., 1200., 0, False)


def _run(backend, ids, crown=False):
    backend = _backend(backend)
    fuels = backend.FuelModels()
    assert _custom(fuels, 256)
    runner = backend.BehaveRun(fuels)
    arguments = _arguments(ids)
    surface = runner.do_surface_run(**arguments, canopy_cover=.4, crown_ratio=.5)
    if not crown:
        return surface
    return runner.do_crown_run(**arguments, surface_results=surface,
                              canopy_base_height=3., canopy_bulk_density=.02, moisture_foliar=100.)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('number', INVALID_IDS + [258, 259, 260])
def test_custom_registration_rejects_undefined_ids_without_mutation(backend, number):
    backend = _backend(backend)
    fuels = backend.FuelModels()
    before = dict(fuels.fuel_models_)
    assert _custom(fuels, number) is False
    assert fuels.fuel_models_ == before
    assert not fuels.is_fuel_model_defined(number)


@pytest.mark.parametrize('backend', BACKENDS)
def test_custom_upper_endpoint_remains_readable_and_zero_row_stays_undefined(backend):
    name = backend
    backend = _backend(backend)
    fuels = backend.FuelModels()
    assert _custom(fuels, 256)
    assert fuels.is_fuel_model_defined(256)
    lookup = backend.fuel_models_array.build_fuel_lookup_arrays(fuels)
    for name, values in lookup.items():
        assert values.shape == (257,), name
        assert values[0] == 0, name
    assert lookup['is_defined'][256]
    assert _run(name, np.array([256]))['spread_rate'][0] > 0


@pytest.mark.parametrize('number', INVALID_IDS)
@pytest.mark.parametrize('scalar', [True, False])
@pytest.mark.parametrize('crown', [False, True])
def test_invalid_ids_match_undefined_zero_row_and_both_backends(number, scalar, crown):
    ids = number if scalar else np.array([124, number, 256])
    canonical = 0 if scalar else np.array([124, 0, 256], dtype=np.int32)
    results = []
    for backend in BACKENDS:
        actual = _run(backend, ids, crown)
        expected = _run(backend, canonical, crown)
        assert actual.keys() == expected.keys()
        for name in actual:
            assert actual[name].dtype == expected[name].dtype, name
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        spread = actual['final_spread_rate' if crown else 'spread_rate']
        assert spread.reshape(-1)[0 if scalar else 1] == 0
        results.append(actual)
    for name in results[0]:
        np.testing.assert_allclose(results[0][name], results[1][name],
                                   rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('number', INVALID_IDS)
def test_particle_lookup_maps_invalid_ids_to_zero_without_selecting_model_256(backend, number):
    backend = _backend(backend)
    fuels = backend.FuelModels()
    assert _custom(fuels, 256)
    lookup = backend.fuel_models_array.build_fuel_lookup_arrays(fuels)
    actual = backend.surface.build_particle_arrays(lookup, np.array([number]),
                                                   *[np.array([.06]) for _ in range(5)])
    expected = backend.surface.build_particle_arrays(lookup, np.array([0]),
                                                     *[np.array([.06]) for _ in range(5)])
    assert not actual['is_defined'].any()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize('number', INVALID_IDS)
def test_staged_surface_uses_zero_row_without_invalidating_the_cell(number):
    from behave_fire.rust import SurfaceGrid

    arguments = _arguments(np.array([number]))
    arguments.pop('wind_speed_units')
    arguments.pop('wind_orientation_mode')
    arguments.update(canopy_cover=.4, crown_ratio=.5)
    grid = SurfaceGrid(arguments)
    grid.run_fused()
    np.testing.assert_array_equal(grid.get('spread_rate'), [0.])
    np.testing.assert_array_equal(grid.get('validity'), [True])
    grid.set_code('fuel_model_grid', np.array([124]))
    grid.run_fused()
    assert grid.get('spread_rate')[0] > 0
