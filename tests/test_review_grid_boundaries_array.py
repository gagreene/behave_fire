"""Review regressions for staged raster structural inputs and public lifecycle controls."""
import numpy as np
import pytest

pytest.importorskip('behave_fire._rust')
from behave_fire import _rust as native
from behave_fire.rust import surface
from behave_fire.rust._bridge import _EMPTY_KEYS
from behave_fire.rust.grid import CrownGrid, SurfaceGrid, _fuel_snapshot


def _inputs(kind='surface'):
    result = dict(m1h=.06, m10h=.07, m100h=.08, mlh=.6, mlw=.9,
                  wind_speed=88., wind_direction=35., slope=12., aspect=15.,
                  canopy_cover=.4, canopy_height=30., crown_ratio=.5, fuel_model_grid=124)
    if kind == 'crown':
        for key in ('canopy_cover', 'crown_ratio', 'fuel_model_grid'):
            result.pop(key)
        result.update(canopy_base_height=3., canopy_bulk_density=.02, moisture_foliar=100.,
                      surface_ros=5., surface_fli=1000., surface_hpua=1000.)
    return result


@pytest.fixture
def fuelbed():
    grid = SurfaceGrid(_inputs(), shape=(2,))
    grid.run_stage('fuelbed')
    return {name: grid.get('ib.'+name) for name in _EMPTY_KEYS['calculate_fuelbed_intermediates']}


@pytest.mark.parametrize('operation', ['constructor', 'setter'])
@pytest.mark.parametrize('flag', [-1, 2])
def test_binary_grid_flags_reject_nonbinary_integers(operation, flag):
    with pytest.raises(ValueError, match='0|1|binary|boolean'):
        if operation == 'constructor':
            SurfaceGrid(dict(_inputs(), validity=flag))
        else:
            SurfaceGrid(_inputs()).set_flag('validity', flag)


@pytest.mark.parametrize('kind', ['surface', 'crown'])
@pytest.mark.parametrize('extent', [0, 2])
def test_high_rank_inferred_grid_shapes_preserve_all_axes(kind, extent):
    shape = (1,)*34+(extent,)
    arguments = dict(_inputs(kind), m1h=np.full(shape, .06))
    grid = (SurfaceGrid if kind == 'surface' else CrownGrid)(arguments,
                outputs=('reaction_intensity',))
    grid.run_remaining()
    actual = grid.get('reaction_intensity')
    assert grid.shape == shape
    assert actual.shape == shape
    assert actual.size == extent
    if extent:
        assert np.isfinite(actual).all()


def test_integral_float_fuel_grid_is_accepted_at_construction_and_edit():
    grid = SurfaceGrid(dict(_inputs(), fuel_model_grid=np.array([124., 124.])),
                       outputs=('reaction_intensity',))
    grid.run_remaining()
    before = grid.get('reaction_intensity')
    grid.set_code('fuel_model_grid', np.array([124., 124.]))
    grid.run_remaining()
    np.testing.assert_allclose(grid.get('reaction_intensity'), before)
    np.testing.assert_array_equal(grid.get('fuel_model_grid'), [124, 124])
    with pytest.raises(TypeError):
        CrownGrid(_inputs('crown')).set_code('fire_type', 1.)


@pytest.mark.parametrize('value', [1.5, np.nan, np.inf])
def test_invalid_float_fuel_grid_edit_does_not_mutate_existing_ids(value):
    grid = SurfaceGrid(_inputs(), shape=(2,))
    with pytest.raises((ValueError, TypeError)):
        grid.set_code('fuel_model_grid', value)
    np.testing.assert_array_equal(grid.get('fuel_model_grid'), [124, 124])


def test_north_relative_aspect_edit_invalidates_direction_descendants():
    grid = SurfaceGrid(_inputs(), outputs=('direction_of_max_spread',), keep=('reaction_intensity',),
                       wind_orientation_mode='RelativeToNorth')
    grid.run_remaining()
    original_direction = grid.get('direction_of_max_spread').copy()
    original_reaction = grid.get('reaction_intensity').copy()
    grid.set('aspect', 110.)
    grid.run_remaining()
    fresh = SurfaceGrid(dict(_inputs(), aspect=110.), outputs=('direction_of_max_spread',),
                        wind_orientation_mode='RelativeToNorth')
    fresh.run_fused()
    np.testing.assert_allclose(grid.get('direction_of_max_spread'), fresh.get('direction_of_max_spread'))
    np.testing.assert_allclose(grid.get('reaction_intensity'), original_reaction)
    assert not np.allclose(grid.get('direction_of_max_spread'), original_direction)


def test_particle_builder_accepts_integral_float_ids_through_public_boundary():
    # Particle construction's public lookup uses these established short field names.
    from behave_fire import FuelModels
    from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

    lookup = build_fuel_lookup_arrays(FuelModels())
    arguments = {name: np.full(2, _inputs()[name]) for name in ('m1h','m10h','m100h','mlh','mlw')}
    actual = surface.build_particle_arrays(lookup, np.array([124.,124.]), **arguments)
    expected = surface.build_particle_arrays(lookup, np.array([124,124], dtype=np.int32), **arguments)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_session_clear_operations_and_output_worker_edits_restore_computation():
    grid = SurfaceGrid(_inputs(), shape=(3,), outputs=('reaction_intensity',))
    grid.run_remaining()
    original = grid.get('reaction_intensity').copy()
    grid.override('reaction_intensity', 123.)
    grid.run_remaining()
    np.testing.assert_array_equal(grid.get('reaction_intensity'), [123.]*3)
    grid.clear_override('reaction_intensity')
    grid.run_remaining()
    np.testing.assert_allclose(grid.get('reaction_intensity'), original)
    grid.set('reaction_intensity', 456.)
    grid.run_remaining()
    np.testing.assert_array_equal(grid.get('reaction_intensity'), [456.]*3)
    grid.clear_injection('reaction_intensity')
    grid.set_outputs(('spread_rate',), keep=('reaction_intensity',))
    grid.set_threads(4)
    grid.run_remaining()
    assert grid.threads == 4 and grid.outputs == ('spread_rate',)
    assert np.isfinite(grid.get('spread_rate')).all()
    np.testing.assert_allclose(grid.get('reaction_intensity'), original)


def test_supplied_fuelbed_does_not_require_an_unused_fuel_grid(fuelbed):
    arguments = _inputs()
    arguments.pop('fuel_model_grid')
    grid = SurfaceGrid(arguments, fuelbed=fuelbed, outputs=('spread_rate',))
    grid.run_remaining()
    reference = SurfaceGrid(_inputs(), shape=(2,), outputs=('spread_rate',))
    reference.run_fused()
    np.testing.assert_allclose(grid.get('spread_rate'), reference.get('spread_rate'))


@pytest.mark.parametrize('entry', ['python', 'native'])
def test_supplied_fuelbed_rejects_unknown_columns(fuelbed, entry):
    fuelbed['bogus'] = np.zeros(2)
    with pytest.raises(ValueError, match='unknown|Unknown|bogus'):
        if entry == 'python':
            SurfaceGrid(_inputs(), fuelbed=fuelbed)
        else:
            physical = {name: np.atleast_1d(np.array(value, dtype=np.float64))
                        for name, value in _inputs().items() if name != 'fuel_model_grid'}
            native.GridSession('surface', physical, {'fuel_model_grid': np.array([124],np.int32)},
                               _fuel_snapshot(None), (2,), {}, ('spread_rate',), (), 1, fuelbed)


def test_user_waf_uses_injection_without_requiring_unused_canopy_values():
    grid = SurfaceGrid(dict(_inputs(), canopy_cover=np.nan, canopy_height=np.nan,
                           crown_ratio=np.nan, user_waf=.25), waf_method='UserInput',
                       outputs=('midflame_wind_speed',))
    grid.run_remaining()
    np.testing.assert_allclose(grid.get('midflame_wind_speed'), 22.)
    grid.set('user_waf', .5)
    grid.run_remaining()
    np.testing.assert_allclose(grid.get('midflame_wind_speed'), 44.)
