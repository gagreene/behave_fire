"""Owned staged grids use actual fields, independent buffers, and explicit lifetimes."""
import pickle

import numpy as np
import pytest

pytest.importorskip('behave_fire._rust')
from behave_fire.rust.grid import CrownGrid, SurfaceGrid


def _crown_inputs():
    surface = {name: value for name, value in _surface_inputs().items()
               if name not in ('canopy_cover', 'crown_ratio', 'fuel_model_grid')}
    return dict(surface, canopy_base_height=3., canopy_bulk_density=0.02,
                moisture_foliar=1., surface_ros=5., surface_fli=1000., surface_hpua=1000.)


def _surface_inputs():
    return dict(m1h=0.06, m10h=0.07, m100h=0.08, mlh=0.6, mlw=0.9,
                wind_speed=88., wind_direction=35., slope=12., aspect=15.,
                canopy_cover=0.4, canopy_height=30., crown_ratio=0.5, fuel_model_grid=124)


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('shape', [(), (0, 3), (7, 3)])
@pytest.mark.parametrize('kind,inputs', [(SurfaceGrid, _surface_inputs), (CrownGrid, _crown_inputs)])
def test_fused_and_staged_complete_snapshots_match(kind, inputs, shape, threads):
    staged = kind(inputs(), shape=shape, threads=threads)
    fused = kind(inputs(), shape=shape, threads=threads)
    staged.run_remaining()
    fused.run_fused()
    for name in staged.outputs:
        assert staged.get(name).shape == shape
        np.testing.assert_allclose(staged.get(name), fused.get(name), rtol=2e-12, atol=1e-10, equal_nan=True)


def test_injection_liveness_and_fused_restart_are_visible():
    grid = CrownGrid(_crown_inputs(), shape=(3,), outputs=('final_spread_rate',), keep=('fire_type',))
    with pytest.raises(ValueError, match='freed or uncomputed'):
        grid.get('final_spread_rate')
    grid.set_code('fire_type', 1)
    grid.set('crown_fraction_burned', 0.5)
    grid.run_remaining()
    rate = grid.get('final_spread_rate')
    grid.set('crown_fraction_burned', 0.)
    grid.run_remaining()
    np.testing.assert_array_equal(grid.get('final_spread_rate'), np.full(3, 5.))
    assert np.any(rate != 5.)
    with pytest.raises(ValueError, match='freed or uncomputed'):
        grid.get('reaction_intensity')
    grid.run_fused()
    assert np.any(grid.get('final_spread_rate') != 5.)


def test_inputs_and_getters_own_storage_and_sessions_cannot_pickle():
    original = np.array([0.05, 0.06, 0.07])
    inputs = dict(_surface_inputs(), m1h=original)
    grid = SurfaceGrid(inputs, outputs=('reaction_intensity',))
    original[:] = 0.8
    np.testing.assert_array_equal(grid.get('m1h'), [0.05, 0.06, 0.07])
    edited = grid.get('m1h')
    edited[:] = 1.
    np.testing.assert_array_equal(grid.get('m1h'), [0.05, 0.06, 0.07])
    grid.override('reaction_intensity', 123.)
    grid.set('reaction_intensity', np.nan)
    grid.run_remaining()
    assert np.isnan(grid.get('reaction_intensity')).all()
    grid.run_stage('reaction')
    np.testing.assert_array_equal(grid.get('reaction_intensity'), [123.] * 3)
    grid.set('m1h', 0.1)
    grid.reset()
    np.testing.assert_array_equal(grid.get('m1h'), [0.1] * 3)
    with pytest.raises(TypeError, match='Grid sessions cannot be pickled'):
        pickle.dumps(grid)


def test_masks_complex_and_typed_control_validation():
    inputs = dict(_surface_inputs(), m1h=np.ma.array([0.05, 0.06], mask=[False, True]))
    grid = SurfaceGrid(inputs)
    assert np.isnan(grid.get('m1h')[1])
    with pytest.raises(TypeError, match='real numeric'):
        grid.set('m1h', 0.05 + 1j)
    with pytest.raises(TypeError, match='integral'):
        grid.set_code('fuel_model_grid', 1.5)
    with pytest.raises(ValueError):
        grid.set('reaction_intensity', [1., 2., 3.])
    with pytest.raises(ValueError):
        grid.run_stage('crown_heat')
    grid.set_flag('validity', [True, False])
    grid.run_remaining()
    assert np.isnan(grid.get('spread_rate')[1])


def test_nan_injection_invalidates_only_heat_descendants():
    grid = CrownGrid(_crown_inputs(), shape=(3,), keep=('reaction_intensity',))
    grid.set_code('fire_type', 1)
    grid.run_remaining()
    reaction, rate, ratio = grid.get('reaction_intensity'), grid.get('final_spread_rate'), grid.get('crown_length_to_width_ratio')
    grid.set('canopy_heat_per_unit_area', np.nan)
    np.testing.assert_array_equal(grid.get('reaction_intensity'), reaction)
    np.testing.assert_array_equal(grid.get('crown_length_to_width_ratio'), ratio)
    with pytest.raises(ValueError, match='freed or uncomputed'):
        grid.get('final_fireline_intensity')
    grid.run_remaining()
    np.testing.assert_array_equal(grid.get('final_spread_rate'), rate)
    assert np.isnan(grid.get('final_fireline_intensity')).all()


@pytest.mark.parametrize('code', [-1, 0, 1, 2, 3])
def test_selected_final_branch_preserves_independent_surface_values_with_missing_heat(code):
    grid = CrownGrid(dict(_crown_inputs(), surface_hpua=np.nan), shape=(3,))
    grid.set_code('fire_type', code)
    grid.run_remaining()
    if code in (0, 2):
        np.testing.assert_array_equal(grid.get('final_spread_rate'), [5.] * 3)
        np.testing.assert_array_equal(grid.get('final_fireline_intensity'), [1000.] * 3)
        assert np.isfinite(grid.get('final_flame_length')).all()
    elif code == 3:
        assert np.isfinite(grid.get('final_spread_rate')).all()
        assert np.isnan(grid.get('final_fireline_intensity')).all()
    else:
        assert np.isnan(grid.get('final_spread_rate')).all()
        assert np.isnan(grid.get('final_fireline_intensity')).all()
    assert np.isnan(grid.get('final_heat_per_unit_area')).all()


def test_rust_only_grids_are_accessible_through_auto_without_complex_replay():
    from behave_fire._auto_namespace import AutoNamespace

    namespace = AutoNamespace('rust')
    assert namespace.SurfaceGrid is SurfaceGrid
    assert 'SurfaceGrid' in dir(namespace)
    with pytest.raises(TypeError, match='real numeric'):
        namespace.SurfaceGrid(dict(_surface_inputs(), m1h=0.05 + 1j))
    with pytest.raises(ImportError, match='Rust-only'):
        AutoNamespace('python').SurfaceGrid


@pytest.mark.parametrize('kind,inputs', [(SurfaceGrid, _surface_inputs), (CrownGrid, _crown_inputs)])
@pytest.mark.parametrize('missing', ['m1h', 'mlh', 'wind_speed', 'wind_direction', 'slope', 'canopy_height'])
def test_staged_missing_inputs_match_fused_per_field_availability(kind, inputs, missing):
    values = dict(inputs(), **{missing: np.array([np.nan, 0., 1.])})
    staged, fused = kind(values, threads=4), kind(values, threads=1)
    staged.run_remaining()
    fused.run_fused()
    for name in staged.outputs:
        np.testing.assert_allclose(staged.get(name), fused.get(name), rtol=2e-12, atol=1e-10, equal_nan=True,
                                   err_msg=f'{missing}: {name}')


@pytest.mark.parametrize('missing', ['surface_ros', 'surface_fli', 'surface_hpua', 'canopy_base_height',
                                     'canopy_bulk_density', 'moisture_foliar'])
def test_staged_missing_supplied_crown_fields_match_fused(missing):
    values = dict(_crown_inputs(), **{missing: np.array([np.nan, 0., 1.])})
    staged, fused = CrownGrid(values, threads=4), CrownGrid(values, threads=1)
    staged.run_remaining()
    fused.run_fused()
    for name in staged.outputs:
        np.testing.assert_allclose(staged.get(name), fused.get(name), rtol=2e-12, atol=1e-10, equal_nan=True,
                                   err_msg=f'{missing}: {name}')


def test_supplied_fuelbed_snapshot_is_owned_editable_and_has_no_particle_source():
    from behave_fire.rust._bridge import _EMPTY_KEYS

    original = SurfaceGrid(_surface_inputs(), shape=(2,))
    original.run_stage('fuelbed')
    bed = {name: original.get('ib.' + name) for name in _EMPTY_KEYS['calculate_fuelbed_intermediates']}
    supplied = SurfaceGrid(_surface_inputs(), fuelbed=bed, shape=(2,))
    bed['sigma'][:] = 0.
    original.run_remaining()
    supplied.run_remaining()
    for name in original.outputs:
        np.testing.assert_allclose(supplied.get(name), original.get(name), rtol=2e-12, atol=1e-10)
    with pytest.raises(ValueError, match='no upstream producer'):
        supplied.run_stage('particles')
    with pytest.raises(ValueError, match='no registry source'):
        supplied.set_fuels(None)
    supplied.set('ib.sigma', 1800.)
    supplied.reset()
    np.testing.assert_array_equal(supplied.get('ib.sigma'), [1800., 1800.])


def test_wind_coefficient_injection_reaches_surface_phi_and_crown_backsolve():
    grid = CrownGrid(_crown_inputs(), shape=(3,), keep=('wind_c', 'wind_phi', 'crowning_surface_fire_spread_rate'))
    grid.run_remaining()
    before_phi, before_backsolve = grid.get('wind_phi'), grid.get('crowning_surface_fire_spread_rate')
    grid.set('wind_c', grid.get('wind_c') * 0.5)
    grid.run_remaining()
    np.testing.assert_allclose(grid.get('wind_phi'), before_phi * 0.5)
    assert np.all(grid.get('crowning_surface_fire_spread_rate') != before_backsolve)
