"""Final review regressions enforce one numeric/availability policy across public APIs."""
import numpy as np
import pytest

from test_native_grid_session_array import _crown_inputs, _surface_inputs

pytestmark = pytest.mark.rust


@pytest.mark.parametrize('persistent', [False, True])
def test_classification_flag_injections_select_final_sources(persistent):
    from behave_fire.rust import CrownGrid

    overrides = {'is_surface_fire': True} if persistent else None
    grid = CrownGrid(_crown_inputs(), shape=(3,), outputs=('final_spread_rate',),
                     keep=('is_surface_fire', 'is_passive_crown', 'is_active_crown'), overrides=overrides)
    grid.set_code('fire_type', 1)
    grid.set('crown_fraction_burned', .5)
    grid.run_remaining()
    if not persistent:
        assert np.any(grid.get('final_spread_rate') != 5.)
        grid.set_flag('is_surface_fire', True)
        grid.run_remaining()
    np.testing.assert_array_equal(grid.get('final_spread_rate'), np.full(3, 5.))
    if persistent:
        grid.run_fused()
        np.testing.assert_array_equal(grid.get('final_spread_rate'), np.full(3, 5.))


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('persistent', [False, True])
def test_invalid_cells_hide_pins_and_restore_original_values_when_reenabled(fused, persistent):
    from behave_fire.rust import SurfaceGrid

    grid = SurfaceGrid(dict(_surface_inputs(), validity=[False, True]), shape=(2,),
        outputs=('spread_rate', 'flame_length'), overrides={'spread_rate': 123.} if persistent else None)
    if not persistent:
        grid.set('spread_rate', 123.)
    if fused:
        grid.run_fused()
    else:
        grid.run_remaining()
    assert np.isnan(grid.get('spread_rate')[0])
    assert np.isnan(grid.get('flame_length')[0])
    if persistent or not fused:
        assert grid.get('spread_rate')[1] == 123.
        grid.set_flag('validity', True)
        grid.run_remaining()
        np.testing.assert_array_equal(grid.get('spread_rate'), [123., 123.])


@pytest.mark.parametrize('family', ['Fraction', 'Slope', 'Temperature', 'BasalArea', 'HeatOfCombustion', 'HeatSink'])
@pytest.mark.parametrize('invalid', [np.nan, np.inf, 1.+2j, np.ma.array(0, mask=True)])
def test_missing_or_nonreal_unit_selectors_raise_before_default_branch(family, invalid):
    from behave_fire.rust import behave_units

    with pytest.raises((TypeError, ValueError)):
        getattr(behave_units, family+'Units').toBaseUnits(10., invalid)


@pytest.mark.parametrize('name,args', [
    ('calculate_no_wind_no_slope_spread_rate', (1., 1., np.nan)),
    ('calculate_residence_time', (np.nan,)),
    ('calculate_wind_adjustment_factor', (0., 0., 0., np.nan)),
    ('calculate_effective_wind_speed', (1., np.nan, 1., 1., 1., 1.)),
])
def test_public_surface_helpers_keep_unknown_thresholds_unavailable(name, args):
    from behave_fire.rust import surface

    assert np.isnan(getattr(surface, name)(*args)).all()


def test_safety_missing_occupancy_preserves_only_independent_separation():
    from behave_fire.rust import safety

    result = safety.calculate_safety_zone(10., np.nan, 10., 0., 0.)
    np.testing.assert_array_equal(result['separation_distance'], 40.)
    assert np.isnan(result['radius']).all() and np.isnan(result['area']).all()


def test_spotting_unknown_required_wind_is_not_valid_zero():
    from behave_fire.rust import spot

    assert np.isnan(spot.calculate_spotting_from_burning_pile(10., np.nan, 10.)).all()


def test_unselected_missing_inputs_do_not_poison_known_zero_or_sheltered_wind():
    from behave_fire.rust import spot, surface

    assert spot.calculate_spotting_from_burning_pile(np.nan, 0., np.nan).item() == 0.
    assert spot.calculate_spotting_from_torching_trees(np.nan, 60., 3., 0., np.nan).item() == 0.
    assert surface.calculate_wind_phi(np.nan, 0., 1., 1., .3, 1.).item() == 0.
    assert surface.calculate_effective_wind_speed(1., 1., np.nan, 1., 1., 1.).item() == 0.
    assert surface.calculate_effective_wind_speed(.5, 1., 1., np.nan, 1., 1.).item() == 0.
    assert np.isfinite(surface.calculate_wind_adjustment_factor(.8, 30., .5, np.nan)).all()
