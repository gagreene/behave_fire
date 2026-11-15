"""Regressions for the owner-approved Surface boundaries in pinned Behave C++."""

from importlib import import_module

import numpy as np
import pytest


@pytest.fixture(params=['python', pytest.param('rust', marks=pytest.mark.rust)])
def _surface_backend(request):
    return import_module('behave_fire.components.surface' if request.param == 'python'
                         else 'behave_fire.rust.surface')


@pytest.mark.parametrize('scalar', [False, True])
def test_effective_wind_preserves_elongation_below_old_spread_cutoff(_surface_backend, scalar):
    """Tiny positive spread must not erase an existing wind-driven ellipse."""
    rates = [1e-3, np.nextafter(1e-7, np.inf), 1e-7, np.nextafter(1e-7, 0), 1e-8, 5.23839427461715e-15]
    for wind, expected_ratio in [(0., 1.), (88., 1.0829361181265)]:
        for baseline in rates if scalar else [np.array(rates)]:
            spread = baseline * (1. + .01 * wind ** 2)
            effective = _surface_backend.calculate_effective_wind_speed(spread, baseline, 2., .01, 1., 1.)
            ratio = _surface_backend.calculate_fire_length_to_width_ratio(effective)
            np.testing.assert_allclose(ratio, expected_ratio, rtol=2e-14, atol=0)


def test_effective_wind_zero_and_negative_denominators_remain_guarded(_surface_backend):
    """Removing the positive-spread cutoff must retain safe zero-denominator handling."""
    result = _surface_backend.calculate_effective_wind_speed(
        np.array([0., 1., -1.]), np.array([0., 0., -1.]), 2., .01, 1., 1.)
    np.testing.assert_array_equal(result, np.zeros(3))


@pytest.mark.parametrize('scalar', [False, True])
def test_flame_length_uses_strict_cpp_intensity_cutoff(_surface_backend, scalar):
    """Zero subthreshold flames while retaining the Byram law at the threshold."""
    intensities = np.array([-1., 0., 1e-10, 1e-8, np.nextafter(1e-7, 0), 1e-7,
                            np.nextafter(1e-7, np.inf), 1e-6, 1.])
    expected = np.array([0., 0., 0., 0., 0., .0002711518137334609,
                         .0002711518137334609, .0007820103729372187, .45])
    if scalar:
        actual = np.array([_surface_backend.calculate_flame_length(value).item() for value in intensities])
    else:
        actual = _surface_backend.calculate_flame_length(intensities)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)


@pytest.mark.rust
@pytest.mark.parametrize('fused', [False, True])
def test_grid_boundary_injections_reach_shape_and_flame_outputs(fused):
    """Arbitrary grid overrides must honor boundaries, including fused-call DAG fallback."""
    from behave_fire.rust import SurfaceGrid

    inputs = dict(fuel_model_grid=1, m1h=.06, m10h=.07, m100h=.08, mlh=.6, mlw=.9,
                  wind_speed=88., wind_direction=0., slope=0., aspect=0.,
                  canopy_cover=0., canopy_height=0., crown_ratio=0.)
    overrides = dict(no_wind_no_slope_spread_rate=1e-8, spread_rate=78.44e-8,
                     wind_b=2., wind_c=.01, wind_e=1., fireline_intensity=[1e-8, 1e-7])
    overrides['ib.relative_packing_ratio'] = 1.
    grid = SurfaceGrid(inputs, shape=(2,), outputs=('fire_length_to_width_ratio', 'flame_length'),
                       overrides=overrides)
    if fused:
        grid.run_fused()
    else:
        grid.run_remaining()
    np.testing.assert_allclose(grid.get('fire_length_to_width_ratio'), 1.0829361181265, rtol=2e-14, atol=0)
    np.testing.assert_allclose(grid.get('flame_length'), [0., .0002711518137334609], rtol=2e-14, atol=0)


@pytest.mark.rust
def test_grid_fused_supported_tweaks_preserve_small_spread_shape():
    """Supported tweaks must exercise the fused kernel with real fuel coefficients."""
    from behave_fire.rust import SurfaceGrid

    inputs = dict(fuel_model_grid=1, m1h=.06, m10h=.07, m100h=.08, mlh=.6, mlw=.9,
                  wind_speed=88., wind_direction=0., slope=0., aspect=0.,
                  canopy_cover=0., canopy_height=0., crown_ratio=0.)
    # Every requested output and override belongs to the fused Surface allowlist.
    grid = SurfaceGrid(inputs, shape=(2,),
                       outputs=('fire_length_to_width_ratio', 'flame_length', 'fireline_intensity'),
                       overrides=dict(no_wind_no_slope_spread_rate=1e-8,
                                      midflame_wind_speed=[0., 88.], reaction_intensity=1e-10))
    grid.run_fused()
    np.testing.assert_allclose(grid.get('fire_length_to_width_ratio'), [1., 1.0829361181265],
                               rtol=2e-14, atol=0)
    intensity = grid.get('fireline_intensity')
    assert np.all((intensity > 0.) & (intensity < 1e-7))
    np.testing.assert_array_equal(grid.get('flame_length'), np.zeros(2))
