"""Review regressions preserve Python's input domain when native execution is unavailable."""
import pickle
import warnings

import numpy as np
import pytest

import behave_fire.auto as auto
from behave_fire import BehaveBackendWarning, BehaveRun, FuelModels
from behave_fire._auto_namespace import AutoNamespace
from behave_fire.python import BehaveRun as PythonRun


class _NoCopyArray(np.ndarray):
    def copy(self, order='C'):
        raise AssertionError('Python preference must not snapshot raster arguments')


class _NoDeepcopyLabel(str):
    def __deepcopy__(self, memo):
        raise AssertionError('Explicit Rust must not snapshot unused registry labels')


def _crown_arguments():
    return dict(surface_results=dict(spread_rate=np.full(2, 2.), fireline_intensity=np.full(2, 200.),
                    heat_per_unit_area=np.array([1000.+.25j, 1000.+.5j])),
                fuel_model_grid=np.full(2, 10, np.int32), m1h=np.full(2, .06), m10h=np.full(2, .07),
                m100h=np.full(2, .08), mlh=np.full(2, .6), mlw=np.full(2, .9), wind_speed=np.full(2, 8.),
                wind_speed_units=0, wind_direction=35., wind_orientation_mode='RelativeToUpslope',
                slope=0., canopy_base_height=4., canopy_height=30., canopy_bulk_density=.01,
                moisture_foliar=100., out_units={'spread_rate': 0})


def _prefer_python(monkeypatch, status):
    monkeypatch.setattr(auto, '_probe', lambda: (status, ImportError('native unavailable for review test')))
    monkeypatch.setattr(auto, '_WARNING_EMITTED', False)
    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False)


def test_explicit_rust_ignores_nonphysical_registry_copy_hooks():
    pytest.importorskip('behave_fire._rust')
    fuels = FuelModels()
    fuels.fuel_models_[1]['name'] = _NoDeepcopyLabel('caller metadata')
    model = BehaveRun(fuels, backend='rust')
    actual = model.calculate_fire_length(2., 1., 0, 3., 0)
    np.testing.assert_allclose(actual, 9.)
    assert model.backend_in_use == 'rust'


def test_explicit_rust_without_auto_snapshot_retains_pickle_behavior():
    pytest.importorskip('behave_fire._rust')
    model = pickle.loads(pickle.dumps(BehaveRun(FuelModels(), backend='rust')))
    np.testing.assert_allclose(model.calculate_fire_length(2., 1., 0, 3., 0), 9.)
    assert model.backend_in_use == 'rust'


@pytest.mark.parametrize('value', [np.complex64(3.+.25j), np.complex128(3.+.25j)])
def test_numpy_complex_elapsed_scalar_uses_proven_area_formula(monkeypatch, value):
    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False)
    namespace = AutoNamespace('rust')
    with pytest.warns(BehaveBackendWarning, match='Complex'):
        actual = namespace.surface.calculate_fire_area(2., 1., 2., value)
    expected = np.pi * (1.5 * complex(value) / 2.) * (1.5 * complex(value))
    np.testing.assert_allclose(actual, expected)
    assert np.imag(actual) != 0.
    assert namespace.backend_in_use == 'python'


@pytest.mark.parametrize('status', ['absent', 'unusable'])
@pytest.mark.parametrize('route', ['model', 'namespace'])
def test_python_preference_does_not_copy_raster_arguments(monkeypatch, status, route):
    _prefer_python(monkeypatch, status)
    values = np.array([2., 4.]).view(_NoCopyArray)
    with pytest.warns(BehaveBackendWarning, match='using Python'):
        selected = BehaveRun(FuelModels()) if route == 'model' else auto.get_backend()
    if route == 'model':
        actual = selected.calculate_fire_length(values, 1., 0, 3., 0)
    else:
        actual = selected.surface.calculate_fire_length(values, 1., 3.)
    np.testing.assert_allclose(actual, [9., 15.])
    assert selected.backend_in_use == 'python'


@pytest.mark.parametrize('status', ['absent', 'unusable'])
def test_python_preference_model_keeps_original_complex_output_unit_calls(monkeypatch, status):
    _prefer_python(monkeypatch, status)
    arguments = _crown_arguments()
    expected = PythonRun(FuelModels()).do_crown_run(**arguments)
    with pytest.warns(BehaveBackendWarning, match='using Python'):
        model = BehaveRun(FuelModels())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', BehaveBackendWarning)
        actual = model.do_crown_run(**arguments)
    assert not caught
    for field in expected:
        np.testing.assert_allclose(actual[field], expected[field])
    assert np.any(np.imag(actual['crown_fire_heat_per_unit_area']) != 0.)
    assert model.backend_in_use == 'python'


@pytest.mark.parametrize('status', ['absent', 'unusable'])
def test_python_preference_namespace_keeps_original_complex_domain(monkeypatch, status):
    _prefer_python(monkeypatch, status)
    with pytest.warns(BehaveBackendWarning, match='using Python'):
        namespace = auto.get_backend()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', BehaveBackendWarning)
        actual = namespace.surface.calculate_fire_length(np.array([2.+.25j]), 1., 3.)
    np.testing.assert_allclose(actual, [9.+.75j])
    assert not caught
    assert namespace.backend_in_use == 'python'
