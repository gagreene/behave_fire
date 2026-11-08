"""The model facade routes one whole operation before native coercion."""
import copy
import importlib
import warnings

import numpy as np
import pytest

import behave_fire.auto as auto
from behave_fire import BehaveBackendWarning, BehaveRun, FuelModels


def _arguments(heat):
    return dict(surface_results=dict(spread_rate=np.array([2., 2.]),
        fireline_intensity=np.array([200., 200.]), heat_per_unit_area=heat),
        fuel_model_grid=np.array([10, 10]), m1h=np.full(2, .06), m10h=np.full(2, .07),
        m100h=np.full(2, .08), mlh=np.full(2, .6), mlw=np.full(2, .9),
        wind_speed=np.full(2, 8.), wind_speed_units=0, wind_direction=np.full(2, 35.),
        wind_orientation_mode='RelativeToUpslope', slope=np.zeros(2), canopy_base_height=4., canopy_height=30.,
        canopy_bulk_density=.01, moisture_foliar=100.)


@pytest.fixture
def delegates(monkeypatch):
    calls = []
    original = importlib.import_module

    class Native:
        def __init__(self, fuels, threads=1):
            calls.append(('construct-rust', fuels))

        def do_crown_run(self, *args, **kwargs):
            calls.append(('rust', kwargs))
            return {'native': True}

    def load(name):
        if name == 'behave_fire.rust':
            from types import SimpleNamespace
            return SimpleNamespace(BehaveRun=Native)
        return original(name)

    monkeypatch.setattr(auto, '_probe', lambda: ('usable', None))
    monkeypatch.setattr(importlib, 'import_module', load)
    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False, raising=False)
    return calls


def test_complex_error_filter_prevents_python_construction_and_call(delegates):
    model = BehaveRun(FuelModels())
    with warnings.catch_warnings():
        warnings.simplefilter('error', BehaveBackendWarning)
        for _ in range(2):
            with pytest.raises(BehaveBackendWarning):
                model.do_crown_run(**_arguments(np.array([1000+.25j, 1000+.5j])))
    assert len(delegates) == 1
    assert model.backend_in_use == 'rust'


@pytest.mark.parametrize('field', ['heat_of_combustion_dead', 'is_dynamic'])
def test_complex_registry_is_rejected_before_native_delegate_construction(delegates, field):
    fuels = FuelModels()
    fuels.fuel_models_[10][field] = 1+.25j
    with pytest.raises(TypeError, match='Complex|complex'):
        BehaveRun(fuels)
    assert not delegates


def test_complex_route_uses_frozen_registry_and_propagates_imaginary_values(delegates):
    fuels = FuelModels()
    model = BehaveRun(fuels)
    frozen = copy.deepcopy(fuels)
    fuels.fuel_models_[10]['dead_1h'] = 999.
    arguments = _arguments(np.array([1000+.25j, 1000+.5j]))
    from behave_fire.python import BehaveRun as PythonRun
    expected = PythonRun(frozen).do_crown_run(**arguments)
    with pytest.warns(BehaveBackendWarning):
        actual = model.do_crown_run(**arguments)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.any(np.imag(actual['crown_fire_heat_per_unit_area']) != 0)
    assert len(delegates) == 1


def test_real_complex_real_calls_report_actual_backend_without_replaying_rust(delegates):
    model = BehaveRun(FuelModels())
    assert model.preferred_backend == 'rust'
    assert model.do_crown_run(**_arguments(np.array([1000., 1000.]))) == {'native': True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', BehaveBackendWarning)
        model.do_crown_run(**_arguments(np.array([1000+.25j, 1000+.5j])))
        assert model.backend_in_use == 'python'
        model.do_crown_run(**_arguments(np.ma.array([1000+.25j, 1000+.5j], mask=[False, True])))
        assert len(caught) == 1
        assert 'complex' in str(caught[0].message).lower()
        assert 'do_crown_run' in str(caught[0].message)
    assert sum(kind == 'rust' for kind, _ in delegates) == 1
    assert model.do_crown_run(**_arguments(np.array([1000., 1000.]))) == {'native': True}
    assert model.backend_in_use == 'rust'
    assert model.preferred_backend == 'rust'


def test_selected_python_failure_propagates_without_native_replay(delegates, monkeypatch):
    from functools import wraps
    from behave_fire.behave import BehaveRun as PythonRun

    original = PythonRun.do_crown_run
    calls = []

    @wraps(original)
    def fail(*args, **kwargs):
        calls.append('python')
        raise ArithmeticError('Python calculation failed')

    monkeypatch.setattr(PythonRun, 'do_crown_run', fail)
    model = BehaveRun(FuelModels())
    with pytest.warns(BehaveBackendWarning), pytest.raises(ArithmeticError, match='Python calculation failed'):
        model.do_crown_run(**_arguments(np.array([1000+.25j, 1000+.5j])))
    assert calls == ['python']
    assert len(delegates) == 1
    assert model.backend_in_use == 'rust'
