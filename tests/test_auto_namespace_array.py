"""Automatic component calls share the model preflight and truthful routing policy."""
import importlib
import inspect
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import behave_fire as bf
import behave_fire.auto as auto
import behave_fire.python as python_backend


@pytest.fixture
def native_namespace(monkeypatch):
    calls = []

    def area(*args, **kwargs):
        calls.append((args, kwargs))
        return np.array([99.])

    surface = SimpleNamespace(calculate_fire_area=area)
    root = SimpleNamespace(surface=surface)
    original = importlib.import_module

    def load(name):
        if name == 'behave_fire.rust':
            return root
        if name == 'behave_fire.rust.surface':
            return surface
        return original(name)

    monkeypatch.setattr(auto, '_probe', lambda: ('usable', None))
    monkeypatch.setattr(importlib, 'import_module', load)
    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False, raising=False)
    return calls, root


def test_complex_namespace_warning_error_prevents_python_execution(native_namespace, monkeypatch):
    from functools import wraps

    original = python_backend.surface.calculate_fire_area
    calls = []

    @wraps(original)
    def python(*args, **kwargs):
        calls.append('python')
        return original(*args, **kwargs)

    monkeypatch.setattr(python_backend.surface, 'calculate_fire_area', python)
    namespace = bf.get_backend('auto')
    with warnings.catch_warnings():
        warnings.simplefilter('error', bf.BehaveBackendWarning)
        for _ in range(2):
            with pytest.raises(bf.BehaveBackendWarning):
                namespace.surface.calculate_fire_area(2., 1., 2., 3+.25j)
    assert not calls and not native_namespace[0]


def test_explicit_namespaces_remain_explicit(native_namespace):
    assert bf.get_backend('python') is python_backend
    assert bf.get_backend('rust') is native_namespace[1]


def test_namespace_routes_real_complex_real_with_original_signature(native_namespace):
    namespace = bf.get_backend('auto')
    function = namespace.surface.calculate_fire_area
    assert inspect.signature(function) == inspect.signature(python_backend.surface.calculate_fire_area)
    np.testing.assert_array_equal(function(2., 1., 2., 3.), [99.])
    assert namespace.backend_in_use == 'rust'
    with pytest.warns(bf.BehaveBackendWarning, match='[Cc]omplex'):
        actual = function(2., 1., 2., np.array([3+.25j]))
    np.testing.assert_allclose(actual, python_backend.surface.calculate_fire_area(2., 1., 2., np.array([3+.25j])))
    assert namespace.backend_in_use == 'python'
    assert namespace.preferred_backend == 'rust'
    assert len(native_namespace[0]) == 1
    np.testing.assert_array_equal(function(2., 1., 2., 3.), [99.])
    assert namespace.backend_in_use == 'rust'
    assert len(native_namespace[0]) == 2


@pytest.mark.rust
def test_unmigrated_direction_selection_keeps_original_strided_layout(monkeypatch):
    from behave_fire.rust import _bridge

    monkeypatch.setattr(auto, '_probe', lambda: ('usable', None))
    monkeypatch.setattr(_bridge, '_atan2_math_profile', lambda: 1)
    selected = []

    def detached(operation, arguments, direction_profile):
        selected.append(direction_profile)
        return np.zeros(3)

    monkeypatch.setattr(_bridge, '_call_owned', detached)
    inputs = dict(x_component=np.array([1., 2., 3.], np.float32)[::-1],
                  y_component=np.array([3., 2., 1.], np.float32)[::-1],
                  rate_vector=np.ones(3, np.float32), wind_orientation_mode='RelativeToUpslope', aspect=0.)
    normalized, single = _bridge._normalize_direction(inputs)
    assert single and _bridge._direction_math_profile(normalized) == 0
    bf.get_backend('auto').surface.calculate_direction_of_max_spread(**inputs)
    assert selected == [0]


def test_unsupported_complex_argument_is_not_replayed_on_python(native_namespace):
    namespace = bf.get_backend('auto')
    with pytest.raises(TypeError, match='complex|Complex'):
        namespace.surface.calculate_fire_area(np.array([2+.25j]), 1., 2., 3.)
    assert not native_namespace[0]
