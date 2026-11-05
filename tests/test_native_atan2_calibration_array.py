"""Identify the linked atan2 kernel from fixed configuration probes, never user data."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def _candidate_record():
    y = np.array([126., 65534., 2., 1., 3., 127., 65535., 32767.,
                  -126., -65534., -1., 0., -0., 1., -1., 5.], dtype=np.float32)
    x = np.array([-127., -1., 2., 2., 1., -128., -127., -128.,
                  -127., -1., -1., -1., -1., 0., 0., 17.], dtype=np.float32)
    observed = np.arctan2(y, x).view(np.uint32).tolist()
    different = list(observed)
    different[0] ^= 1
    return y.view(np.uint32).tolist(), x.view(np.uint32).tolist(), observed, different


@pytest.fixture
def configuration(monkeypatch):
    from behave_fire.rust import _bridge

    monkeypatch.setattr(_bridge, "_ATAN2_PROFILE_CACHE", None, raising=False)
    monkeypatch.setattr(_bridge, "_ATAN2_PROFILE_LOCK", Lock(), raising=False)
    return _bridge


@pytest.mark.parametrize("target", ["AVX512_SKX", "X86_V4", "baseline(AVX512_SKX)", "baseline(X86_V4)", "unknown-future-target", None])
@pytest.mark.parametrize("selected", [0, 1])
def test_calibration_identifies_linked_kernel_independently_of_target(configuration, monkeypatch, target, selected):
    from numpy._core import _multiarray_umath

    y, x, observed, different = _candidate_record()
    candidates = (y, x, observed, different) if selected == 0 else (y, x, different, observed)
    monkeypatch.setattr(configuration._native, "atan2_profile_candidates", lambda: candidates, raising=False)
    monkeypatch.setattr(_multiarray_umath, "__cpu_targets_info__", {"arctan2": {"fff": {"current": target}}})
    assert configuration._atan2_math_profile() == selected


@pytest.mark.parametrize("x,y", [
    (np.array([-1.], dtype=np.float32), 65534),
    (np.array([-1.], dtype=np.float32), 65534.),
    (np.array([-1], dtype=np.int8), np.array([65534], dtype=np.uint16)),
    (np.array([-1], dtype=np.int8), np.array([False], dtype=bool)),
    (np.array([True], dtype=bool), False),
])
def test_calibration_is_used_for_resolved_float32_direction(configuration, monkeypatch, x, y):
    def unavailable():
        raise RuntimeError('configuration probe unavailable')

    monkeypatch.setattr(configuration._native, "atan2_profile_candidates", unavailable, raising=False)
    with pytest.raises(RuntimeError, match='configuration probe unavailable'):
        configuration._call('surface.calculate_direction_of_max_spread', {
            'x_component': x, 'y_component': y, 'rate_vector': np.ones(1),
            'wind_orientation_mode': 'RelativeToNorth', 'aspect': 95.,
        })


@pytest.mark.parametrize("case", ['float64', 'empty-float32', 'invalid-shape'])
def test_calibration_leaves_unneeded_or_invalid_direction_paths_alone(configuration, monkeypatch, case):
    from behave_fire.components import surface as reference

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError('configuration must not run for this input')

    monkeypatch.setattr(configuration._native, "atan2_profile_candidates", forbidden, raising=False)
    if case == 'float64':
        x, y = np.array([-1.], dtype=np.float64), np.array([65534.], dtype=np.float64)
    elif case == 'empty-float32':
        x, y = np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    else:
        x, y = np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32)
    arguments = dict(x_component=x, y_component=y, rate_vector=np.ones(x.shape), wind_orientation_mode='RelativeToNorth', aspect=95.)
    if case == 'invalid-shape':
        with pytest.raises(ValueError):
            configuration._call('surface.calculate_direction_of_max_spread', arguments)
    else:
        expected = reference.calculate_direction_of_max_spread(**arguments)
        actual = configuration._call('surface.calculate_direction_of_max_spread', arguments)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert not calls


def test_calibration_matches_scalar_without_supported_svml(configuration, monkeypatch):
    from numpy._core import _multiarray_umath

    y, x, observed, _ = _candidate_record()
    monkeypatch.setattr(configuration._native, "atan2_profile_candidates", lambda: (y, x, observed, None), raising=False)
    monkeypatch.setattr(_multiarray_umath, "__cpu_targets_info__", {"arctan2": {"fff": {"current": "AVX512_SKX"}}})
    assert configuration._atan2_math_profile() == 0


@pytest.mark.parametrize("problem", ["record", "length", "boolean", "range", "nonfinite", "unknown", "ambiguous"])
def test_calibration_rejects_malformed_unknown_or_ambiguous_kernels(configuration, monkeypatch, problem):
    y, x, observed, different = _candidate_record()
    record = [y, x, observed, different]
    if problem == "record":
        record = {"unexpected": True}
    elif problem == "length":
        record[0] = y[:-1]
    elif problem == "boolean":
        record[0][0] = True
    elif problem == "range":
        record[0][0] = 2**32
    elif problem == "nonfinite":
        record[0][0] = 0x7f800000
    elif problem == "unknown":
        record[2], record[3] = different, None
    else:
        record[3] = list(observed)
    monkeypatch.setattr(configuration._native, "atan2_profile_candidates", lambda: record, raising=False)
    with pytest.raises(RuntimeError, match="rebuild|reinstall"):
        configuration._atan2_math_profile()


@pytest.mark.parametrize("failure", [False, True])
def test_calibration_runs_once_and_caches_only_immutable_failure_text(configuration, monkeypatch, failure):
    y, x, observed, different = _candidate_record()
    counts = {"native": 0, "numpy": 0}
    original = np.arctan2

    def native_candidates():
        counts["native"] += 1
        return y, x, different if failure else observed, None

    def numpy_probe(first, second):
        counts["numpy"] += 1
        np.testing.assert_array_equal(first.view(np.uint32), y)
        np.testing.assert_array_equal(second.view(np.uint32), x)
        assert first.shape == second.shape == (16,)
        return original(first, second)

    monkeypatch.setattr(configuration._native, "atan2_profile_candidates", native_candidates, raising=False)
    monkeypatch.setattr(np, "arctan2", numpy_probe)
    barrier = Barrier(8)

    def select():
        barrier.wait()
        try:
            return configuration._atan2_math_profile()
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: select(), range(8)))
    assert counts == {"native": 1, "numpy": 1}
    if failure:
        assert all(isinstance(result, RuntimeError) for result in results)
        assert len({id(result) for result in results}) == 8
        assert isinstance(configuration._ATAN2_PROFILE_CACHE[1], str)
    else:
        assert results == [0] * 8
