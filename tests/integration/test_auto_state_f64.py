"""Auto state migration preflights, snapshots complete state, and pins Python."""
import copy
import pickle
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def test_complex_fire_size_mutation_moves_complete_state_once_and_stays_python(monkeypatch):
    from behave_fire import auto
    from behave_fire.components.fire_size import FireSize as PythonFireSize

    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False)
    namespace = auto.get_backend('auto')
    state = namespace.FireSize()
    state.calculate_fire_basic_dimensions(False, 5., 5, 10., 0)
    state.eccentricity_ = 0.5 + 0.1j
    state.elapsed_time_ = 23.
    state.elliptical_a_ = 42.
    expected = PythonFireSize()
    for name in ('effective_wind_speed_', 'forward_spread_rate_', 'elapsed_time_', 'elliptical_a_',
                 'elliptical_b_', 'elliptical_c_', 'eccentricity_', 'backing_spread_rate_',
                 'flanking_spread_rate_', 'fire_length_to_width_ratio_', 'heading_to_backing_ratio_'):
        setattr(expected, name, copy.deepcopy(getattr(state, name)))
    expected.calculate_backing_spread_rate()
    with pytest.warns(auto.BehaveBackendWarning, match='[Cc]omplex'):
        state.calculate_backing_spread_rate()
    assert state.backend_in_use == namespace.backend_in_use == 'python'
    assert state.requested_backend == 'auto'
    assert state.backing_spread_rate_ == expected.backing_spread_rate_
    for name, value in vars(expected).items():
        np.testing.assert_equal(getattr(state, name), value)
    assert state.elapsed_time_ == 23. and state.elliptical_a_ == 42.
    restored = pickle.loads(pickle.dumps(state))
    assert restored.backend_in_use == 'python'
    restored.eccentricity_ = 0.25
    restored.calculate_backing_spread_rate()
    assert restored.backend_in_use == 'python'


def test_complex_state_warning_error_preserves_delegate_values_and_warning_availability(monkeypatch):
    from behave_fire import auto

    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False)
    namespace = auto.get_backend('auto')
    state = namespace.FireSize()
    state.eccentricity_ = 0.5 + 0.1j
    before = copy.deepcopy(vars(state._delegate))
    with warnings.catch_warnings():
        warnings.simplefilter('error', auto.BehaveBackendWarning)
        with pytest.raises(auto.BehaveBackendWarning):
            state.calculate_backing_spread_rate()
    assert state.backend_in_use == namespace.backend_in_use == 'rust'
    assert vars(state._delegate) == before
    assert not auto._COMPLEX_WARNING_EMITTED
    with pytest.warns(auto.BehaveBackendWarning):
        state.calculate_backing_spread_rate()
    assert state.backend_in_use == 'python'


def test_unverified_or_lossy_complex_state_method_rejects_without_switch(monkeypatch):
    from behave_fire import auto

    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False)
    state = auto.get_backend('auto').FireSize()
    with pytest.raises(TypeError, match='complex'):
        state.get_fire_area(False, 0, 1. + 2j, 0)
    assert state.backend_in_use == 'rust'
