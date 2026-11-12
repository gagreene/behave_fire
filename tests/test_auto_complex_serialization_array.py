"""Mixed-backend model state survives copying and pickle without reselection."""
import copy
import pickle
import warnings

import numpy as np
import pytest

from behave_fire import BehaveBackendWarning, BehaveRun, FuelModels
from parity_support import ROOT

pytestmark = pytest.mark.rust


def _arguments(heat):
    return dict(surface_results=dict(spread_rate=np.full(2, 2.), fireline_intensity=np.full(2, 200.),
        heat_per_unit_area=heat), fuel_model_grid=np.full(2, 10, np.int32),
        m1h=np.full(2, .06), m10h=np.full(2, .07), m100h=np.full(2, .08),
        mlh=np.full(2, .6), mlw=np.full(2, .9), wind_speed=np.full(2, 8.),
        wind_speed_units=0, wind_direction=35., wind_orientation_mode='RelativeToUpslope',
        slope=0., canopy_base_height=4., canopy_height=30., canopy_bulk_density=.01)


@pytest.mark.parametrize('method', ['copy', 'deepcopy', 'pickle'])
def test_complex_then_real_state_roundtrip_keeps_policy_and_frozen_registry(monkeypatch, method):
    import behave_fire.auto as auto
    from behave_fire.python import BehaveRun as PythonRun
    from float64_model_reference import model_reference
    from performance_cases import assert_outputs

    monkeypatch.setattr(auto, '_COMPLEX_WARNING_EMITTED', False)
    registry = FuelModels()
    frozen = copy.deepcopy(registry)
    model = BehaveRun(registry, backend='auto', threads=2)
    registry.fuel_models_[10]['dead_1h'] = 999.
    complex_arguments = _arguments(np.array([1000.+.25j, 1000.+.5j]))
    with pytest.warns(BehaveBackendWarning):
        original = model.do_crown_run(**complex_arguments)
    assert np.any(np.imag(original['crown_fire_heat_per_unit_area']) != 0.)

    def forbid_probe():
        raise AssertionError('State restoration reselected a backend')

    monkeypatch.setattr(auto, '_probe', forbid_probe)
    restored = pickle.loads(pickle.dumps(model)) if method == 'pickle' else getattr(copy, method)(model)
    assert restored.preferred_backend == 'rust'
    assert restored.backend_in_use == 'python'
    assert restored.__getstate__()['requested'] == 'auto'
    assert restored._delegate is restored._python_delegate
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', BehaveBackendWarning)
        repeated = restored.do_crown_run(**complex_arguments)
    assert not caught
    for field in original:
        np.testing.assert_allclose(repeated[field], original[field])
    real_arguments = _arguments(np.full(2, 1000.))
    expected = model_reference(PythonRun(frozen).do_crown_run, real_arguments)
    assert_outputs(restored.do_crown_run(**real_arguments), expected)
    assert restored.backend_in_use == restored.preferred_backend == 'rust'
    assert restored._delegate is restored._rust_delegate
