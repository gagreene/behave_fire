"""Continuously varying diagnostics must not reuse the acceptance weather cycle."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _cases():
    spec = importlib.util.spec_from_file_location("diagnostic_cases", Path(__file__).resolve().parents[2]
        / "tools/performance_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- seeded weather generation tests ----

def test_continuous_weather_is_unique_reproducible_and_seeded():
    cases = _cases()
    first = cases.build_inputs(1001, "continuous", seed=7)
    second = cases.build_inputs(1001, "continuous", seed=7)
    other = cases.build_inputs(1001, "continuous", seed=8)
    for name in ("m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed", "wind_direction", "slope"):
        assert np.unique(first[name]).size == 1001
        np.testing.assert_array_equal(first[name], second[name])
        assert not np.array_equal(first[name], other[name])
        assert np.isfinite(first[name]).all()
    assert set(first["fuel_model_grid"]) == {0, 1, 4, 10, 101, 124, 145, 201}


@pytest.mark.parametrize("cells", [0, 1])
def test_continuous_weather_supports_empty_and_one_cell_vectors(cells):
    inputs = _cases().build_inputs(cells, "continuous")
    assert set(inputs) == {'fuel_model_grid', 'm1h', 'm10h', 'm100h', 'mlh', 'mlw', 'wind_speed',
                           'wind_speed_units', 'wind_direction', 'wind_orientation_mode', 'slope'}
    for name in ('m1h', 'm10h', 'm100h', 'mlh', 'mlw', 'wind_speed', 'wind_direction', 'slope'):
        assert inputs[name].shape == (cells,), name
        assert inputs[name].dtype == np.float64, name
        assert np.isfinite(inputs[name]).all(), name
    assert inputs['fuel_model_grid'].shape == (cells,)
    assert inputs['fuel_model_grid'].dtype == np.int32
    assert inputs['fuel_model_grid'].tolist() == ([] if cells == 0 else [1])
    assert inputs['wind_speed_units'] == 0
    assert inputs['wind_orientation_mode'] == 'RelativeToUpslope'
