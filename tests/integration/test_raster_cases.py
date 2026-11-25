"""Shared raster cases are reproducible, typed as declared, and compared exactly."""

import numpy as np
import pytest

from parity_support import ROOT as ROOT

# ---- raster case tests ----

def test_continuous_weather_is_unique_reproducible_and_seeded():
    from raster_cases import build_inputs

    first = build_inputs(1001, "continuous", seed=7)
    second = build_inputs(1001, "continuous", seed=7)
    other = build_inputs(1001, "continuous", seed=8)
    for name in ("m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed", "wind_direction", "slope"):
        assert np.unique(first[name]).size == 1001
        np.testing.assert_array_equal(first[name], second[name])
        assert not np.array_equal(first[name], other[name])
        assert np.isfinite(first[name]).all()
    assert set(first["fuel_model_grid"]) == {0, 1, 4, 10, 101, 124, 145, 201}


@pytest.mark.parametrize("cells", [0, 1])
def test_continuous_weather_supports_empty_and_one_cell_vectors(cells):
    from raster_cases import build_inputs

    inputs = build_inputs(cells, "continuous")
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


def test_mixed_case_layouts_are_distinct_and_reproducible():
    from behave_fire.python import BehaveRun, FuelModels
    from raster_cases import build_inputs, supplied_surface

    inputs = build_inputs(16, "heterogeneous", 20261105)
    repeat = build_inputs(16, "heterogeneous", 20261105)
    np.testing.assert_array_equal(inputs["fuel_model_grid"], [1, 4, 10, 101, 124, 145, 201, 0] * 2)
    for field in ("m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed", "slope"):
        np.testing.assert_array_equal(inputs[field], repeat[field])
    model = BehaveRun(FuelModels())
    first = supplied_surface(inputs, "typed-crown", model)
    second = supplied_surface(inputs, "typed-crown-b", model)
    assert first["spread_rate"].dtype == np.float32
    assert first["fireline_intensity"].dtype == np.uint16
    np.testing.assert_array_equal(first["heat_per_unit_area"].mask, np.arange(16) % 7 == 0)
    assert second["heat_per_unit_area"].dtype == np.float32
    assert second["fireline_intensity"].dtype == np.int32
    np.testing.assert_array_equal(second["spread_rate"].mask, np.arange(16) % 5 == 0)


@pytest.mark.rust
@pytest.mark.parametrize("threads", [1, 4])
def test_mixed_layout_validation_compares_complete_native_outputs(threads):
    from raster_cases import validate_layout

    result = validate_layout("typed-crown-b", "heterogeneous", 16, threads=threads)
    assert result["passed"] is True, result["error"]
    assert result["output_fields"] == 22 == len(result["fields"])
    assert result["native_boundary_calls"] == ["model_batch"]


def test_output_comparator_keeps_masked_raw_data_and_zero_signs():
    from raster_cases import assert_outputs

    expected = np.ma.array([-0.0, 5.0], mask=[True, False])
    actual = expected.copy()
    actual.data[0] = 0.0
    with pytest.raises(AssertionError):
        assert_outputs(actual, expected)
    with pytest.raises(AssertionError):
        assert_outputs(np.array([2**60], dtype=np.uint64), np.array([2**60 + 1], dtype=np.uint64))
