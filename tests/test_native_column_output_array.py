"""API-3 output buffers preserve typed payloads without expanded metadata columns."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def test_column_wire_keeps_float32_and_mask_schema_compact():
    from behave_fire import _rust

    result = _rust.batch("surface.calculate_fire_length", {
        "forward_ros": np.array([1.5, 2.0]), "backing_ros": np.array([0.5, 1.0]),
        "elapsed_min": np.array([3.0, 4.0]),
        "_single.forward_ros": np.array([1.0]), "_single.backing_ros": np.array([1.0]),
        "_single.elapsed_min": np.array([1.0]), "_masked.elapsed_min": np.array([1.0]),
        "_mask.elapsed_min": np.array([False, False], dtype=float),
        "_fill.elapsed_min": np.array([-17.0]),
    }, {}, (2,), True)
    assert _rust.__binding_api_version__ == 4
    assert set(result) == {"", "__schema"}
    assert result[""].dtype == np.float32
    schema = result["__schema"][""]
    assert schema["masked"] is True
    assert isinstance(schema["single"], bool)
    assert schema["mask"].dtype == bool
    assert schema["mask"].tolist() == [False, False]


def test_column_wire_keeps_integer_classification_as_int32():
    from behave_fire import _rust

    result = _rust.batch("crown.classify_crown_fire_type", {
        "transition_ratio": np.array([0.0, 2.0]), "active_ratio": np.array([0.0, 2.0]),
    }, {}, (2,), False)
    assert result["fire_type"].dtype == np.int32
    assert result["is_surface_fire"].dtype == bool
    np.testing.assert_array_equal(result["fire_type"], [0, 3])
