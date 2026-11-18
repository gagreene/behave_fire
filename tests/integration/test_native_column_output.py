"""Primitive wire storage uses f64, exact codes, and no dtype sidecars."""
import numpy as np
import pytest

pytestmark = pytest.mark.rust


def test_column_wire_rejects_retired_numeric_sidecars():
    from behave_fire import _rust

    with pytest.raises(ValueError, match='metadata'):
        _rust.component_batch('surface.calculate_fire_length', {
            'forward_ros': np.array([1.]), 'backing_ros': np.array([0.]), 'elapsed_min': np.array([1.]),
            '_single.forward_ros': np.array([1.])}, {}, {}, [1])


def test_column_wire_returns_plain_float64_without_schema_metadata():
    from behave_fire import _rust

    result = _rust.component_batch('surface.calculate_fire_length', {
        'forward_ros': np.array([1.5, 2.]), 'backing_ros': np.array([0.5, 1.]),
        'elapsed_min': np.array([3., 4.])}, {}, {}, [2])
    assert set(result) == {''}
    assert result[''].dtype == np.float64
    np.testing.assert_array_equal(result[''], [6., 12.])


def test_column_wire_stores_classification_as_int32_and_flags_as_bool():
    from behave_fire import _rust

    result = _rust.component_batch('crown.classify_crown_fire_type', {
        'transition_ratio': np.array([0., 2., np.nan]), 'active_ratio': np.array([0., 2., 1.])}, {}, {}, [3])
    assert result['fire_type'].dtype == np.int32
    assert result['is_surface_fire'].dtype == bool
    np.testing.assert_array_equal(result['fire_type'], [0, 3, -1])
