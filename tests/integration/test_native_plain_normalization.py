"""Plain physical normalization avoids mask objects and redundant owned copies."""
import numpy as np
import pytest

pytestmark = pytest.mark.rust


@pytest.mark.parametrize('value', [3., np.float32(.1), np.array([1., -0., np.nan]),
    np.array([1, 2], dtype=np.uint16), np.array([1., 2.], dtype='>f4')[::-1]])
def test_plain_physical_values_never_construct_a_masked_array(monkeypatch, value):
    from behave_fire.rust._numeric import normalize_physical

    expected = np.array(value, dtype=np.float64, copy=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('Plain data constructed a MaskedArray')

    monkeypatch.setattr(np.ma, 'asarray', forbidden)
    result = normalize_physical(value)
    assert type(result) is np.ndarray
    assert result.dtype == np.float64 and result.flags.c_contiguous
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, value)


def test_reuse_is_limited_to_registered_owned_plain_float64_storage():
    from behave_fire._ownership import _owned_inputs
    from behave_fire.rust._numeric import normalize_physical

    original = np.array([1., -0., np.nan])
    with _owned_inputs() as owner:
        owned = owner.snapshot(original)
        result = normalize_physical(owned)
        assert result is owned
        assert not np.shares_memory(result, original)
        original[0] = 100.
        assert result[0] == 1.
        assert normalize_physical(original) is not original
    assert normalize_physical(owned) is not owned


def test_subclasses_masks_and_noncontiguous_storage_keep_safe_normalization():
    from behave_fire._ownership import _owned_inputs
    from behave_fire.rust._numeric import normalize_physical

    class Array(np.ndarray):
        pass

    values = [np.ma.array([1., 2.], mask=[False, True]), np.array([1., 2.]).view(Array),
              np.asfortranarray(np.arange(6.).reshape(2, 3))]
    for value in values:
        with _owned_inputs() as owner:
            owned = owner.snapshot(value)
            result = normalize_physical(owned)
            assert type(result) is np.ndarray and result.dtype == np.float64
            assert result.flags.c_contiguous
            assert not np.shares_memory(result, value)
            if np.ma.isMaskedArray(value):
                assert np.isnan(result[1]) and result[0] == 1.
            else:
                np.testing.assert_array_equal(result, value)
