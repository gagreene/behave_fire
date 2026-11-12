"""Complex routing must preserve the original Python arithmetic and owned inputs."""
import numpy as np
import pytest

from behave_fire.python import surface


@pytest.mark.parametrize('value', [3+.25j, np.array([3+.25j]),
    np.ma.array([3+.25j, 4+.5j], mask=[False, True]), np.array([3+.25j], dtype=object)])
def test_area_elapsed_complex_uses_a_proven_python_counterpart(value):
    from behave_fire._auto_routing import capture_call, choose_backend

    call = capture_call('surface.calculate_fire_area', surface.calculate_fire_area,
                        (2., 1., 2., value), {})
    assert choose_backend(call, 'rust') == 'python'
    actual = call.invoke(surface.calculate_fire_area)
    expected = np.pi * (1.5 * value / 2.) * (1.5 * value)
    np.testing.assert_allclose(np.ma.asarray(actual).compressed().astype(complex),
                               np.ma.asarray(expected).compressed().astype(complex))
    assert np.any(np.imag(np.ma.asarray(actual).compressed().astype(complex)) != 0)


def test_lossy_arguments_and_complex_controls_are_rejected_before_execution():
    from behave_fire._auto_routing import capture_call, choose_backend

    for args, kwargs in [((np.array([2+.25j]), 1., 2., 3.), {}),
                         ((2., 1., 2., 3.), dict(is_crown=1+.25j))]:
        call = capture_call('surface.calculate_fire_area', surface.calculate_fire_area, args, kwargs)
        with pytest.raises(TypeError, match='complex|Complex'):
            choose_backend(call, 'rust')


def test_masked_real_inputs_keep_rust_and_snapshot_data_and_masks():
    from behave_fire._auto_routing import capture_call, choose_backend

    source = np.ma.array(np.array([3., 4.], dtype='>f4')[::-1], mask=[False, True])
    call = capture_call('surface.calculate_fire_area', surface.calculate_fire_area,
                        (2., 1., 2., source), {})
    assert choose_backend(call, 'rust') == 'rust'
    source.data[:] = 99.
    source.mask[:] = False
    captured = call.arguments['elapsed_min']
    np.testing.assert_array_equal(captured.data, [4., 3.])
    np.testing.assert_array_equal(captured.mask, [False, True])
    assert not np.shares_memory(captured.data, source.data)
    assert not np.shares_memory(captured.mask, source.mask)


def test_opaque_array_subclass_does_not_gain_a_complex_fallback():
    from behave_fire._auto_routing import capture_call, choose_backend

    class OpaqueArray(np.ndarray):
        pass

    value = np.array([3+.25j]).view(OpaqueArray)
    call = capture_call('surface.calculate_fire_area', surface.calculate_fire_area, (2., 1., 2., value), {})
    with pytest.raises(TypeError, match='complex|Complex'):
        choose_backend(call, 'rust')


def test_unknown_complex_operation_has_no_guessed_python_fallback():
    from behave_fire._auto_routing import capture_call, choose_backend

    call = capture_call('unknown.operation', surface.calculate_fire_area, (2., 1., 2., 3+.25j), {})
    with pytest.raises(TypeError, match='complex|Complex'):
        choose_backend(call, 'rust')
