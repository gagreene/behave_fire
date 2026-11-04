"""Keep exact NumPy integer scalar aliases through object arithmetic and selection."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust
_CLASSES = (np.intc, np.uintc, np.int32, np.uint32, np.int64, np.uint64, np.longlong, np.ulonglong, int)


def _assert_boxed(actual, expected):
    assert actual.dtype == expected.dtype == object
    assert actual.shape == expected.shape
    for left, right in zip(actual.ravel(), expected.ravel()):
        assert type(left) is type(right)
        if isinstance(right, np.generic):
            assert left.dtype.char == right.dtype.char
            assert left.dtype.num == right.dtype.num
        assert left == right


@pytest.mark.parametrize('left_class', _CLASSES)
@pytest.mark.parametrize('right_class', _CLASSES)
def test_boxed_integer_alias_mixed_arithmetic(left_class, right_class):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    left = np.array([left_class(2)], dtype=object)
    right = np.array([right_class(3)], dtype=object)
    expected = reference.calculate_fire_length(left, right, 1)
    actual = surface.calculate_fire_length(left, right, 1)
    _assert_boxed(actual, expected)


@pytest.mark.parametrize('left_class', _CLASSES)
@pytest.mark.parametrize('right_class', _CLASSES)
@pytest.mark.parametrize('choose_surface', [False, True])
def test_boxed_integer_alias_selection_retains_chosen_class(left_class, right_class, choose_surface):
    from behave_fire.components import crown as reference
    from behave_fire.rust import crown

    left = np.array([left_class(2)], dtype=object)
    right = np.array([right_class(3)], dtype=object)
    arguments = (choose_surface, True, left, 100., 200., right, 100., 200., 1., right, 100., 200., 1.)
    expected = reference.assign_final_crown_fire_behavior(*arguments)['final_spread_rate']
    actual = crown.assign_final_crown_fire_behavior(*arguments)['final_spread_rate']
    _assert_boxed(actual, expected)
