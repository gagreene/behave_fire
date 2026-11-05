"""Probe hostile dimensions only in subprocesses, never inside the pytest process."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.rust
_MAX_DIMENSIONS = np._core.multiarray.MAXDIMS
_MAX_ELEMENTS = np.iinfo(np.intp).max // np.dtype(np.float64).itemsize


def _probe_shape(shape, valid, layout='', integer=False, subclass=''):
    from behave_fire import _rust, rust

    script = '''
import importlib.util
from importlib.machinery import ExtensionFileLoader
import json
import sys
import behave_fire
behave_fire.__path__.insert(0, sys.argv[2])
spec = importlib.util.spec_from_file_location('behave_fire._rust', sys.argv[1], loader=ExtensionFileLoader('behave_fire._rust', sys.argv[1]))
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
import numpy as np
from behave_fire.components import surface as reference
shape = tuple(json.loads(sys.argv[3]))
valid = sys.argv[4] == 'True'
layout = sys.argv[5]
integer = sys.argv[6] == 'True'
subclass = sys.argv[7]
calls = []
class UntrustedRavel(np.ndarray):
    def ravel(self, *args, **kwargs):
        calls.append('ravel')
        return self if subclass == 'self' else 'malformed flatten result'
if valid:
    if layout:
        values = (np.arange(6, dtype=np.int64 if integer else np.float64) + 2).reshape(shape)
        if layout == 'reverse':
            values = values[::-1]
        elif layout == 'fortran':
            values = np.asfortranarray(values)
        else:
            values.setflags(write=False)
    else:
        values = np.full(shape, 2., dtype=np.float64)
    expected = reference.calculate_eccentricity(values)
else:
    try:
        np.empty(shape, dtype=np.float64)
    except ValueError:
        pass
    else:
        raise AssertionError('NumPy accepted a shape classified as invalid')
    values = np.empty(0, dtype=np.float64) if 0 in shape else np.array([2.])
try:
    integers = None
    if integer:
        bits = values.view(np.uint64)
        if subclass:
            bits = bits.view(UntrustedRavel)
        integers = {'fire_length_to_width_ratio': ('int64', bits)}
    elif subclass:
        values = values.view(UntrustedRavel)
    result = native.batch('surface.calculate_eccentricity', {'fire_length_to_width_ratio': values.astype(np.float64) if integer else values}, {}, shape, False, integers)
except ValueError:
    if valid:
        raise
except BaseException as error:
    raise AssertionError('non-ValueError shape failure: ' + type(error).__name__) from error
else:
    if not valid:
        raise AssertionError('native accepted an invalid NumPy shape')
    assert calls == [], 'native invoked the overridable ndarray.ravel method'
    actual = result['']
    assert actual.shape == expected.shape == shape
    assert actual.dtype == expected.dtype
    assert not np.shares_memory(actual, values)
    if len(shape) > 32:
        assert actual.base is not None
    del result, values, integers
    import gc
    gc.collect()
    np.testing.assert_array_equal(actual, expected)
    if actual.size:
        index = (0,) * len(shape)
        original = expected[index]
        actual[index] = -7.
        assert expected[index] == original
print('shape contract passed')
'''
    environment = os.environ.copy()
    root = next(parent for parent in Path(__file__).resolve().parents if (parent/'pytest.ini').is_file())
    environment['PYTHONPATH'] = str(root/'src')
    completed = subprocess.run(
        [sys.executable, '-c', script, _rust.__file__, str(Path(rust.__file__).parent.parent), json.dumps(shape), str(valid), layout, str(integer), subclass],
        capture_output=True, text=True, env=environment, timeout=30,
    )
    assert completed.returncode == 0, f'exit={completed.returncode}\n{completed.stdout}\n{completed.stderr}'
    assert completed.stdout.strip() == 'shape contract passed'
    assert completed.stderr == ''


@pytest.mark.parametrize('shape', [
    [0, 2**62], [2**62, 0], [0, 2**63], [0, 2**32, 2**32],
    [0, _MAX_ELEMENTS + 1], [0] + [1]*_MAX_DIMENSIONS, [1]*(_MAX_DIMENSIONS + 1),
])
def test_invalid_shape_extents_raise_value_error_in_subprocess(shape):
    _probe_shape(shape, False)


@pytest.mark.parametrize('layout', ['reverse', 'fortran', 'readonly'])
@pytest.mark.parametrize('integer', [False, True])
@pytest.mark.parametrize('rank', [33, _MAX_DIMENSIONS])
def test_maximum_rank_input_layouts_preserve_row_major_values_in_subprocess(layout, integer, rank):
    _probe_shape([2, 3] + [1]*(rank - 2), True, layout, integer)


@pytest.mark.parametrize('subclass', ['self', 'malformed'])
@pytest.mark.parametrize('integer', [False, True])
@pytest.mark.parametrize('rank', [33, _MAX_DIMENSIONS])
def test_subclass_ravel_is_bypassed_in_subprocess(subclass, integer, rank):
    _probe_shape([2, 3] + [1]*(rank - 2), True, 'reverse', integer, subclass)


@pytest.mark.parametrize('shape', [
    [], [0], [2, 0, 3], [0, _MAX_ELEMENTS], [_MAX_ELEMENTS, 0],
    [0] + [1]*32, [1]*33, [0] + [1]*(_MAX_DIMENSIONS - 1), [1]*_MAX_DIMENSIONS,
])
def test_valid_empty_and_dynamic_shapes_match_reference_in_subprocess(shape):
    _probe_shape(shape, True)
