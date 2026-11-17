"""Probe hostile dimensions only in subprocesses, never inside the pytest process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.rust
_MAX_DIMENSIONS = np._core.multiarray.MAXDIMS
_MAX_ELEMENTS = np.iinfo(np.intp).max // np.dtype(np.float64).itemsize


def _probe_shape(shape, valid, layout='', integer=False, subclass=''):
    from behave_fire import _rust, rust

    script = (
        '\n'
        'import importlib.util\n'
        'from importlib.machinery import ExtensionFileLoader\n'
        'import json\n'
        'import sys\n'
        'import behave_fire\n'
        'behave_fire.__path__.insert(0, sys.argv[2])\n'
        "spec = importlib.util.spec_from_file_location('behave_fire._rust', sys.argv[1], "
        "loader=ExtensionFileLoader('behave_fire._rust', sys.argv[1]))\n"
        'native = importlib.util.module_from_spec(spec)\n'
        'sys.modules[spec.name] = native\n'
        'spec.loader.exec_module(native)\n'
        'import numpy as np\n'
        'from behave_fire.components import surface as reference\n'
        'shape = tuple(json.loads(sys.argv[3]))\n'
        "valid = sys.argv[4] == 'True'\n"
        'layout = sys.argv[5]\n'
        "integer = sys.argv[6] == 'True'\n"
        'subclass = sys.argv[7]\n'
        'calls = []\n'
        'class UntrustedRavel(np.ndarray):\n'
        '    def ravel(self, *args, **kwargs):\n'
        "        calls.append('ravel')\n"
        "        return self if subclass == 'self' else 'malformed flatten result'\n"
        'if valid:\n'
        '    if layout:\n'
        '        values = (np.arange(6, dtype=np.int64 if integer else np.float64) + 2).reshape(shape)\n'
        "        if layout == 'reverse':\n"
        '            values = values[::-1]\n'
        "        elif layout == 'fortran':\n"
        '            values = np.asfortranarray(values)\n'
        '        else:\n'
        '            values.setflags(write=False)\n'
        '    else:\n'
        '        values = np.full(shape, 2., dtype=np.float64)\n'
        '    expected = reference.calculate_eccentricity(values)\n'
        'else:\n'
        '    try:\n'
        '        np.empty(shape, dtype=np.float64)\n'
        '    except ValueError:\n'
        '        pass\n'
        '    else:\n'
        "        raise AssertionError('NumPy accepted a shape classified as invalid')\n"
        '    values = np.empty(0, dtype=np.float64) if 0 in shape else np.array([2.])\n'
        'try:\n'
        '    integers = None\n'
        '    if integer:\n'
        '        bits = values.view(np.uint64)\n'
        '        if subclass:\n'
        '            bits = bits.view(UntrustedRavel)\n'
        "        integers = {'fire_length_to_width_ratio': ('int64', bits)}\n"
        '    elif subclass:\n'
        '        values = values.view(UntrustedRavel)\n'
        '    if subclass:\n'
        '        from behave_fire.rust import surface\n'
        "        result = {'': surface.calculate_eccentricity(values.astype(np.float64) if integer else "
        'values)}\n'
        '    else:\n'
        "        result = native.component_batch('surface.calculate_eccentricity', "
        "{'fire_length_to_width_ratio': values.astype(np.float64) if integer else values}, {}, {}, shape)\n"
        'except ValueError:\n'
        '    if valid:\n'
        '        raise\n'
        'except BaseException as error:\n'
        "    raise AssertionError('non-ValueError shape failure: ' + type(error).__name__) from error\n"
        'else:\n'
        '    if not valid:\n'
        "        raise AssertionError('native accepted an invalid NumPy shape')\n"
        "    assert calls == [], 'native invoked the overridable ndarray.ravel method'\n"
        "    actual = result['']\n"
        '    assert actual.shape == expected.shape == shape\n'
        '    assert actual.dtype == expected.dtype\n'
        '    assert not np.shares_memory(actual, values)\n'
        '    if len(shape) > 32:\n'
        '        assert actual.base is not None\n'
        '    del result, values, integers\n'
        '    import gc\n'
        '    gc.collect()\n'
        '    np.testing.assert_array_equal(actual, expected)\n'
        '    if actual.size:\n'
        '        index = (0,) * len(shape)\n'
        '        original = expected[index]\n'
        '        actual[index] = -7.\n'
        '        assert expected[index] == original\n'
        "print('shape contract passed')\n"
    )
    environment = os.environ.copy()
    root = next(parent for parent in Path(__file__).resolve().parents if (parent/'pytest.ini').is_file())
    environment['PYTHONPATH'] = str(root/'src')
    completed = subprocess.run(
        [sys.executable, '-c', script, _rust.__file__, str(Path(rust.__file__).parent.parent), json.dumps(shape), str(
            valid), layout, str(integer), subclass],
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
