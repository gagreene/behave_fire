"""Probe hostile dimensions only in subprocesses, never inside the pytest process."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.rust
_MAX_DIMENSIONS = np._core.multiarray.MAXDIMS
_MAX_ELEMENTS = np.iinfo(np.intp).max // np.dtype(np.float64).itemsize
# Preserve the original test IDs for retained subprocess cases.
_PREFIX = 'tests/test_native_shape_safety_array.py::'
_INVALID_SHAPES = [
    [0, 2**62], [2**62, 0], [0, 2**63], [0, 2**32, 2**32],
    [0, _MAX_ELEMENTS + 1], [0] + [1]*_MAX_DIMENSIONS, [1]*(_MAX_DIMENSIONS + 1),
]
_VALID_SHAPES = [
    [], [0], [2, 0, 3], [0, _MAX_ELEMENTS], [_MAX_ELEMENTS, 0],
    [0] + [1]*32, [1]*33, [0] + [1]*(_MAX_DIMENSIONS - 1), [1]*_MAX_DIMENSIONS,
]


def _assert_probe(completed, cases):
    """Require exact ordered completion, individual input families, and clean stderr."""
    last_started = '<none>'
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get('event') == 'start':
            last_started = event.get('id', '<missing id>')
    detail = (f'exit={completed.returncode}; last_started={last_started}\n'
              f'{completed.stdout}\n{completed.stderr}')
    assert completed.returncode == 0, detail
    assert completed.stderr == '', detail
    try:
        actual = [json.loads(line) for line in completed.stdout.splitlines()]
    except ValueError as error:
        raise AssertionError('Malformed case transcript: ' + detail) from error
    expected = []
    for case in cases:
        kind = case.get('kind', 'float')
        expected.append(dict(event='start', id=case['id']))
        expected.append(dict(event='passed', id=case['id'], warnings=case.get('warnings', []),
                             input_dtype='int64' if kind == 'integer' else 'float64',
                             input_subclass=bool(case.get('subclass')) and kind != 'legacy_integer_cast',
                             input_profile=_expected_profile(case),
                             outcome='computed' if case['valid'] else 'value_error'))
    assert actual == expected, f'Incomplete or incorrect case transcript: {detail}\nexpected={expected!r}'


def _case(identity, shape, valid=True, layout='', kind='float', route='native', subclass=''):
    """Declare a named case without changing its caller-side physical input."""
    return dict(id=identity, shape=shape, valid=valid, layout=layout, kind=kind, route=route,
                subclass=subclass, warnings=[])


def _expected_profile(case):
    """Pin the distinct effective legacy layouts and genuine public integer inputs."""
    shape = case['shape'] if case['valid'] else [0 if 0 in case['shape'] else 1]
    layout = case.get('layout', '')
    cast = case.get('kind') == 'legacy_integer_cast'
    if layout:
        values = [5, 6, 7, 2, 3, 4] if layout == 'reverse' else [2, 3, 4, 5, 6, 7]
        negative = layout == 'reverse' and not cast
        return dict(shape=shape, values=values, writeable=layout != 'readonly' or cast,
                    c_contiguous=layout != 'fortran' and not negative,
                    f_contiguous=layout == 'fortran', negative_strides=negative)
    return dict(shape=shape, values=[2.] * math.prod(shape), writeable=True,
                c_contiguous=True, f_contiguous=True, negative_strides=False)


def _layout_cases(rank):
    """Retain all six effective legacy layouts and add three genuine integer routes."""
    shape = [2, 3] + [1]*(rank - 2)
    cases = []
    for integer in (False, True):
        for layout in ('reverse', 'fortran', 'readonly'):
            identity = _PREFIX + ('test_maximum_rank_input_layouts_preserve_row_major_values_in_subprocess'
                                  f'[{rank}-{integer}-{layout}]')
            cases.append(_case(identity, shape, layout=layout, kind='legacy_integer_cast' if integer else 'float'))
    cases.extend(_case(f'integer-layout/{rank}/{layout}', shape, layout=layout, kind='integer', route='public')
                 for layout in ('reverse', 'fortran', 'readonly'))
    return cases


def _probe_shapes(cases):
    """Use one child for compatible cases while isolating every invalid shape."""
    from behave_fire import _rust, rust

    assert cases and len({case['id'] for case in cases}) == len(cases)
    assert len(cases) == 1 or all(case['valid'] for case in cases), 'Invalid shapes require a fresh process per case'
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / 'pytest.ini').is_file())
    environment = dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONOPTIMIZE='0')
    command = [sys.executable, '-B', str(root / 'tests/shape_safety_probe.py'),
               _rust.__file__, str(Path(rust.__file__).parent.parent), json.dumps(cases)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                   errors='backslashreplace', env=environment, timeout=30)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ''
        stderr = error.stderr or ''
        if isinstance(stdout, bytes):
            stdout = stdout.decode('utf-8', errors='backslashreplace')
        if isinstance(stderr, bytes):
            stderr = stderr.decode('utf-8', errors='backslashreplace')
        try:
            _assert_probe(SimpleNamespace(returncode=-1, stdout=stdout, stderr=stderr), cases)
        except AssertionError as failure:
            raise AssertionError(f'Shape child timed out: {failure}') from error
        raise AssertionError('Shape child timed out') from error
    _assert_probe(completed, cases)


def _subclass_cases(rank):
    """Retain all four effective legacy inputs and add actual integer subclasses."""
    shape = [2, 3] + [1]*(rank - 2)
    cases = []
    for integer in (False, True):
        for subclass in ('self', 'malformed'):
            identity = _PREFIX + f'test_subclass_ravel_is_bypassed_in_subprocess[{rank}-{integer}-{subclass}]'
            cases.append(_case(identity, shape, layout='reverse', kind='legacy_integer_cast' if integer else 'float',
                               route='public', subclass=subclass))
    cases.extend(_case(f'integer-subclass/{rank}/{subclass}', shape, layout='reverse', kind='integer',
                       route='public', subclass=subclass) for subclass in ('self', 'malformed'))
    return cases


# ---- subprocess shape validation and diagnostics tests ----

def test_child_assertions_survive_inherited_optimization(monkeypatch):
    original_run = subprocess.run
    monkeypatch.setenv('PYTHONOPTIMIZE', '1')
    wrapper = '''import runpy, sys
import numpy as np
namespace = runpy.run_path(sys.argv[1], run_name="shape_dtype_mutation")
original_loader = namespace["_load_native"]
def wrong_loader(extension, package):
    native, public, reference = original_loader(extension, package)
    original = native.component_batch
    def wrong_dtype(*args, **kwargs):
        result = original(*args, **kwargs)
        result[""] = result[""].astype(np.complex128)
        return result
    native.component_batch = wrong_dtype
    return native, public, reference
namespace["main"].__globals__["_load_native"] = wrong_loader
namespace["main"](sys.argv[2:])
'''

    def mutated_child(command, **options):
        return original_run([command[0], '-B', '-c', wrapper, command[2], *command[3:]], **options)

    monkeypatch.setattr(subprocess, 'run', mutated_child)
    with pytest.raises(AssertionError, match='result dtype differs from reference'):
        _probe_shapes([_case('wrong-result-dtype', [2])])


def test_compatible_valid_cases_share_one_child_process(monkeypatch):
    original = subprocess.run
    launches = []

    def observe(*arguments, **options):
        launches.append(arguments[0])
        return original(*arguments, **options)

    monkeypatch.setattr(subprocess, 'run', observe)
    _probe_shapes([_case('scalar', []), _case('two-cells', [2])])
    assert len(launches) == 1


def test_crash_bytes_preserve_exact_last_started_case(monkeypatch):
    original_run = subprocess.run
    script = 'import os, sys; os.write(1, b\'{"event":"start","id":"second"}\\n\\x81\'); sys.exit(3)'

    def crashing_child(command, **options):
        return original_run([command[0], '-B', '-c', script], **options)

    monkeypatch.setattr(subprocess, 'run', crashing_child)
    with pytest.raises(AssertionError, match='last_started=second') as caught:
        _probe_shapes([_case('first', []), _case('second', [2])])
    assert str(caught.value).splitlines()[0] == 'exit=3; last_started=second'


@pytest.mark.parametrize('shape', _INVALID_SHAPES)
def test_invalid_shape_extents_raise_value_error_in_subprocess(shape):
    index = _INVALID_SHAPES.index(shape)
    _probe_shapes([_case(_PREFIX + f'test_invalid_shape_extents_raise_value_error_in_subprocess[shape{index}]',
                         shape, valid=False)])


@pytest.mark.parametrize('rank', [33, _MAX_DIMENSIONS])
def test_maximum_rank_input_layouts_preserve_row_major_values_in_subprocess(rank):
    _probe_shapes(_layout_cases(rank))


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'order', 'json', 'stderr', 'exit', 'timeout'])
def test_probe_protocol_rejects_incomplete_or_failed_children(monkeypatch, fault):
    cases = [_case('first', []), _case('second', [2])]
    events = [dict(event='start', id='first'),
              dict(event='passed', id='first', warnings=[], input_dtype='float64', input_subclass=False,
                   input_profile=dict(shape=[], values=[2.], writeable=True, c_contiguous=True,
                                      f_contiguous=True, negative_strides=False),
                   outcome='computed'),
              dict(event='start', id='second'),
              dict(event='passed', id='second', warnings=[], input_dtype='float64', input_subclass=False,
                   input_profile=dict(shape=[2], values=[2., 2.], writeable=True, c_contiguous=True,
                                      f_contiguous=True, negative_strides=False),
                   outcome='computed')]
    if fault == 'missing':
        events.pop()
    elif fault == 'duplicate':
        events.append(events[-1])
    elif fault == 'order':
        events = events[2:] + events[:2]
    elif fault == 'exit':
        events.pop()
    stdout = '\n'.join(json.dumps(event) for event in events)
    if fault == 'json':
        stdout = 'not-json'
    completed = SimpleNamespace(returncode=-11 if fault == 'exit' else 0, stdout=stdout,
                                stderr='unexpected warning' if fault == 'stderr' else '')

    def broken_child(command, **options):
        if fault == 'timeout':
            raise subprocess.TimeoutExpired(command, 30, output=json.dumps(events[2]).encode(), stderr=b'')
        return completed

    monkeypatch.setattr(subprocess, 'run', broken_child)
    with pytest.raises(AssertionError) as caught:
        _probe_shapes(cases)
    assert 'last_started=' in str(caught.value)
    if fault == 'exit':
        assert str(caught.value).splitlines()[0] == 'exit=-11; last_started=second'
    if fault == 'timeout':
        assert str(caught.value).splitlines()[0] == 'Shape child timed out: exit=-1; last_started=second'


@pytest.mark.parametrize('rank', [33, _MAX_DIMENSIONS])
def test_subclass_ravel_is_bypassed_in_subprocess(rank):
    _probe_shapes(_subclass_cases(rank))


def test_valid_empty_and_dynamic_shapes_match_reference_in_subprocess():
    _probe_shapes([_case(_PREFIX + f'test_valid_empty_and_dynamic_shapes_match_reference_in_subprocess[shape{index}]',
                         shape) for index, shape in enumerate(_VALID_SHAPES)])
