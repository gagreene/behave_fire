"""Execute named native shape cases only inside an isolated child interpreter."""

import gc
import importlib.util
import json
import sys
import warnings
from importlib.machinery import ExtensionFileLoader

import numpy as np


def _input_profile(values):
    """Record the actual caller layout and row-major values without subclass callbacks."""
    plain = np.asarray(values)
    return dict(shape=list(values.shape), values=plain.reshape(-1).tolist(),
                writeable=bool(values.flags.writeable), c_contiguous=bool(values.flags.c_contiguous),
                f_contiguous=bool(values.flags.f_contiguous), negative_strides=any(step < 0 for step in values.strides))


def _load_native(extension, package):
    """Load the requested binary and its public/reference modules."""
    import behave_fire

    behave_fire.__path__.insert(0, package)
    spec = importlib.util.spec_from_file_location(
        'behave_fire._rust', extension, loader=ExtensionFileLoader('behave_fire._rust', extension))
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as public

    return native, public, reference


def _probe_case(case, native, public, reference):
    """Check one fresh input, error domain, numeric result, and owned-output lifetime."""
    shape = tuple(case['shape'])
    valid = case['valid']
    calls = []
    subclass = case.get('subclass', '')
    kind = case.get('kind', 'float')

    class UntrustedRavel(np.ndarray):
        def ravel(self, *args, **kwargs):
            """Record any forbidden dispatch through the overridable method."""
            calls.append('ravel')
            return self if subclass == 'self' else 'malformed flatten result'

    if valid:
        values = _source_values(shape, case.get('layout', ''), kind != 'float')
        expected = reference.calculate_eccentricity(values)
    else:
        try:
            np.empty(shape, dtype=np.float64)
        except ValueError:
            pass
        else:
            raise AssertionError('NumPy accepted a shape classified as invalid')
        values = np.empty(0, dtype=np.float64) if 0 in shape else np.array([2.])
    passed = values.astype(np.float64) if kind == 'legacy_integer_cast' else values
    if subclass and kind != 'legacy_integer_cast':
        passed = passed.view(UntrustedRavel)
    wanted_dtype = np.dtype('int64' if kind == 'integer' else 'float64')
    assert passed.dtype == wanted_dtype, (case['id'], passed.dtype, wanted_dtype)
    wanted_subclass = bool(subclass) and kind != 'legacy_integer_cast'
    assert isinstance(passed, UntrustedRavel) == wanted_subclass
    original_values = np.array(values, copy=True, subok=False)
    original_passed = np.array(passed, copy=True, subok=False)
    summary = dict(input_dtype=passed.dtype.name, input_subclass=isinstance(passed, UntrustedRavel),
                   input_profile=_input_profile(passed))
    try:
        if case.get('route', 'native') == 'public':
            # Genuine integer/subclass cases reach this call without a worker-side cast.
            result = {'': public.calculate_eccentricity(passed)}
        else:
            result = native.component_batch('surface.calculate_eccentricity',
                {'fire_length_to_width_ratio': passed}, {}, {}, shape)
    except ValueError:
        if valid:
            raise
        summary['outcome'] = 'value_error'
    except BaseException as error:
        raise AssertionError('non-ValueError shape failure: ' + type(error).__name__) from error
    else:
        if not valid:
            raise AssertionError('native accepted an invalid NumPy shape')
        assert calls == [], 'native invoked the overridable ndarray.ravel method'
        actual = result['']
        assert actual.shape == expected.shape == shape
        assert actual.dtype == expected.dtype, 'result dtype differs from reference'
        assert not np.shares_memory(actual, values)
        assert not np.shares_memory(actual, passed)
        np.testing.assert_array_equal(values, original_values)
        np.testing.assert_array_equal(np.asarray(passed), original_passed)
        assert calls == [], 'comparison or native execution invoked ndarray.ravel'
        if len(shape) > 32:
            assert actual.base is not None
        del result, values, passed, original_values, original_passed
        gc.collect()
        np.testing.assert_array_equal(actual, expected)
        if actual.size:
            index = (0,) * len(shape)
            original = expected[index]
            actual[index] = -7.
            assert expected[index] == original
        summary['outcome'] = 'computed'
    return summary


def _source_values(shape, layout, integer):
    """Preserve the legacy source dtype, layout, and readonly construction."""
    if not layout:
        return np.full(shape, 2., dtype=np.float64)
    values = (np.arange(6, dtype=np.int64 if integer else np.float64) + 2).reshape(shape)
    if layout == 'reverse':
        return values[::-1]
    if layout == 'fortran':
        return np.asfortranarray(values)
    values.setflags(write=False)
    return values


def main(arguments):
    """Emit an ordered flushed start/completion pair for every requested case."""
    if sys.flags.optimize != 0:
        raise RuntimeError('Shape probe checks require optimization level zero')
    native, public, reference = _load_native(arguments[0], arguments[1])
    cases = json.loads(arguments[2])
    for case in cases:
        print(json.dumps(dict(event='start', id=case['id'])), flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            summary = _probe_case(case, native, public, reference)
        actual_warnings = [type(item.message).__name__ for item in caught]
        assert actual_warnings == case.get('warnings', []), (case['id'], actual_warnings)
        print(json.dumps(dict(event='passed', id=case['id'], warnings=actual_warnings, **summary)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
