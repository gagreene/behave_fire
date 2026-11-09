"""Record independent atan2 layout selection without changing production code."""

import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def _capture_corpora():
    """Capture every original standalone direction record from Python only.

    :return: Independently evaluated rows retaining complete original case identities.
    """
    from behave_fire.components import surface as reference

    root = next(parent for parent in Path(__file__).resolve().parents if (parent / 'pytest.ini').is_file())
    records = []
    for path in sorted((root / 'rust/crates/behave-fire/tests/data').glob('*.json')):
        if path.name.endswith('_platform.json'):
            continue
        source = path.read_bytes()
        document = json.loads(source)
        cases = document if isinstance(document, list) else document.get('cases', []) if isinstance(document, dict) else []
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or (case.get('name') != 'calculate_direction_of_max_spread'
                    and case.get('entry_point') != 'behave_fire.components.surface.calculate_direction_of_max_spread'):
                continue
            arguments = {key: _decode(value) for key, value in case['inputs'].items()}
            if case.get('normalize_half_loop'):
                for key in ('x_component', 'y_component'):
                    value = arguments[key]
                    arguments[key] = value.astype(np.float32) if isinstance(value, (np.ndarray, np.generic)) else np.float32(value)
            row = {'fixture': path.name, 'fixture_sha256': hashlib.sha256(source).hexdigest(),
                   'case_index': index, 'original_case': case}
            with np.errstate(all='ignore'):
                try:
                    value = reference.calculate_direction_of_max_spread(**arguments, **case.get('options', {}))
                    array = np.asanyarray(value)
                    row['output'] = {'dtype': str(array.dtype), 'shape': array.shape,
                                     'values': [float(v) if np.isfinite(v) else 'nan' if np.isnan(v) else '+inf' if v > 0 else '-inf'
                                                for v in np.ma.getdata(array).reshape(-1)],
                                     'masks': np.ma.getmaskarray(array).reshape(-1).tolist(),
                                     'masked_array': bool(np.ma.isMaskedArray(value)),
                                     'scalar_array': array.ndim == 0, 'numpy_scalar': isinstance(value, np.generic)}
                except Exception as error:
                    row['exception'] = type(error).__name__
            records.append(row)
    return records


def _decode(value):
    """Reconstruct numeric operands from the existing standalone fixture schemas.

    :param value: Original encoded scalar, array, or tagged value.
    :return: Independent Python input with dtype, rank, mask, and weak-scalar identity.
    """
    if not isinstance(value, dict):
        return value
    if 'float' in value:
        return float(value['float'])
    if 'dtype' not in value:
        return {key: _decode(item) for key, item in value.items()}
    dtype = np.dtype(value['dtype'])
    if 'raw_bits' in value:
        array = np.array(value['raw_bits'], dtype=np.uint64).astype(dtype)
    elif 'object_payloads' in value:
        array = np.array([_decode(item) for item in value['object_payloads']], dtype=object)
    else:
        array = np.array([float(item) if isinstance(item, str) and item in ('nan', '+inf', '-inf') else _decode(item)
                          for item in value['values']], dtype=dtype)
    array = array.reshape(value.get('shape', () if value.get('scalar_array') else array.shape))
    if value.get('masked_array') or 'mask' in value:
        array = np.ma.array(array, mask=np.array(value.get('masks', value.get('mask'))).reshape(array.shape))
        if value.get('fill_value') is not None:
            array.fill_value = _decode(value['fill_value'])
    elif value.get('numpy_scalar'):
        array = array[()]
    if value.get('weak_kind'):
        array = {'int': int, 'float': float, 'bool': bool}[value['weak_kind']](array)
    return array


def _describe(value):
    """Describe structural array properties.

    :param value: Operand or result.
    :return: JSON-compatible dtype, shape, and strides.
    """
    array = np.asanyarray(value)
    return {'dtype': str(array.dtype), 'shape': array.shape, 'strides': array.strides,
            'aligned': bool(array.flags.aligned), 'C': bool(array.flags.c_contiguous), 'F': bool(array.flags.f_contiguous)}


def _record(name, x, y, atan2f):
    """Compare actual NumPy layout execution with contiguous and scalar-libm evidence.

    :param name: Diagnostic scenario.
    :param x: First reference operand.
    :param y: Second reference operand.
    :param atan2f: Independently loaded system primitive.
    :return: JSON-compatible result classification and structural iterator trace.
    """
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native
    # Retain an independent float32 NumPy/libm trace for historical attribution.
    # The native correctness reference uses the actual inputs widened to float64.
    nx, ny = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    with np.errstate(all='ignore'):
        actual = np.asarray(np.arctan2(ny, nx))
        bx, by = np.broadcast_arrays(np.asarray(nx, dtype=np.float32), np.asarray(ny, dtype=np.float32))
        contiguous = np.arctan2(np.array(by, order='C', copy=True), np.array(bx, order='C', copy=True))
        scalar = np.array([atan2f(float(vy), float(vx)) for vx, vy in zip(bx.flat, by.flat)], dtype=np.float32).reshape(bx.shape)
        expected_direction = reference.calculate_direction_of_max_spread(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.ones(bx.shape), 'RelativeToNorth', 95.)
        native_direction = native.calculate_direction_of_max_spread(x, y, np.ones(bx.shape), 'RelativeToNorth', 95.)
    mask = ~(np.isnan(actual) | np.isnan(scalar) | np.isnan(contiguous))
    actual_bits, scalar_bits, vector_bits = [np.asarray(value).view(np.uint32) for value in (actual, scalar, contiguous)]
    difference = mask & ((actual_bits != scalar_bits) | (actual_bits != vector_bits))
    positions = np.flatnonzero(difference)[:12]
    result = {'name': name, 'native_physical_dtype': 'float64', 'x': _describe(x), 'y': _describe(y), 'normalized_x': _describe(nx),
              'normalized_y': _describe(ny), 'out': _describe(actual),
              'numpy_equals_libm': bool(np.array_equal(actual_bits[mask], scalar_bits[mask])),
              'numpy_equals_contiguous': bool(np.array_equal(actual_bits[mask], vector_bits[mask])),
              'native_strict_pass': bool(np.allclose(native_direction, expected_direction, rtol=1e-9, atol=1e-12, equal_nan=True)),
              'witnesses': [{'index': int(i), 'x': float(bx.flat[i]), 'y': float(by.flat[i]),
                             'numpy': hex(int(actual_bits.flat[i])), 'libm': hex(int(scalar_bits.flat[i])),
                             'contiguous': hex(int(vector_bits.flat[i]))} for i in positions]}
    with np.nditer([ny, nx, None], flags=['external_loop', 'buffered', 'grow_inner', 'zerosize_ok'],
                   op_flags=[['readonly', 'aligned'], ['readonly', 'aligned'], ['writeonly', 'allocate', 'aligned']],
                   op_dtypes=[np.float32] * 3, order='K', casting='unsafe', buffersize=np.getbufsize()) as iterator:
        result['iterator_chunks'] = []
        for chunk in iterator:
            result['iterator_chunks'].append([_describe(item) for item in chunk])
            if len(result['iterator_chunks']) == 3:
                break
    return result


def main():
    """Print compact layout evidence for current NumPy and platform libm.

    :return: Writes JSON diagnostic evidence to standard output.
    """
    library = ctypes.CDLL(ctypes.util.find_library('m') or 'ucrtbase.dll')
    atan2f = library.atan2f
    atan2f.argtypes = [ctypes.c_float, ctypes.c_float]
    atan2f.restype = ctypes.c_float
    x = np.array([-1., -127., 1., -1., -0., 0., 2., -1.], dtype=np.float32)
    y = np.array([65534., 126., 1., np.finfo(np.float32).max, 1., 1., 3., -126.], dtype=np.float32)
    layouts = [('contiguous', x, y), ('reverse-both', x[::-1], y[::-1]),
               ('reverse-x', x[::-1], y), ('reverse-y', x, y[::-1]),
               ('slice-positive', np.repeat(x, 2)[::2], np.repeat(y, 2)[::2]),
               ('slice-negative', np.repeat(x, 2)[::-2], np.repeat(y, 2)[::-2]),
               ('broadcast', np.broadcast_to(x, (2, 8)), np.broadcast_to(y, (2, 8))),
               ('weak-int-reverse', x[::-1], 65534), ('weak-float-reverse', x[::-1], 65534.)]
    readonly = x.copy()
    readonly.setflags(write=False)
    unaligned_x = np.ndarray(x.shape, dtype=np.float32, buffer=bytearray(x.nbytes + 1), offset=1)
    unaligned_y = np.ndarray(y.shape, dtype=np.float32, buffer=bytearray(y.nbytes + 1), offset=1)
    unaligned_x[:] = x
    unaligned_y[:] = y
    layouts.extend([('readonly', readonly, y), ('unaligned', unaligned_x, unaligned_y),
                    ('unaligned-reverse', unaligned_x[::-1], unaligned_y[::-1]),
                    ('unaligned-one-reverse', unaligned_x[::-1], y[::-1])])
    for shape in [(2, 4), (2, 2, 2)]:
        a, b = x.reshape(shape), y.reshape(shape)
        layouts.extend([(f'C-{shape}', a, b), (f'F-{shape}', np.asfortranarray(a), np.asfortranarray(b)),
                        (f'mixed-order-{shape}', a, np.asfortranarray(b)),
                        (f'reverse-first-{shape}', a[::-1], b[::-1]),
                        (f'reverse-last-{shape}', a[..., ::-1], b[..., ::-1]),
                        (f'one-reverse-{shape}', a[..., ::-1], b),
                        (f'F-reverse-first-{shape}', np.asfortranarray(a)[::-1], np.asfortranarray(b)[::-1]),
                        (f'F-reverse-last-{shape}', np.asfortranarray(a)[..., ::-1], np.asfortranarray(b)[..., ::-1])])
    for count in [8, np.getbufsize() + 17]:
        a = np.resize(np.array([-1, -127, 2, 1], dtype=np.int8), count)[::-1]
        b = np.resize(np.array([65534, 126, 3, 2], dtype=np.uint16), count)[::-1]
        layouts.extend([(f'cast-both-{count}', a, b), (f'cast-one-{count}', a.astype(np.float32)[::-1], b),
                        (f'cast-f32-reverse-{count}', a, b.astype(np.float32)[::-1]),
                        (f'half-normalized-{count}', a, a)])
    report = {'platform': platform.platform(), 'numpy': np.__version__, 'buffer_size': np.getbufsize(),
              'cases': [_record(name, a, b, atan2f) for name, a, b in layouts],
              'corpora': _capture_corpora()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
