"""Owned float64 data, typed controls, and dependency-local shape planning."""
from dataclasses import dataclass
from enum import IntEnum
from itertools import zip_longest

import numpy as np
from .._ownership import _OWNER


def _validate_value(value):
    """Reject opaque arithmetic and non-real storage before any float coercion.

    :param value: Original scalar, numeric array, or nested numeric sequence.
    :return: None; unsupported numeric objects raise TypeError.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError("Rust accepts real numeric arrays; use the Python backend for object or complex data.")
    elif isinstance(value, np.generic):
        if type(value) is not np.dtype(value.dtype).type or value.dtype.kind not in "biuf":
            raise TypeError("Rust accepts ordinary real numeric scalars.")
    elif type(value) in (list, tuple):
        for item in value:
            _validate_value(item)
    elif type(value) not in (bool, int, float):
        raise TypeError("Rust accepts ordinary real numeric inputs; opaque arithmetic requires Python.")


def broadcast_shape(*shapes):
    """Combine logical shapes without NumPy's legacy 32-axis broadcast iterator.

    :param shapes: Actual operand shapes, including scalar and empty shapes.
    :return: Broadcast shape; incompatible nonunit extents raise ValueError.
    """
    result = []
    for extents in zip_longest(*(reversed(shape) for shape in shapes), fillvalue=1):
        sizes = set(extents) - {1}
        if len(sizes) > 1:
            raise ValueError('Input shapes cannot be broadcast together.')
        result.append(next(iter(sizes), 1))
    return tuple(reversed(result))


def normalize_control(value, dtype):
    """Own a validated integer or flag control without a float64 intermediate.

    :param value: Integer/boolean control after its entry-point-specific normalization.
    :param dtype: Required signed/unsigned integer or boolean storage dtype.
    :return: Independent contiguous typed array; missing, nonintegral, or out-of-range controls raise.
    """
    _validate_value(value)
    target = np.dtype(dtype)
    if target.kind not in "biu":
        raise TypeError("Control storage must be integer or boolean.")
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError("A typed control cannot be missing.")
    if type(value) is int and target.kind in "iu":
        limits = np.iinfo(target)
        if value < limits.min or value > limits.max:
            raise OverflowError("Control value is outside its declared integer range.")
    data = np.asarray(np.ma.getdata(value))
    if data.dtype.kind not in "biu":
        raise TypeError("A typed control must be integral before storage conversion.")
    if target.kind in "iu" and data.size:
        limits = np.iinfo(target)
        if data.min().item() < limits.min or data.max().item() > limits.max:
            raise OverflowError("Control value is outside its declared integer range.")
    return np.array(data, dtype=target, copy=True, order="C", subok=False)


def normalize_flag(value):
    """Own a boolean condition using the public real-value truth convention.

    :param value: Ordinary real boolean condition or integral enum; missing flags are invalid.
    :return: Independent boolean array; complex/object conditions and missing values raise.
    """
    if isinstance(value, IntEnum):
        value = int(value)
    _validate_value(value)
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError('A typed flag cannot be missing.')
    data = np.asarray(np.ma.getdata(value))
    if data.dtype.kind == 'f' and np.isnan(data).any():
        raise ValueError('A typed flag cannot be missing.')
    return np.array(data, dtype=np.bool_, copy=True, order='C', subok=False)


def normalize_fuel_identifier(value):
    """Own fuel IDs with the undefined-row rule instead of generic control bounds.

    :param value: Real fuel identifier scalar or array; nonfinite IDs are undefined.
    :return: Owned int32 IDs in 0..256; masked and fractional identifiers still raise.
    """
    from ..components.fuel_models_array import _fuel_model_indices

    _validate_value(value)
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError('A typed identifier cannot be missing.')
    data = np.asarray(value)
    if data.dtype.kind == 'f' and np.any(np.isfinite(data) & (data != np.trunc(data))):
        raise ValueError('Identifiers must be finite integral values.')
    return _fuel_model_indices(data)


def normalize_identifier(value):
    """Retain valid integral float identifiers without narrowing integer inputs through float64.

    :param value: Original real identifier scalar or array.
    :return: Owned int32 IDs; missing, fractional, and out-of-range values raise.
    """
    _validate_value(value)
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError('A typed identifier cannot be missing.')
    data = np.asarray(np.ma.getdata(value))
    if data.dtype.kind == 'f':
        if not np.all(np.isfinite(data)) or np.any(data != np.trunc(data)):
            raise ValueError('Identifiers must be finite integral values.')
        limits = np.iinfo(np.int32)
        if data.size and (data.min() < limits.min or data.max() > limits.max):
            raise OverflowError('Identifier is outside the int32 range.')
        return np.array(data, dtype=np.int32, copy=True, order='C')
    return normalize_control(value, np.int32)


def normalize_physical(value):
    """Own supplied physical values as float64, filling masked positions with NaN.

    :param value: Real numeric scalar, sequence, ndarray, or masked array.
    :return: Contiguous float64 storage independent of caller buffers; an existing
            registered snapshot may be reused within the same public call.
    """
    _validate_value(value)
    plain_sequence = type(value) in (list, tuple) and all(
        type(item) in (bool, int, float) or isinstance(item, np.generic) for item in value)
    if type(value) is np.ndarray or type(value) in (bool, int, float) or isinstance(value, np.generic) or plain_sequence:
        owner = _OWNER.get()
        if (owner is not None and owner.owns_storage(value) and value.dtype == np.float64
                and value.flags.c_contiguous and value.flags.aligned):
            return value
        # Plain real storage has no mask to inspect or materialize. Keep the
        # masked/subclass path below for operands whose metadata matters.
        return np.array(value, dtype=np.float64, copy=True, order="C", subok=False)
    # Cast before filling: integer storage cannot represent NaN.
    converted = np.ma.asarray(value, dtype=np.float64)
    return np.array(converted.filled(np.nan), dtype=np.float64, copy=True, order="C", subok=False)


def normalize_selector(kind, value):
    """Retain the public unit-index convention before owning exact typed codes.

    :param kind: Existing unit family name.
    :param value: Original selector, including supported index-array conventions.
    :return: Owned int32 selector codes with their normalized logical axes.
    """
    from ._bridge import _unit_code

    if isinstance(value, IntEnum):
        value = int(value)
    _validate_value(value)
    # Unit indexing can discard a mask, so validate missingness on the original input.
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError("A typed selector cannot be missing.")
    data = np.asarray(np.ma.getdata(value))
    if data.dtype.kind == 'f' and not np.isfinite(data).all():
        raise ValueError('A typed selector must be finite.')
    return normalize_control(_unit_code(kind, value), np.int32)


def plan_shapes(inputs, dependencies):
    """Group outputs by real dependency closure and broadcast shape.

    :param inputs: Already normalized structural input mapping.
    :param dependencies: Output names mapped to required input names only.
    :return: Shape groups; unrelated axes never participate and missing inputs raise KeyError.
    """
    groups = {}
    for output, names in dependencies.items():
        required = tuple(sorted(set(names)))
        shape = broadcast_shape(*(np.shape(inputs[name]) for name in required))
        groups.setdefault((shape, required), []).append(output)
    return [ShapeGroup(shape, required, tuple(outputs))
            for (shape, required), outputs in groups.items()]


@dataclass(frozen=True)
class ShapeGroup:
    """One dependency-local output group, including a genuine zero-length shape."""

    shape: tuple
    inputs: tuple
    outputs: tuple
