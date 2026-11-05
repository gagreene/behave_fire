"""Record platform-specific NumPy scalar aliases for native compatibility checks."""

import json
import operator
import platform

import numpy as np


def _describe(value):
    """Describe a scalar without losing its concrete NumPy class.

    :param value: Scalar or zero-dimensional result.
    :return: JSON-compatible class, dtype, and value evidence.
    """
    if isinstance(value, np.ndarray):
        value = value[()]
    dtype = np.asarray(value).dtype
    return {"type": type(value).__name__, "char": dtype.char, "num": dtype.num, "value": str(value)}


def _evaluate(name, left, right):
    """Evaluate one explicit reference operation for a scalar pair.

    :param name: Operation label.
    :param left: Left scalar payload.
    :param right: Right scalar payload.
    :return: Result description or the reference exception category.
    """
    try:
        if name == "object_select_left":
            result = np.where(True, np.array(left, dtype=object), np.array(right, dtype=object))
        elif name == "ordinary_select_left":
            result = np.where(True, left, right)
        else:
            operation = {"add": operator.add, "multiply": operator.mul, "power": operator.pow, "minimum": np.minimum}[name]
            result = operation(left, right)
        return _describe(result)
    except Exception as error:
        return {"error": type(error).__name__}


def main():
    """Print scalar alias and promotion evidence from the current platform.

    :return: None; writes a single JSON record to standard output.
    """
    values = {
        "canonical_signed": np.int64(2), "alias_signed": np.longlong(2),
        "canonical_unsigned": np.uint64(2), "alias_unsigned": np.ulonglong(2),
        "builtin": 2, "int32": np.int32(2), "bool": True,
    }
    operations = ("add", "multiply", "power", "object_select_left", "ordinary_select_left", "minimum")
    report = {
        "numpy": np.__version__, "platform": platform.system(),
        "types": {key: _describe(value) for key, value in values.items()}, "pairs": {},
    }
    for left_name, left in values.items():
        for right_name, right in values.items():
            report["pairs"][left_name + "/" + right_name] = {
                name: _evaluate(name, left, right) for name in operations
            }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
