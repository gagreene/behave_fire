"""Print independent NumPy and system-libm evidence for direction fixtures."""

import ctypes
import ctypes.util
import json
import math
import platform
import sys

import numpy as np


def _describe(value):
    """Describe array values and exact float32 payloads.

    :param value: Numeric array to inspect.
    :return: JSON-compatible dtype, values, and binary payloads.
    """
    array = np.asarray(value)
    return {"dtype": str(array.dtype), "values": array.tolist(),
            "f32_bits": [hex(int(item)) for item in array.astype(np.float32).view(np.uint32).reshape(-1)]}


def main():
    """Print oracle evidence without changing fixtures or selecting expectations.

    :return: Writes JSON evidence to standard output.
    """
    from behave_fire.components import surface as reference

    report = {"platform": platform.platform(), "machine": platform.machine(),
              "libc": platform.libc_ver(), "numpy": np.__version__, "python": sys.version}
    from numpy._core import _multiarray_umath
    report["cpu_targets"] = getattr(_multiarray_umath, "__cpu_targets_info__", {})
    report["cpu_features"] = getattr(_multiarray_umath, "__cpu_features__", {})
    library_name = ctypes.util.find_library("m")
    report["libm"] = library_name
    atan2f = None
    if library_name:
        library = ctypes.CDLL(library_name)
        atan2f = library.atan2f
        atan2f.argtypes = [ctypes.c_float, ctypes.c_float]
        atan2f.restype = ctypes.c_float
    try:
        from behave_fire.rust import surface as native
        from behave_fire import _rust
        report["native_extension"] = _rust.__file__
        report["native_physical_dtype"] = "float64"
    except ImportError:
        native = None
    report["cases"] = []
    for name, x, y, normalize in (
            ("case21", np.array([-1], dtype=np.int8), np.array([65534], dtype=np.uint16), False),
            ("case33", np.array([-1], dtype=np.int8), np.array([False], dtype=np.bool_), True)):
        xf, yf = x.astype(np.float32), y.astype(np.float32)
        angles = np.arctan2(yf, xf)
        case = {"case": name, "numpy_angles": _describe(angles),
                "numpy_degrees": _describe(np.degrees(angles)),
                "double_then_f32": _describe(np.array([math.atan2(float(yf[0]), float(xf[0]))], dtype=np.float32))}
        if atan2f:
            case["libm_angles"] = _describe(np.array([atan2f(float(yf[0]), float(xf[0]))], dtype=np.float32))
        for orientation in ("RelativeToUpslope", "RelativeToNorth"):
            reference_x, reference_y = (xf, yf) if normalize else (x, y)
            original = reference.calculate_direction_of_max_spread(reference_x, reference_y, np.ones(1), orientation, 95.0)
            expected = reference.calculate_direction_of_max_spread(x.astype(np.float64), y.astype(np.float64), np.ones(1), orientation, 95.0)
            case[orientation] = {"python": _describe(original), "python_widened": _describe(expected)}
            if native is not None:
                actual = native.calculate_direction_of_max_spread(x, y, np.ones(1), orientation, 95.0)
                case[orientation]["rust"] = _describe(actual)
                case[orientation]["strict_pass"] = bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-12))
        report["cases"].append(case)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
