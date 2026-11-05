"""Capture all direction fixture outputs independently from the Python reference."""

import hashlib
import json
import os
from pathlib import Path
import platform
import warnings

import numpy as np


def _decode(value):
    """Reconstruct the frozen operand, preserving scalar and mask metadata.

    :param value: Encoded direction operand.
    :return: Original Python or NumPy input.
    """
    dtype = np.dtype(value["dtype"])
    if "raw_bits" in value:
        array = np.array(value["raw_bits"], dtype=np.uint64).astype(dtype)
    else:
        array = np.array([_number(item) for item in value["values"]], dtype=dtype)
    if value["scalar_array"]:
        array = array.reshape(())
    if value["masked_array"]:
        array = np.ma.array(array, mask=np.array(value["masks"]).reshape(array.shape))
        if value.get("fill_value") is not None:
            array.fill_value = _number(value["fill_value"])
    elif value["numpy_scalar"]:
        array = array[()]
    if value.get("weak_kind"):
        array = {"int": int, "bool": bool, "float": float}[value["weak_kind"]](array)
    return array


def _encode(value):
    """Capture result values and structural metadata from NumPy.

    :param value: Independent reference result.
    :return: JSON-compatible structural oracle.
    """
    array = np.asanyarray(value)
    values = np.ma.getdata(array).reshape(-1)
    return {"values": [float(item) if np.isfinite(item) else "nan" if np.isnan(item) else "+inf" if item > 0 else "-inf" for item in values],
            "masks": np.ma.getmaskarray(array).reshape(-1).tolist(),
            "dtype": str(array.dtype), "masked_array": bool(np.ma.isMaskedArray(value)),
            "scalar_array": array.ndim == 0, "numpy_scalar": isinstance(value, np.generic),
            "fill_value": float(value.fill_value) if np.ma.isMaskedArray(value) else None,
            "math_profile": "numpy-simd", "weak_kind": None}


def _number(value):
    """Decode a finite or tagged floating value.

    :param value: Encoded numeric value.
    :return: Python scalar number.
    """
    return float(value["float"] if isinstance(value, dict) else value)


def main():
    """Write independently evaluated fixture rows with exact input fingerprints.

    :return: Writes JSON evidence to standard output without modifying fixtures.
    """
    from behave_fire.components import surface as reference

    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())
    path = root / "rust/crates/behave-fire/tests/data/integer_direction.json"
    source = path.read_bytes()
    fixture = json.loads(source)
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "libc": platform.libc_ver(), "numpy": np.__version__,
              "fixture_sha256": hashlib.sha256(source).hexdigest(), "cases": []}
    from numpy._core import _multiarray_umath
    report["disabled_cpu_features"] = os.environ.get("NPY_DISABLE_CPU_FEATURES", "")
    report["cpu_features"] = getattr(_multiarray_umath, "__cpu_features__", {})
    report["cpu_targets"] = getattr(_multiarray_umath, "__cpu_targets_info__", {})
    for index, case in enumerate(fixture):
        inputs = {key: _decode(value) for key, value in case["inputs"].items()}
        if case["normalize_half_loop"]:
            for key in ("x_component", "y_component"):
                value = inputs[key]
                inputs[key] = value.astype(np.float32) if isinstance(value, (np.ndarray, np.generic)) else np.float32(value)
        identity = {key: case[key] for key in ("name", "inputs", "options", "normalize_half_loop", "original_loop_dtype")}
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        result = {"index": index, "identity": identity,
                  "identity_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                result["output"] = _encode(reference.calculate_direction_of_max_spread(**inputs, **case["options"]))
            except Exception as error:
                result["exception"] = type(error).__name__
        report["cases"].append(result)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
