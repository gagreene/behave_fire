"""Bound fixture portability to reviewed non-finite-to-int32 conversion cases.

The explicit Python implementation inherits NumPy's host conversion behavior.
These four original inputs have reviewed x86 frozen diagnostics. Only an actual
different int32 conversion activates a live Python acceptance oracle. All other
cases retain the frozen oracle, and no scientific tolerance changes here.
"""

from copy import deepcopy
import hashlib
from importlib import import_module
import json
import warnings

import numpy as np


# Exact original inputs prevent this rule from spreading to other scenarios.
_CASES = {
    ("behave_fire.components.crown.coerce_crown_fire_inputs", "coerce_crown_fire_inputs/nan"):
        ("fuel_model_grid", "3f331e79f09ed9b17d0d8f0c9d82d0d3e4b0e6c9d28077d68d256aee5cdc6648"),
    ("behave_fire.components.crown.coerce_crown_fire_inputs", "coerce_crown_fire_inputs/positive-infinity"):
        ("fuel_model_grid", "ab21341a73d4a5e5a61f398bcd8903bde7d8d73f50b928442e329291cc8841a1"),
    ("behave_fire.components.ignite.calculate_lightning_ignition_probability", "calculate_lightning_ignition_probability/positive-infinity"):
        ("fuel_bed_type_grid", "2de549fdc61817dc2a173778d824d54e210c6b7df807d10559b897da59a6a6c1"),
    ("behave_fire.components.fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture", "calculate_fine_dead_fuel_moisture/nan"):
        ("dry_bulb_i", "24b5e4057fd4e0fa804783d5b5742c38a20fe659f7961058c019d5e6413afed0"),
}


def _decode(value):
    """Decode the simple numeric arguments in the four reviewed scenarios.

    :param value: Encoded scalar, array, list, or mapping.
    :return: Explicit Python input without native imports.
    """
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "float" in value:
        return float(value["float"])
    if "dtype" in value:
        return np.array([_decode(item) for item in value["values"]], dtype=value["dtype"]).reshape(value["shape"])
    return {key: _decode(item) for key, item in value.items()}


def acceptance_output(case, encode, matches):
    """Select live Python only for a demonstrated alternate host int32 cast.

    :param case: Original frozen fixture, retained as a diagnostic.
    :param encode: Exporter's lossless value encoder.
    :param matches: Existing strict structural and scientific comparator.
    :return: Live encoded output for approved alternate casts, else frozen output.
    """
    outputs = reference_outputs(case, encode)
    if outputs is None:
        return case["output"]
    current, original = outputs
    assert matches(case["output"], original, True), f"Frozen cast diagnostic drift: {case['id']}"
    return current


def canonical_case(case, encode, matches):
    """Keep default exports and --check on the original reviewed diagnostics.

    :param case: Fresh explicit Python record on the current host.
    :param encode: Exporter's lossless value encoder.
    :param matches: Existing strict structural and scientific comparator.
    :return: Copy with only cast-dependent output canonicalized, or input record.
    """
    outputs = reference_outputs(case, encode)
    if outputs is None:
        return case
    current, original = outputs
    assert matches(case["output"], current, True), f"Live Python cast oracle drift: {case['id']}"
    result = deepcopy(case)
    result["output"] = original
    return result


def reference_outputs(case, encode):
    """Calculate live and original-cast outputs for exact reviewed input cases.

    :param case: Frozen or freshly exported case with exact original inputs.
    :param encode: Exporter's lossless value encoder.
    :return: Current and original-cast outputs, or None for frozen-only cases.
    """
    specification = _CASES.get((case.get("entry_point"), case.get("id")))
    if specification is None or "exception" in case:
        return None
    argument, fingerprint = specification
    serialized = json.dumps(case["inputs"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(serialized.encode()).hexdigest() != fingerprint:
        return None
    arguments = _decode(case["inputs"])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        converted = np.asarray(arguments[argument], dtype=np.int32)
        original_cast = np.full(converted.shape, -(2**31), dtype=np.int32)
        if np.array_equal(converted, original_cast):
            return None
        module, name = case["entry_point"].rsplit(".", 1)
        operation = getattr(import_module(module), name)
        current = encode(operation(**arguments))
        original = encode(operation(**dict(arguments, **{argument: original_cast})))
    return current, original
