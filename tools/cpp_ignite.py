"""Verify and measure the four approved standalone Ignite reference cases."""

import json
import math
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

APPROVED_CASES = (
    ("ignite-dry", 0),
    ("ignite-moisture-12-5", 0.125),
    ("ignite-moisture-25", 0.25),
    ("ignite-wet", 0.5),
)
DATASET = Path("tests/data/cpp_reference/ignite")
FIELDS = ("firebrand_probability", "fuel_temperature_f")
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
MEASUREMENT_POLICY = {"approved": False, "fields": {field: {"kind": "measurement_only"} for field in FIELDS}}
SOURCES = ("behaveUnits.cpp", "behaveUnits.h", "ignite.cpp", "ignite.h", "igniteInputs.cpp", "igniteInputs.h")


def _parse_int(token):
    """Preserve the negative sign when decoding a lexical zero.

    :param token: JSON integer token.
    :return: Integer value or negative binary64 zero for the signed zero token.
    """
    return -0.0 if token == "-0" else int(token)


def _validate_output(output):
    """Require both finite scalar outputs with probability in its valid interval.

    :param output: Observed Fahrenheit fuel temperature and probability fraction.
    :return: None. Raises for invalid values or incomplete output.
    """
    if (not isinstance(output, dict) or set(output) != set(FIELDS)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in output.values())
            or not 0 <= output["firebrand_probability"] <= 1):
        raise ValueError("Incomplete or invalid Ignite output")


def artifacts(root):
    """Bind a measurement to the exact guarded Ignite dataset.

    :param root: Reference checkout root.
    :return: File names mapped to their SHA256 digests.
    """
    return {name: reference._digest(Path(root) / DATASET / name) for name in FILES}


def load_cases(root):
    """Require exactly the approved inputs in their stable case order.

    :param root: Reference checkout root.
    :return: Validated cases.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=_parse_int)
    expected = [dict(id=case, operation="ignite", air_temperature_f=71, sun_shade_fraction=0.5, moisture_1h=moisture)
                for case, moisture in APPROVED_CASES]
    if json.dumps(rows, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Ignite inputs differ from the approved cases")
    return rows


def load_observations(root):
    """Verify pinned raw Ignite results before reporting or measuring parity.

    :param root: Reference checkout root.
    :return: Guarded independent C++ observations with verified provenance.
    """
    directory = Path(root) / DATASET
    manifest = reference._document(directory / "manifest.json")
    schema = reference._document(directory / "schema.json")
    results = read_cpp_results(root, directory / "results.jsonl")
    expected = {name: reference._digest(directory / name) for name in FILES if name != "manifest.json"}
    sources = reference._prepared_build(root)["upstream_source_sha256_portable"]
    trusted = {f"src/behave/{name}": sources[f"src/behave/{name}"] for name in SOURCES}
    if (manifest["schema_version"] != 1 or manifest["operation"] != "ignite"
            or manifest["upstream_sha"] != reference.PIN or manifest["fields"] != list(FIELDS)
            or manifest["files"] != expected or schema["operation"] != "ignite" or set(schema["output"]) != set(FIELDS)
            or any(manifest[key] != len(results) for key in ("available", "case_count", "result_count"))
            or manifest["build"]["pinned_checkout_clean"] is not True
            or manifest["build"]["upstream_sources"] != trusted):
        raise ValueError("Incomplete Ignite observations or provenance")
    return results


def measure_ports(root, backends):
    """Record unqualified public port results in each requested backend.

    :param root: Reference checkout root.
    :param backends: Explicit backend names already configured for measurement.
    :return: Per-backend rows in the approved case order.
    """
    import numpy as np

    cases = load_cases(root)
    measurements = {}
    for backend in backends:
        ignite = import_module(f"behave_fire.{backend}").ignite
        rows = []
        for case in cases:
            air = case["air_temperature_f"]
            shade = case["sun_shade_fraction"]
            moisture = case["moisture_1h"]
            output = {
                "firebrand_probability": float(np.asarray(
                    ignite.calculate_firebrand_ignition_probability(air, shade, moisture)).item()),
                "fuel_temperature_f": float(np.asarray(ignite.calculate_fuel_temperature(air, shade)).item()),
            }
            _validate_output(output)
            rows.append({"id": case["id"], "output": output})
        measurements[backend] = rows
    return measurements


def read_cpp_results(root, path):
    """Check the result stored by the pinned Ignite driver.

    :param root: Reference checkout root.
    :param path: Raw C++ JSONL observations.
    :return: Approved case rows with verified execution and output counts.
    """
    rows = reference._rows(path)
    validate_cpp_rows(root, rows)
    return rows


def validate_cpp_rows(root, rows):
    """Require one guarded C++ entry and one complete output per case.

    :param root: Reference checkout root.
    :param rows: Parsed raw observations or compiler evidence.
    :return: None. Raises for incomplete or invalid observations.
    """
    if [row["id"] for row in rows] != [case["id"] for case in load_cases(root)]:
        raise ValueError("Ignite case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "output_entries", "output"} or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))):
            raise ValueError("Ignite row lacks one complete output action")
        _validate_output(row["output"])
