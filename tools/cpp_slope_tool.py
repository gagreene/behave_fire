"""Validate and measure the two approved standalone SlopeTool cases."""

import hashlib
import json
import math
import struct
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

DATASET = Path("tests/data/cpp_reference/slope_tool")
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
OPERATION = "slope_tool"
SOURCES = ("behaveUnits.cpp", "behaveUnits.h", "slopeTool.cpp", "slopeTool.h")
GUARDS = dict(negative_cases=8, negative_upstream_entries=0, upstream_trap_exit=86,
              output_trap_exit=87, duplicate_ids_rejected=True)
PROJECTION_ID = "slope-projection"
MAP_ID = "slope-map"
CASE_FIELDS = {
    PROJECTION_ID: tuple(f"horizontal_{angle}_ft" for angle in range(0, 91, 15)),
    MAP_ID: ("elevation_ft", "horizontal_ft", "slope_degrees", "slope_percent"),
}
FIELDS = tuple(sorted(field for case_fields in CASE_FIELDS.values() for field in case_fields))
MEASUREMENT_POLICY = dict(approved=False, fields={field: {"kind": "measurement_only"} for field in FIELDS})
REQUESTS = (
    dict(id=PROJECTION_ID, operation=OPERATION, distance_ft=8, slope_degrees=30),
    dict(id=MAP_ID, operation=OPERATION, scale=1980, distance_ft=0.125, contour_interval_ft=20, contours=3),
)


def _validate_output(output, fields):
    """Require exactly the expected finite, nonnegative fields.

    :param output: Observed SlopeTool values or comparison maxima.
    :param fields: Required field names for the case or receipt.
    :return: None. Raises for invalid or incomplete output.
    """
    if not isinstance(output, dict) or set(output) != set(fields):
        raise ValueError("Incomplete SlopeTool output")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in output.values()):
        raise ValueError("Invalid SlopeTool output")


def artifacts(root):
    """Bind a measurement to the approved SlopeTool fixture files.

    :param root: Reference checkout root.
    :return: File names mapped to SHA256 digests.
    """
    return {name: reference._digest(Path(root) / DATASET / name) for name in FILES}


def load_cases(root):
    """Require the two approved requests, in diagnostic order.

    :param root: Reference checkout root.
    :return: Validated case rows.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=reference._signed_int)
    if json.dumps(rows, sort_keys=True, allow_nan=False) != json.dumps(REQUESTS, sort_keys=True):
        raise ValueError("SlopeTool inputs differ from the approved cases")
    return rows


def load_observations(root):
    """Validate the pinned SlopeTool values, fixture digests, and upstream closure.

    :param root: Reference checkout root.
    :return: Validated C++ observation rows.
    """
    directory = Path(root) / DATASET
    manifest = reference._document(directory / "manifest.json")
    schema = reference._document(directory / "schema.json")
    results = read_cpp_results(root, directory / "results.jsonl")
    expected = {name: reference._digest(directory / name) for name in FILES if name != "manifest.json"}
    prepared_sources = reference._prepared_build(root)["upstream_source_sha256_portable"]
    sources = {f"src/behave/{name}": prepared_sources[f"src/behave/{name}"] for name in SOURCES}
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["operation"] != OPERATION or manifest["upstream_sha"] != reference.PIN
            or manifest["fields"] != list(FIELDS) or manifest["files"] != expected
            or schema["operation"] != OPERATION or schema["output"] != {k: list(v) for k, v in CASE_FIELDS.items()}
            or any(type(manifest[key]) is not int or manifest[key] != len(results)
                   for key in ("available", "case_count", "result_count"))
            or manifest["build"]["pinned_checkout_clean"] is not True
            or manifest["build"]["upstream_sources"] != sources):
        raise ValueError("Incomplete SlopeTool observations or provenance")
    return results


def measure_ports(root, backends):
    """Measure both reviewed SlopeTool cases through the selected public ports.

    :param root: Reference checkout root.
    :param backends: Explicit backend names configured for measurement.
    :return: Per-backend observations in approved case order.
    """
    import numpy as np

    from behave_fire.components.behave_units import LengthUnits, SlopeUnits

    projection, map_slope = load_cases(root)
    feet = LengthUnits.LengthUnitsEnum.Feet
    degrees = SlopeUnits.SlopeUnitsEnum.Degrees
    measurements = {}
    for backend in backends:
        tool_type = import_module(f"behave_fire.{backend}").SlopeTool
        tool = tool_type()
        tool.calculate_horizontal_distance(projection["distance_ft"], feet, projection["slope_degrees"], degrees)
        projected = {field: float(np.asarray(tool.get_horizontal_distance_at_index(index, feet)).item())
                     for index, field in enumerate(CASE_FIELDS[PROJECTION_ID])}
        tool = tool_type()
        tool.calculate_slope_from_map_measurements(map_slope["scale"], map_slope["distance_ft"], feet,
                                                   map_slope["contour_interval_ft"], map_slope["contours"], feet)
        mapped = dict(
            elevation_ft=float(np.asarray(tool.get_slope_elevation_change_from_map_measurements(feet)).item()),
            horizontal_ft=float(np.asarray(tool.get_slope_horizontal_distance_from_map_measurements(feet)).item()),
            slope_degrees=float(np.asarray(tool.get_slope_from_map_measurements_in_degrees()).item()),
            slope_percent=float(np.asarray(tool.get_slope_from_map_measurements_in_percent()).item()),
        )
        _validate_output(projected, CASE_FIELDS[PROJECTION_ID])
        _validate_output(mapped, CASE_FIELDS[MAP_ID])
        measurements[backend] = [dict(id=PROJECTION_ID, output=projected), dict(id=MAP_ID, output=mapped)]
    return measurements


def measurement_evidence(root, paths, record):
    """Summarize SlopeTool values from the authenticated source matrix.

    :param root: Checkout with the approved SlopeTool observations.
    :param paths: Compiler and platform report paths.
    :param record: Qualified source record binding the report hashes.
    :return: Measurement receipt with count, differing bits, and field maxima.
    """
    expected = load_observations(root)
    signature = artifacts(root)
    pending = {name: report["sha256"] for name, report in record["reports"].items()}
    if len(pending) != 9 or {name for name in pending if not name.startswith("ports-")} != {"cpp-GNU", "cpp-MSVC"}:
        raise ValueError("Incomplete SlopeTool compiler or platform evidence")
    count, nonmatching = 0, 0
    maxima = dict.fromkeys(FIELDS, 0.0)
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw, object_pairs_hook=reference._unique_object,
                            parse_constant=reference._reject_constant, parse_int=reference._signed_int)
        kind = report["kind"]
        if kind not in ("cpp", "ports"):
            raise ValueError("Unknown SlopeTool report kind")
        name = "cpp-" + report["compiler"]["id"] if kind == "cpp" else "ports-" + report["platform_key"]
        if pending.pop(name, None) != hashlib.sha256(raw).hexdigest():
            raise ValueError("SlopeTool report differs from the source-qualified evidence")
        measurement = report[OPERATION]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != signature:
            raise ValueError("SlopeTool report used a different dataset")
        if kind == "cpp":
            if json.dumps(report["protocol"][OPERATION], sort_keys=True) != json.dumps(GUARDS, sort_keys=True):
                raise ValueError("Missing SlopeTool entry-guard evidence")
            rows = measurement["observations"]
            validate_cpp_rows(root, rows)
            streams = (rows,)
        else:
            backends = measurement["observations"]
            if not isinstance(backends, dict) or set(backends) != {"python", "rust"} or any(
                not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {"id", "output"}
                                                  for row in rows) for rows in backends.values()
            ):
                raise ValueError("Incomplete SlopeTool backend observations")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Incomplete SlopeTool report case identities")
            for row, original in zip(rows, expected, strict=True):
                case_fields = CASE_FIELDS[row["id"]]
                _validate_output(row["output"], case_fields)
                for field in case_fields:
                    actual = float(row["output"][field])
                    pinned = float(original["output"][field])
                    nonmatching += struct.pack("!d", actual) != struct.pack("!d", pinned)
                    maxima[field] = max(maxima[field], abs(actual - pinned))
                    count += 1
    if pending or count != 176:
        raise ValueError("Incomplete SlopeTool measurement evidence")
    return dict(artifacts=signature, comparisons=count, nonmatching=nonmatching, max_abs=maxima)


def read_cpp_results(root, path):
    """Read the guarded C++ results without losing the sign of zero.

    :param root: Reference checkout root.
    :param path: Raw C++ observations in JSONL format.
    :return: Validated rows in approved case order.
    """
    rows = reference._rows(path, parse_int=reference._signed_int)
    validate_cpp_rows(root, rows)
    return rows


def validate_cpp_rows(root, rows):
    """Require one upstream and output entry for each reviewed SlopeTool input.

    :param root: Reference checkout root.
    :param rows: Parsed raw observations or compiler evidence.
    :return: None. Raises on incomplete or invalid observations.
    """
    if [row["id"] for row in rows] != [case["id"] for case in load_cases(root)]:
        raise ValueError("SlopeTool case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "output_entries", "output"} or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))):
            raise ValueError("SlopeTool row lacks one complete output action")
        _validate_output(row["output"], CASE_FIELDS[row["id"]])


def validate_measurement(root, policy, qualification):
    """Require measurement-only policy and verify any matrix receipt.

    :param root: Reference checkout root.
    :param policy: SlopeTool section of the canonical policy.
    :param qualification: Verified source qualification, or None while measuring.
    :return: None. Raises on invalid policy or receipt.
    """
    if not isinstance(policy, dict) or policy.get("approved") is not False or policy != MEASUREMENT_POLICY:
        raise ValueError("SlopeTool policy must remain measurement-only")
    if qualification is None:
        return
    receipt = qualification.get(OPERATION) if isinstance(qualification, dict) else None
    if (not isinstance(receipt, dict) or set(receipt) != {"artifacts", "comparisons", "nonmatching", "max_abs"}
            or receipt["artifacts"] != artifacts(root) or type(receipt["comparisons"]) is not int
            or receipt["comparisons"] != 176 or type(receipt["nonmatching"]) is not int):
        raise ValueError("Incomplete or stale SlopeTool measurement receipt")
    maxima = receipt["max_abs"]
    _validate_output(maxima, FIELDS)
    pinned = {field: value for row in load_observations(root) for field, value in row["output"].items()}
    streams = receipt["comparisons"] // len(FIELDS)
    minimum = sum(value > 0 for value in maxima.values())
    maximum = streams * sum(maxima[field] > 0 or pinned[field] == 0 for field in FIELDS)
    if not minimum <= receipt["nonmatching"] <= maximum:
        raise ValueError("Inconsistent SlopeTool comparison statistics")
