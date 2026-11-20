"""Collect standalone FireSize observations for a separate comparison policy."""

import math
import struct
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

APPROVED_CASES = (
    ("calm-crown", "Crown", 0, 4, 2),
    ("surface-5-mph", "Surface", 5, 10, 10),
    ("zero-elapsed", "Surface", 5, 10, 0),
)
DATASET = Path("tests/data/cpp_reference/fire_size")
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
FIELDS = ("area_ft2", "length_to_width_ratio", "perimeter_ft")
INPUTS = ("id", "mode", "wind_mph", "spread_ft_min", "elapsed_min")
MAGNITUDE_MASK = (1 << 63) - 1


def _validate_cpp_rows(root, rows):
    """Require one guarded C++ calculation and geometry read for each case.

    :param root: Reference checkout root.
    :param rows: Parsed driver observations or recorded compiler evidence.
    :return: None. Raises on incomplete or invalid driver observations.
    """
    if [row["id"] for row in rows] != [case["id"] for case in load_cases(root)]:
        raise ValueError("FireSize case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "geometry_entries", "output"}
                or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "geometry_entries"))):
            raise ValueError("FireSize result lacks one complete geometry action")
        _validate_output(row["output"])


def _validate_output(output):
    """Require the complete finite scalar geometry result.

    :param output: Observed area, length-to-width ratio, and perimeter.
    :return: None. Raises on an unsupported shape or geometry value.
    """
    if not isinstance(output, dict) or set(output) != set(FIELDS) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in output.values()
    ):
        raise ValueError("Incomplete or invalid FireSize geometry output")


def artifacts(root):
    """Bind measurements to the exact FireSize dataset contents.

    :param root: Reference checkout root.
    :return: SHA256 digest of each independent reference artifact.
    """
    directory = Path(root) / DATASET
    return {name: reference._digest(directory / name) for name in FILES}


def load_cases(root):
    """Validate the exact three approved cases and their order.

    :param root: Reference checkout root.
    :return: Cases with exact approved inputs and field names.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl")
    expected = [dict(zip(INPUTS, case, strict=True), operation="fire_size") for case in APPROVED_CASES]
    if rows != expected or any(type(row[key]) is not int for row in rows for key in INPUTS[2:]):
        raise ValueError("FireSize cases differ from the approved case set")
    return rows


def load_observations(root):
    """Check the raw, pinned FireSize observations without accepting numerical parity.

    :param root: Reference checkout root.
    :return: Original C++ result rows after integrity and provenance checks.
    """
    directory = Path(root) / DATASET
    manifest = reference._document(directory / "manifest.json")
    schema = reference._document(directory / "schema.json")
    results = read_cpp_results(root, directory / "results.jsonl")
    expected = {name: reference._digest(directory / name) for name in FILES if name != "manifest.json"}
    build = manifest["build"]
    prepared = reference._prepared_build(root)
    sources = prepared["upstream_source_sha256_portable"]
    trusted = {f"src/behave/{name}": sources[f"src/behave/{name}"]
               for name in ("behaveUnits.cpp", "behaveUnits.h", "fireSize.cpp", "fireSize.h")}
    if (manifest["schema_version"] != 1 or manifest["operation"] != "fire_size" or manifest["upstream_sha"] !=
            reference.PIN or manifest["fields"] != list(FIELDS) or manifest["files"] != expected
            or schema["operation"] != "fire_size" or set(schema["output"]) != set(FIELDS)
            or any(manifest[key] != len(results) for key in ("available", "case_count", "result_count"))
            or build["pinned_checkout_clean"] is not True or build["upstream_sources"] != trusted):
        raise ValueError("Incomplete FireSize observations or provenance")
    return results


def measure_ports(root, backends):
    """Record each selected public port's standalone FireSize geometry in base units.

    :param root: Reference checkout root.
    :param backends: Explicit backend names already set up for measurement.
    :return: Per-backend observations in the approved case order.
    """
    cases = load_cases(root)
    units = import_module("behave_fire.components.behave_units")
    speed = units.SpeedUnits.SpeedUnitsEnum
    area = units.AreaUnits.AreaUnitsEnum
    length = units.LengthUnits.LengthUnitsEnum
    time = units.TimeUnits.TimeUnitsEnum
    report = {}
    for backend in backends:
        size_class = import_module(f"behave_fire.{backend}").FireSize
        rows = []
        for case in cases:
            size = size_class()
            crown = case["mode"] == "Crown"
            wind, spread, elapsed = (case[field] for field in INPUTS[2:])
            size.calculate_fire_basic_dimensions(crown, wind, speed.MilesPerHour, spread, speed.FeetPerMinute)
            output = {
                "area_ft2": float(size.get_fire_area(crown, area.SquareFeet, elapsed, time.Minutes)),
                "length_to_width_ratio": float(size.get_fire_length_to_width_ratio()),
                "perimeter_ft": float(size.get_fire_perimeter(crown, length.Feet, elapsed, time.Minutes)),
            }
            _validate_output(output)
            rows.append({"id": case["id"], "output": output})
        report[backend] = rows
    return report


def read_cpp_results(root, path):
    """Verify complete independent FireSize observations from the guarded driver.

    :param root: Reference checkout root.
    :param path: Result file produced by the pinned C++ driver.
    :return: Results joined in the approved case order with verified entry counts.
    """
    rows = reference._rows(path)
    _validate_cpp_rows(root, rows)
    return rows


def summarize_evidence(root, paths):
    """Measure compiler and backend differences without choosing tolerances.

    :param root: Checkout holding the original FireSize observations.
    :param paths: Complete calibration reports from one source-bound CI run.
    :return: Measured per-field maxima and distinct observed binary64 values.
    """
    from cpp_calibrate import validate_evidence

    identity = validate_evidence(root, paths)
    data = {row["id"]: row["output"] for row in load_observations(root)}
    signature = artifacts(root)
    maxima = {field: dict(absolute=0.0, relative=0.0, ulp=0, nonmatching=0) for field in FIELDS}
    observed = {identity: {field: set() for field in FIELDS} for identity in data}
    count = 0
    for path in paths:
        report = reference._document(path)
        fire_size = report["fire_size"]
        if fire_size["artifacts"] != signature:
            raise ValueError("Calibration used different FireSize inputs or observations")
        if report["kind"] == "cpp":
            controls = dict(negative_cases=8, negative_upstream_entries=0, upstream_trap_exit=86,
                            geometry_trap_exit=87, duplicate_ids_rejected=True)
            if report["protocol"]["fire_size"] != controls:
                raise ValueError("Missing FireSize protocol evidence")
            rows = fire_size["observations"]
            _validate_cpp_rows(root, rows)
            streams = [rows]
        else:
            backends = fire_size["observations"]
            if set(backends) != {"python", "rust"} or any(
                set(row) != {"id", "output"} for rows in backends.values() for row in rows
            ):
                raise ValueError("Missing FireSize backend evidence")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != list(data):
                raise ValueError("Incomplete FireSize calibration cases")
            for row in rows:
                case = row["id"]
                actual = row["output"]
                _validate_output(actual)
                for field, value in actual.items():
                    expected = data[case][field]
                    difference = abs(value - expected)
                    actual_bits, expected_bits = struct.unpack("!QQ", struct.pack("!dd", value, expected))
                    measured = maxima[field]
                    measured["absolute"] = max(measured["absolute"], difference)
                    if expected != 0:
                        measured["relative"] = max(measured["relative"], difference / expected)
                    ulp = abs((actual_bits & MAGNITUDE_MASK) - (expected_bits & MAGNITUDE_MASK))
                    measured["ulp"] = max(measured["ulp"], ulp)
                    measured["nonmatching"] += actual_bits != expected_bits
                    observed[case][field].add(float(value).hex())
                count += len(actual)
    return {
        "source_commit": identity["source_commit"], "ci_run_url": identity["ci_run_url"], "artifacts": signature,
        "comparisons": count, "maxima": maxima,
        "values": {
            case: {field: sorted(values) for field, values in output.items()} for case, output in observed.items()
        },
    }
