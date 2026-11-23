"""Validate and measure the three approved fine dead fuel moisture cases."""

import hashlib
import json
import math
import struct
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

DATASET = Path("tests/data/cpp_reference/fine_dead_fuel_moisture")
FIELDS = ("moisture_fraction",)
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
MEASUREMENT_POLICY = dict(approved=False, fields={field: {"kind": "measurement_only"} for field in FIELDS})
OPERATION = "fine_dead_fuel_moisture"
SOURCES = ("behaveUnits.cpp", "behaveUnits.h", "fineDeadFuelMoistureTool.cpp", "fineDeadFuelMoistureTool.h")
GUARDS = dict(negative_cases=12, negative_upstream_entries=0, upstream_trap_exit=86,
              output_trap_exit=87, duplicate_ids_rejected=True)
INDEX_FIELDS = ("dry_bulb_i", "rh_i", "slope_i", "aspect_i", "shading_i", "month_i", "elev_i", "time_i")
REQUESTS = tuple(
    dict(id=case_id, operation=OPERATION, **dict(zip(INDEX_FIELDS, indices, strict=True)))
    for case_id, indices in (
        ("exposed-low-slope", (0, 20, 0, 0, 0, 0, 1, 3)),
        ("exposed-high-slope", (0, 20, 1, 0, 0, 0, 1, 3)),
        ("shaded-upper-indices", (5, 20, 1, 3, 1, 2, 2, 5)),
    )
)


def _validate_output(output):
    """Require the single finite, nonnegative moisture output.

    :param output: Observed moisture fraction or its maximum absolute difference.
    :return: None. Raises for invalid values or missing fields.
    """
    if not isinstance(output, dict) or set(output) != set(FIELDS):
        raise ValueError("Incomplete fine dead fuel moisture output")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in output.values()):
        raise ValueError("Invalid fine dead fuel moisture output")


def artifacts(root):
    """Bind a measurement to the approved fine dead fuel moisture fixture files.

    :param root: Reference checkout root.
    :return: File names mapped to SHA256 digests.
    """
    return {name: reference._digest(Path(root) / DATASET / name) for name in FILES}


def load_cases(root):
    """Require the three approved index tuples, in diagnostic order.

    :param root: Reference checkout root.
    :return: Validated case rows.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=reference._signed_int)
    if json.dumps(rows, sort_keys=True, allow_nan=False) != json.dumps(REQUESTS, sort_keys=True):
        raise ValueError("fine dead fuel moisture inputs differ from the approved cases")
    return rows


def load_observations(root):
    """Validate the pinned fine dead fuel moisture results, fixture digests, and upstream closure.

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
            or schema["operation"] != OPERATION or set(schema["output"]) != set(FIELDS)
            or any(type(manifest[key]) is not int or manifest[key] != len(results)
                   for key in ("available", "case_count", "result_count"))
            or manifest["build"]["pinned_checkout_clean"] is not True
            or manifest["build"]["upstream_sources"] != sources):
        raise ValueError("Incomplete fine dead fuel moisture observations or provenance")
    return results


def measure_ports(root, backends):
    """Measure the reviewed moisture indices through each selected public port.

    :param root: Reference checkout root.
    :param backends: Explicit backend names configured for measurement.
    :return: Per-backend observations in approved case order.
    """
    import numpy as np

    cases = load_cases(root)
    measurements = {}
    for backend in backends:
        module = import_module(f"behave_fire.{backend}").fine_dead_fuel_moisture_tool
        rows = []
        for case in cases:
            indices = {field: case[field] for field in INDEX_FIELDS}
            value = module.calculate_fine_dead_fuel_moisture(**indices)
            output = dict(moisture_fraction=float(np.asarray(value).item()))
            _validate_output(output)
            rows.append(dict(id=case["id"], output=output))
        measurements[backend] = rows
    return measurements


def measurement_evidence(root, paths, record):
    """Summarize fine dead fuel moisture values from the complete authenticated matrix.

    :param root: Checkout with the approved fine dead fuel moisture observations.
    :param paths: Compiler and platform report paths.
    :param record: Qualified source record binding the report hashes.
    :return: Measurement receipt with count, differing bits, and maxima per field.
    """
    expected = load_observations(root)
    signature = artifacts(root)
    pending = {name: report["sha256"] for name, report in record["reports"].items()}
    if len(pending) != 9 or {name for name in pending if not name.startswith("ports-")} != {"cpp-GNU", "cpp-MSVC"}:
        raise ValueError("Incomplete fine dead fuel moisture compiler or platform evidence")
    count, nonmatching = 0, 0
    maxima = {field: 0.0 for field in FIELDS}
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw, object_pairs_hook=reference._unique_object,
                            parse_constant=reference._reject_constant, parse_int=reference._signed_int)
        kind = report["kind"]
        if kind not in ("cpp", "ports"):
            raise ValueError("Unknown fine dead fuel moisture report kind")
        name = "cpp-" + report["compiler"]["id"] if kind == "cpp" else "ports-" + report["platform_key"]
        if pending.pop(name, None) != hashlib.sha256(raw).hexdigest():
            raise ValueError("fine dead fuel moisture report differs from the source-qualified evidence")
        measurement = report[OPERATION]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != signature:
            raise ValueError("fine dead fuel moisture report used a different dataset")
        if kind == "cpp":
            if json.dumps(report["protocol"][OPERATION], sort_keys=True) != json.dumps(GUARDS, sort_keys=True):
                raise ValueError("Missing fine dead fuel moisture entry-guard evidence")
            rows = measurement["observations"]
            validate_cpp_rows(root, rows)
            streams = (rows,)
        else:
            backends = measurement["observations"]
            if not isinstance(backends, dict) or set(backends) != {"python", "rust"} or any(
                not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {"id", "output"}
                                                  for row in rows) for rows in backends.values()
            ):
                raise ValueError("Incomplete fine dead fuel moisture backend observations")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Incomplete fine dead fuel moisture report case identities")
            for row, original in zip(rows, expected, strict=True):
                _validate_output(row["output"])
                for field in FIELDS:
                    actual = float(row["output"][field])
                    pinned = float(original["output"][field])
                    nonmatching += struct.pack("!d", actual) != struct.pack("!d", pinned)
                    maxima[field] = max(maxima[field], abs(actual - pinned))
                    count += 1
    if pending or count != 48:
        raise ValueError("Incomplete fine dead fuel moisture measurement evidence")
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
    """Require one upstream and output entry per approved case.

    :param root: Reference checkout root.
    :param rows: Parsed raw observations or compiler evidence.
    :return: None. Raises on incomplete or invalid observations.
    """
    if [row["id"] for row in rows] != [case["id"] for case in load_cases(root)]:
        raise ValueError("fine dead fuel moisture case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "output_entries", "output"} or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))):
            raise ValueError("fine dead fuel moisture row lacks one complete output action")
        _validate_output(row["output"])


def validate_measurement(root, policy, qualification):
    """Require measurement-only policy and verify any matrix receipt.

    :param root: Reference checkout root.
    :param policy: fine dead fuel moisture section of the canonical policy.
    :param qualification: Verified source qualification, or None while measuring.
    :return: None. Raises on invalid policy or receipt.
    """
    if not isinstance(policy, dict) or policy.get("approved") is not False or policy != MEASUREMENT_POLICY:
        raise ValueError("fine dead fuel moisture policy must remain measurement-only")
    if qualification is None:
        return
    receipt = qualification.get(OPERATION) if isinstance(qualification, dict) else None
    if (not isinstance(receipt, dict) or set(receipt) != {"artifacts", "comparisons", "nonmatching", "max_abs"}
            or receipt["artifacts"] != artifacts(root) or type(receipt["comparisons"]) is not int
            or receipt["comparisons"] != 48 or type(receipt["nonmatching"]) is not int
            or not 0 <= receipt["nonmatching"] <= 48):
        raise ValueError("Incomplete or stale fine dead fuel moisture measurement receipt")
    maxima = receipt["max_abs"]
    _validate_output(maxima)
    pinned = load_observations(root)
    streams = receipt["comparisons"] // (len(pinned) * len(FIELDS))
    # When the absolute maximum is zero, only the sign of a pinned zero can differ.
    minimum = sum(value > 0 for value in maxima.values())
    maximum = streams * sum(
        len(pinned) if maxima[field] > 0 else sum(row["output"][field] == 0 for row in pinned)
        for field in FIELDS
    )
    if not minimum <= receipt["nonmatching"] <= maximum:
        raise ValueError("Inconsistent fine dead fuel moisture comparison statistics")
