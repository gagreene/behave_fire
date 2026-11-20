"""Verify and measure the four approved standalone flat-terrain Spot cases."""

import hashlib
import json
import math
import struct
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

APPROVED_CASES = (
    ("spot-flat-calm-zero-cover", 5, 0, 0),
    ("spot-flat-calm-ten-cover", 5, 0, 10),
    ("spot-flat-wind-zero-cover", 5, 5, 0),
    ("spot-flat-wind-ten-cover", 5, 5, 10),
)
DATASET = Path("tests/data/cpp_reference/spot")
FIELDS = ("flat_distance_ft",)
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
MEASUREMENT_POLICY = dict(approved=False, fields={field: {"kind": "measurement_only"} for field in FIELDS})
OPERATION = "spot_surface"
SOURCES = tuple(f"{name}.{extension}" for name in ("behaveUnits", "CrownFirebrandProcessor", "spot", "spotInputs")
                for extension in ("cpp", "h"))


def _validate_output(output):
    """Require a complete, finite flat-terrain spotting distance in feet.

    :param output: Observed standalone spotting distance.
    :return: None. Raises for invalid distances or incomplete output.
    """
    if not isinstance(output, dict) or set(output) != set(FIELDS):
        raise ValueError("Incomplete Spot output")
    distance = output["flat_distance_ft"]
    if type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0:
        raise ValueError("Invalid Spot flat-terrain spotting distance")


def artifacts(root):
    """Bind a measurement to the exact guarded Spot dataset.

    :param root: Reference checkout root.
    :return: Dataset file names mapped to their SHA256 digests.
    """
    return {name: reference._digest(Path(root) / DATASET / name) for name in FILES}


def load_cases(root):
    """Require exactly the approved inputs in stable case order.

    :param root: Reference checkout root.
    :return: Validated cases.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=reference._signed_int)
    expected = [dict(id=case, operation=OPERATION, flame_ft=flame, wind_mph=wind, cover_ft=cover)
                for case, flame, wind, cover in APPROVED_CASES]
    if json.dumps(rows, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Spot inputs differ from the approved cases")
    return rows


def load_observations(root):
    """Verify the stored raw Spot results and their pinned provenance.

    :param root: Reference checkout root.
    :return: Guarded independent C++ observations.
    """
    directory = Path(root) / DATASET
    manifest = reference._document(directory / "manifest.json")
    schema = reference._document(directory / "schema.json")
    results = read_cpp_results(root, directory / "results.jsonl")
    expected = {name: reference._digest(directory / name) for name in FILES if name != "manifest.json"}
    sources = reference._prepared_build(root)["upstream_source_sha256_portable"]
    trusted = {f"src/behave/{name}": sources[f"src/behave/{name}"] for name in SOURCES}
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["operation"] != OPERATION or manifest["upstream_sha"] != reference.PIN
            or manifest["fields"] != list(FIELDS) or manifest["files"] != expected
            or schema["operation"] != OPERATION or set(schema["output"]) != set(FIELDS)
            or any(type(manifest[key]) is not int or manifest[key] != len(results)
                   for key in ("available", "case_count", "result_count"))
            or manifest["build"]["pinned_checkout_clean"] is not True
            or manifest["build"]["upstream_sources"] != trusted):
        raise ValueError("Incomplete Spot observations or provenance")
    return results


def measure_ports(root, backends):
    """Record unqualified standalone spotting distances from the requested ports.

    :param root: Reference checkout root.
    :param backends: Explicit backend names already configured for measurement.
    :return: Per-backend rows in the approved case order.
    """
    import numpy as np

    cases = load_cases(root)
    measurements = {}
    for backend in backends:
        spot = import_module(f"behave_fire.{backend}").spot
        rows = []
        for case in cases:
            distance = spot.calculate_spotting_from_surface_fire(case["flame_ft"], case["wind_mph"], case["cover_ft"])
            output = {"flat_distance_ft": float(np.asarray(distance).item())}
            _validate_output(output)
            rows.append({"id": case["id"], "output": output})
        measurements[backend] = rows
    return measurements


def measurement_evidence(root, paths, record):
    """Summarize guarded Spot observations from the source-qualified matrix.

    :param root: Checkout with the approved raw dataset.
    :param paths: Full compiler and platform reports for the shared qualification.
    :param record: Source-qualified record that binds each report digest.
    :return: Spot measurement receipt with comparison count, differing bits, and absolute difference in feet.
    """
    expected = load_observations(root)
    signature = artifacts(root)
    guards = dict(negative_cases=8, negative_upstream_entries=0, upstream_trap_exit=86,
                  output_trap_exit=87, duplicate_ids_rejected=True)
    pending = {name: report["sha256"] for name, report in record["reports"].items()}
    if len(pending) != 9 or {name for name in pending if not name.startswith("ports-")} != {"cpp-GNU", "cpp-MSVC"}:
        raise ValueError("Incomplete Spot compiler or platform evidence")
    count, nonmatching, max_abs_ft = 0, 0, 0.0
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw, object_pairs_hook=reference._unique_object,
                            parse_constant=reference._reject_constant, parse_int=reference._signed_int)
        kind = report["kind"]
        if kind not in ("cpp", "ports"):
            raise ValueError("Unknown Spot report kind")
        name = "cpp-" + report["compiler"]["id"] if kind == "cpp" else "ports-" + report["platform_key"]
        if pending.pop(name, None) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Spot report differs from the source-qualified evidence")
        measurement = report["spot"]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != signature:
            raise ValueError("Spot report used a different dataset")
        if kind == "cpp":
            actual = report["protocol"]["spot"]
            if json.dumps(actual, sort_keys=True) != json.dumps(guards, sort_keys=True):
                raise ValueError("Missing Spot entry-guard evidence")
            rows = measurement["observations"]
            validate_cpp_rows(root, rows)
            streams = (rows,)
        else:
            backends = measurement["observations"]
            if not isinstance(backends, dict) or set(backends) != {"python", "rust"} or any(
                not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {"id", "output"}
                                                  for row in rows) for rows in backends.values()
            ):
                raise ValueError("Incomplete Spot backend observations")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Incomplete Spot report case identities")
            for row, original in zip(rows, expected, strict=True):
                _validate_output(row["output"])
                actual_distance = float(row["output"]["flat_distance_ft"])
                original_distance = float(original["output"]["flat_distance_ft"])
                nonmatching += struct.pack("!d", actual_distance) != struct.pack("!d", original_distance)
                max_abs_ft = max(max_abs_ft, abs(actual_distance - original_distance))
                count += 1
    if pending or count != 64:
        raise ValueError("Incomplete Spot measurement evidence")
    return dict(artifacts=signature, comparisons=count, nonmatching=nonmatching, max_abs_ft=max_abs_ft)


def read_cpp_results(root, path):
    """Check the raw results from the pinned Spot driver.

    :param root: Reference checkout root.
    :param path: Raw C++ observations in JSONL format.
    :return: Rows with verified execution and output counts.
    """
    rows = reference._rows(path, parse_int=reference._signed_int)
    validate_cpp_rows(root, rows)
    return rows


def validate_cpp_rows(root, rows):
    """Require one guarded C++ entry and one complete output per case.

    :param root: Reference checkout root.
    :param rows: Parsed raw observations or compiler evidence.
    :return: None. Raises for incomplete or invalid observations.
    """
    if [row["id"] for row in rows] != [case["id"] for case in load_cases(root)]:
        raise ValueError("Spot case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "output_entries", "output"} or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))):
            raise ValueError("Spot row lacks one complete output action")
        _validate_output(row["output"])


def validate_measurement(root, policy, qualification):
    """Require measurement-only Spot policy and validate any shared measurement receipt.

    :param root: Checkout with the approved raw dataset.
    :param policy: Spot section of the canonical policy.
    :param qualification: Verified shared qualification or None while pending.
    :return: None. Raises on an incorrect policy or an invalid receipt.
    """
    if not isinstance(policy, dict) or policy.get("approved") is not False or policy != MEASUREMENT_POLICY:
        raise ValueError("Spot policy must remain measurement-only")
    if qualification is None:
        return
    receipt = qualification.get("spot") if isinstance(qualification, dict) else None
    if (not isinstance(receipt, dict) or set(receipt) != {"artifacts", "comparisons", "nonmatching", "max_abs_ft"}
            or receipt["artifacts"] != artifacts(root) or type(receipt["comparisons"]) is not int
            or receipt["comparisons"] != 64 or type(receipt["nonmatching"]) is not int
            or not 0 <= receipt["nonmatching"] <= 64):
        raise ValueError("Incomplete or stale Spot measurement receipt")
    maximum = receipt["max_abs_ft"]
    if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum < 0:
        raise ValueError("Invalid maximum absolute Spot difference")
    if receipt["nonmatching"] == 0 and maximum > 0:
        raise ValueError("Inconsistent Spot comparison statistics")
