"""Verify and measure the four approved standalone Ignite reference cases."""

import hashlib
import json
import math
import struct
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
EXACT_FIELDS = {field: {"kind": "exact_binary64"} for field in FIELDS}
SOURCES = ("behaveUnits.cpp", "behaveUnits.h", "ignite.cpp", "ignite.h", "igniteInputs.cpp", "igniteInputs.h")


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


def assert_matches_cpp(actual, expected, policy):
    """Compare both finite Ignite outputs as exact binary64 values.

    :param actual: Observed Ignite result from a selected port.
    :param expected: Pinned C++ result in the same units.
    :param policy: Validated Ignite comparison policy.
    :return: None. Raises without approval or on a result mismatch.
    """
    if policy.get("approved") is not True:
        raise reference.UnapprovedCppPolicy("Ignite comparisons require complete qualification")
    check_exact(actual, expected, policy)


def check_exact(actual, expected, policy):
    """Apply the established exact binary64 rule whatever the release approval state.

    The full local check calls this directly, so a withdrawn qualification never hides a
    numerical regression; ``assert_matches_cpp`` adds the release approval requirement.

    :param actual: Observed Ignite result from a selected port.
    :param expected: Pinned C++ result in the same units.
    :param policy: Validated Ignite comparison policy. Its fields must be the established exact rules.
    :return: Number of exactly compared fields; raises on a rule change or a mismatch.
    """
    if policy.get("fields") != EXACT_FIELDS:
        raise ValueError("Unexpected Ignite comparison fields")
    _validate_output(actual)
    _validate_output(expected)
    for field in FIELDS:
        if struct.pack("!d", actual[field]) != struct.pack("!d", expected[field]):
            raise AssertionError(f"{field}: port={actual[field]!r}, C++={expected[field]!r}")
    return len(FIELDS)


def exact_evidence(root, paths, record):
    """Verify the Ignite portion of the nine source-qualified reports.

    :param root: Checkout with the approved raw dataset.
    :param paths: The full original compiler and platform reports.
    :param record: Source-qualified record binding the exact report digests.
    :return: Compact receipt for the shared qualification.
    """
    expected = load_observations(root)
    signature = artifacts(root)
    controls = dict(negative_cases=8, negative_upstream_entries=0, upstream_trap_exit=86,
                    output_trap_exit=87, duplicate_ids_rejected=True)
    pending = {name: report["sha256"] for name, report in record["reports"].items()}
    policy = dict(approved=True, fields=EXACT_FIELDS)
    count = 0
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw, object_pairs_hook=reference._unique_object,
                            parse_constant=reference._reject_constant, parse_int=reference._signed_int)
        kind = report["kind"]
        if kind not in ("cpp", "ports"):
            raise ValueError("Unknown Ignite report kind")
        name = "cpp-" + report["compiler"]["id"] if kind == "cpp" else "ports-" + report["platform_key"]
        if pending.pop(name, None) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Ignite report differs from the source-qualified evidence")
        measurement = report["ignite"]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != signature:
            raise ValueError("Ignite report used a different dataset")
        if kind == "cpp":
            actual = report["protocol"]["ignite"]
            if json.dumps(actual, sort_keys=True) != json.dumps(controls, sort_keys=True):
                raise ValueError("Missing Ignite entry-guard evidence")
            rows = measurement["observations"]
            validate_cpp_rows(root, rows)
            streams = [rows]
        else:
            backends = measurement["observations"]
            if set(backends) != {"python", "rust"} or any(
                set(row) != {"id", "output"} for rows in backends.values() for row in rows
            ):
                raise ValueError("Incomplete Ignite backend observations")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Incomplete Ignite report case identities")
            for row, original in zip(rows, expected, strict=True):
                assert_matches_cpp(row["output"], original["output"], policy)
                count += len(FIELDS)
    if pending or count != 128:
        raise ValueError("Incomplete exact Ignite evidence")
    return dict(artifacts=signature, comparisons=count, nonmatching=0)


def load_cases(root):
    """Require exactly the approved inputs in their stable case order.

    :param root: Reference checkout root.
    :return: Validated cases.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=reference._signed_int)
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
        raise ValueError("Ignite case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "output_entries", "output"} or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))):
            raise ValueError("Ignite row lacks one complete output action")
        _validate_output(row["output"])


def validate_policy(root, policy, qualification):
    """Require the approved rule and a complete compact Ignite proof.

    :param root: Reference checkout root.
    :param policy: Ignite section of the canonical policy.
    :param qualification: Shared qualification already verified by the caller.
    :return: None. Raises on missing or malformed evidence.
    """
    if (not isinstance(policy, dict) or set(policy) != {"approved", "fields"}
            or type(policy["approved"]) is not bool or policy["fields"] != EXACT_FIELDS):
        raise ValueError("Incomplete exact Ignite comparison policy")
    if policy["approved"]:
        proof = qualification.get("ignite") if isinstance(qualification, dict) else None
        expected = dict(artifacts=artifacts(root), comparisons=128, nonmatching=0)
        if json.dumps(proof, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError("Incomplete or stale Ignite qualification")
