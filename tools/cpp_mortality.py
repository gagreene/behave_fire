"""Verify and measure the six approved standalone Mortality scorch-height cases."""

import hashlib
import json
import math
import struct
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

APPROVED_CASES = (
    ("scorch-zero-calm-77", 0, 0, 77),
    ("scorch-zero-wind-77", 0, 2, 77),
    ("scorch-zero-calm-14", 0, 0, 14),
    ("scorch-one-calm-77", 1, 0, 77),
    ("scorch-one-wind-77", 1, 2, 77),
    ("scorch-one-calm-14", 1, 0, 14),
)
DATASET = Path("tests/data/cpp_reference/mortality")
FIELDS = ("scorch_height_ft",)
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
EXACT_FIELDS = {field: {"kind": "exact_binary64"} for field in FIELDS}
SOURCES = tuple(f"{name}.{extension}" for name in (
    "behaveUnits", "canopy_coefficient_table", "mortality", "mortality_equation_table",
    "mortality_inputs", "species_master_table",
) for extension in ("cpp", "h"))


def _validate_output(output):
    """Require a complete, finite scorch height in feet.

    :param output: Observed standalone scorch height.
    :return: None. Raises for invalid heights or incomplete output.
    """
    if not isinstance(output, dict) or set(output) != set(FIELDS):
        raise ValueError("Incomplete Mortality output")
    height = output["scorch_height_ft"]
    if type(height) not in (int, float) or not math.isfinite(height) or height < 0:
        raise ValueError("Invalid Mortality scorch height")


def artifacts(root):
    """Bind a measurement to the exact guarded Mortality dataset.

    :param root: Reference checkout root.
    :return: Dataset file names mapped to their SHA256 digests.
    """
    return {name: reference._digest(Path(root) / DATASET / name) for name in FILES}


def assert_matches_cpp(actual, expected, policy):
    """Compare one finite scorch height as exact binary64, including signed zero.

    :param actual: Observed standalone scorch height from a selected port.
    :param expected: Pinned C++ result in feet.
    :param policy: Validated Mortality comparison policy.
    :return: None. Raises without qualification or on a mismatch.
    """
    if policy.get("approved") is not True:
        raise reference.UnapprovedCppPolicy("Mortality comparisons require complete qualification")
    if policy.get("fields") != EXACT_FIELDS:
        raise ValueError("Unexpected Mortality comparison fields")
    _validate_output(actual)
    _validate_output(expected)
    field = FIELDS[0]
    if struct.pack("!d", actual[field]) != struct.pack("!d", expected[field]):
        raise AssertionError(f"{field}: port={actual[field]!r}, C++={expected[field]!r}")


def exact_evidence(root, paths, record):
    """Verify all 96 comparisons from the source-qualified reports.

    :param root: Checkout with the approved raw dataset.
    :param paths: The original compiler and platform reports.
    :param record: Source qualification binding the exact report digests.
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
            raise ValueError("Unknown Mortality report kind")
        name = "cpp-" + report["compiler"]["id"] if kind == "cpp" else "ports-" + report["platform_key"]
        if pending.pop(name, None) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Mortality report differs from the source-qualified evidence")
        measurement = report["mortality"]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != signature:
            raise ValueError("Mortality report used a different dataset")
        if kind == "cpp":
            actual = report["protocol"]["mortality"]
            if json.dumps(actual, sort_keys=True) != json.dumps(controls, sort_keys=True):
                raise ValueError("Missing Mortality entry-guard evidence")
            rows = measurement["observations"]
            validate_cpp_rows(root, rows)
            streams = (rows,)
        else:
            backends = measurement["observations"]
            if set(backends) != {"python", "rust"} or any(
                set(row) != {"id", "output"} for rows in backends.values() for row in rows
            ):
                raise ValueError("Incomplete Mortality backend observations")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Incomplete Mortality report case identities")
            for row, original in zip(rows, expected, strict=True):
                assert_matches_cpp(row["output"], original["output"], policy)
                count += 1
    if pending or count != 96:
        raise ValueError("Incomplete exact Mortality evidence")
    return dict(artifacts=signature, comparisons=count, nonmatching=0)


def load_cases(root):
    """Require exactly the approved inputs in stable case order.

    :param root: Reference checkout root.
    :return: Validated cases.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=reference._signed_int)
    expected = [dict(id=case, operation="mortality", intensity_btu_ft_s=intensity, wind_mph=wind,
                     air_temperature_f=air) for case, intensity, wind, air in APPROVED_CASES]
    if json.dumps(rows, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Mortality inputs differ from the approved cases")
    return rows


def load_observations(root):
    """Verify the stored raw Mortality results and their pinned provenance.

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
    if (manifest["schema_version"] != 1 or manifest["operation"] != "mortality"
            or manifest["upstream_sha"] != reference.PIN or manifest["fields"] != list(FIELDS)
            or manifest["files"] != expected or schema["operation"] != "mortality"
            or set(schema["output"]) != set(FIELDS)
            or any(type(manifest[key]) is not int or manifest[key] != len(results)
                   for key in ("available", "case_count", "result_count"))
            or manifest["build"]["pinned_checkout_clean"] is not True
            or manifest["build"]["upstream_sources"] != trusted):
        raise ValueError("Incomplete Mortality observations or provenance")
    return results


def measure_ports(root, backends):
    """Record unqualified standalone scorch heights from the requested ports.

    :param root: Reference checkout root.
    :param backends: Explicit backend names already configured for measurement.
    :return: Per-backend rows in the approved case order.
    """
    import numpy as np

    cases = load_cases(root)
    measurements = {}
    for backend in backends:
        mortality = import_module(f"behave_fire.{backend}").mortality
        rows = []
        for case in cases:
            height = mortality.calculate_scorch_height(
                case["intensity_btu_ft_s"], case["wind_mph"], case["air_temperature_f"],
            )
            output = {"scorch_height_ft": float(np.asarray(height).item())}
            _validate_output(output)
            rows.append({"id": case["id"], "output": output})
        measurements[backend] = rows
    return measurements


def read_cpp_results(root, path):
    """Check the raw results from the pinned Mortality driver.

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
        raise ValueError("Mortality case/result identity mismatch")
    for row in rows:
        if (set(row) != {"id", "status", "upstream_entries", "output_entries", "output"} or row["status"] != "ok"
                or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))):
            raise ValueError("Mortality row lacks one complete output action")
        _validate_output(row["output"])


def validate_policy(root, policy, qualification):
    """Require the approved rule and an exact source-qualified receipt.

    :param root: Reference checkout root.
    :param policy: Mortality section of the canonical policy.
    :param qualification: Shared qualification already verified by the caller.
    :return: None. Raises on missing, malformed, or stale evidence.
    """
    if (not isinstance(policy, dict) or set(policy) != {"approved", "fields"}
            or type(policy["approved"]) is not bool or policy["fields"] != EXACT_FIELDS):
        raise ValueError("Incomplete exact Mortality comparison policy")
    if policy["approved"]:
        actual = qualification.get("mortality") if isinstance(qualification, dict) else None
        expected = dict(artifacts=artifacts(root), comparisons=96, nonmatching=0)
        if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError("Incomplete or stale Mortality qualification")
