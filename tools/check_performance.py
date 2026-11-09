"""Check complete, paired public-call performance evidence against release targets."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from assemble_allocation_evidence import assemble


def _allocation_errors(report, requirements, identity):
    """Check native kernel and encoder allocation evidence.

    :param report: Candidate measurements and separate probe evidence.
    :param requirements: Required routes and datasets.
    :param identity: Expected native source and extension hashes.
    :return: Allocation evidence errors.
    """
    errors = []
    if not requirements.get("require_allocations", True):
        return errors
    for dataset in requirements["datasets"]:
        for operation in requirements["operations"]:
            matches = [row for row in report.get("allocations", [])
                       if (row.get("dataset"), row.get("operation")) == (dataset, operation)]
            if len(matches) != 1:
                errors.append(f"{dataset}/{operation}: require one native allocation report")
                continue
            row = matches[0]
            if requirements.get('binding_api_version') == 4 and row.get('evidence', {}).get('manifest', {}).get('binding_api_version') != 4:
                errors.append(f'{dataset}/{operation}: allocation manifest must identify binding API 4')
            if (row.get("source_hash"), row.get("extension_hash")) != identity:
                errors.append(f"{dataset}/{operation}: allocation provenance mismatch")
            try:
                certified = assemble(row.get("evidence", {}), requirements)["allocations"]
                expected = next(item for item in certified if
                                (item["dataset"], item["operation"]) == (dataset, operation))
                if row != expected:
                    errors.append(f"{dataset}/{operation}: allocation row differs from raw evidence")
            except (ValueError, StopIteration):
                errors.append(f"{dataset}/{operation}: invalid or missing scoped allocation evidence")
            for key in ("kernel_per_cell", "encoder_per_cell"):
                if type(row.get(key)) not in (int, float) or row[key] != 0:
                    errors.append(f"{dataset}/{operation}: {key} must be measured as zero")
    return errors


def _bootstrap(python, native, requirements):
    """Compute a reproducible interval for the paired ratio of medians.

    :param python: Explicit reference durations in paired block order.
    :param native: Native durations in the same block order.
    :param requirements: Resample count, confidence level, and seed.
    :return: Median speed ratio and its lower and upper interval endpoints.
    """
    rng = random.Random(requirements["seed"])
    ratios = []
    for _ in range(requirements.get("bootstrap_resamples", 5000)):
        indexes = rng.choices(range(len(python)), k=len(python))
        ratios.append(statistics.median(python[i] for i in indexes)
                      / statistics.median(native[i] for i in indexes))
    ratios.sort()
    tail = (1 - requirements.get("confidence", 0.95)) / 2
    return {"ratio": statistics.median(python) / statistics.median(native),
            "lower": _quantile(ratios, tail), "upper": _quantile(ratios, 1 - tail)}


def _duration(row, warmed, requirements):
    """Read a finite positive first-call or warmed worker duration.

    :param row: One fresh-process worker observation.
    :param warmed: Whether to use its warmed-call median.
    :param requirements: Minimum warmed-call count.
    :return: A positive duration, or None for invalid evidence.
    """
    values = row.get("warm_seconds") if warmed else [row.get("first_seconds")]
    if not isinstance(values, list) or len(values) < (requirements["warm_calls"] if warmed else 1):
        return None
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        return None
    return statistics.median(values)


def _group_errors(rows, key, report, requirements, identity):
    """Evaluate paired timing, memory, and full-result validation for one case.

    :param rows: All unprofiled worker observations.
    :param key: Dataset, operation, cell count, and native worker request.
    :param report: Candidate measurements and separate correctness evidence.
    :param requirements: Confirmed target and measurement rules.
    :param identity: Expected native source and extension hashes.
    :return: Case statistics and a list of failures.
    """
    dataset, operation, cells, threads = key
    label = f"{dataset}/{operation}/{cells}/{threads}"
    case = {"dataset": dataset, "operation": operation, "cells": cells, "threads": threads}
    python = [row for row in rows if (row.get("dataset"), row.get("operation"), row.get("cells"),
                                      row.get("backend")) == (dataset, operation, cells, "python")]
    native = [row for row in rows if (row.get("dataset"), row.get("operation"), row.get("cells"),
                                      row.get("backend"), row.get("threads"))
              == (dataset, operation, cells, "rust", threads)]
    python_blocks = [row.get("block") for row in python]
    native_blocks = [row.get("block") for row in native]
    errors = []
    if (len(python) < requirements["minimum_pairs"] or set(python_blocks) != set(native_blocks)
            or len(set(python_blocks)) != len(python) or len(set(native_blocks)) != len(native)):
        return case, [f"{label}: missing or duplicate paired observations"]
    if any(type(block) is not int for block in python_blocks):
        return case, [f"{label}: invalid block identity"]
    ordered_python = sorted(python, key=lambda row: row["block"])
    ordered_native = sorted(native, key=lambda row: row["block"])
    for name in ("numpy", "numpy_build", "python", "platform", "seed", "source_hash"):
        values = {row.get(name) for row in python + native}
        if len(values) != 1 or None in values or "" in values:
            errors.append(f"{label}: inconsistent or missing {name}")
    if any(row.get("profile") is not False for row in python + native):
        errors.append(f"{label}: profiled or unspecified timing mode")
    expected = {"dataset": dataset, "operation": operation, "cells": cells, "threads": threads,
                "numpy": python[0].get("numpy"), "numpy_build": python[0].get("numpy_build"),
                "source_hash": identity[0], "extension_hash": identity[1], "seed": requirements["seed"]}
    validation = [row for row in report.get("validations", [])
                  if all(row.get(name) == value for name, value in expected.items())]
    if len(validation) != 1 or validation[0].get("passed") is not True:
        errors.append(f"{label}: missing matching full-output validation")
    target = _target(operation, cells, threads, requirements)
    case["target"] = target
    for warmed in (False, True):
        name = "warmed" if warmed else "first"
        left = [_duration(row, warmed, requirements) for row in ordered_python]
        right = [_duration(row, warmed, requirements) for row in ordered_native]
        if None in left or None in right:
            errors.append(f"{label}/{name}: invalid or missing durations")
            continue
        interval = _bootstrap(left, right, requirements)
        case[name] = interval
        if any(not math.isfinite(value) for value in interval.values()) or interval["lower"] < target:
            errors.append(f"{label}/{name}: lower speed interval {interval['lower']:.6g} < {target:g}")
    if cells >= 100_000:
        peaks = [row.get("peak_bytes") for row in python + native]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in peaks):
            errors.append(f"{label}: missing peak memory evidence")
        else:
            memory_ratio = statistics.median(row["peak_bytes"] for row in native) / statistics.median(
                row["peak_bytes"] for row in python)
            case["peak_memory_ratio"] = memory_ratio
            if memory_ratio > 1:
                errors.append(f"{label}: native median peak process memory exceeds Python")
    case["passed"] = not errors
    return case, errors


def _quantile(values, fraction):
    """Interpolate a quantile from ordered scalar observations.

    :param values: Nonempty sorted observations.
    :param fraction: Quantile between zero and one.
    :return: Linearly interpolated quantile.
    """
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _scalar_errors(report, requirements):
    """Apply the allowed one-cell regression budget to the old native baseline.

    :param report: Candidate and pre-optimization native worker observations.
    :param requirements: Dataset, operation, warmed-call rules, and independently declared baseline identity.
    :return: Missing baseline or scalar regression failures.
    """
    errors = []
    if not requirements.get("require_scalar_baseline", True):
        return errors
    identity = requirements.get("baseline_identity")
    if (not isinstance(identity, dict)
            or any(not isinstance(identity.get(key), str) or not identity[key].strip()
                   for key in ("source_hash", "extension_hash"))):
        return ["Scalar baseline requires a declared nonempty pre-optimization source/binary identity"]
    for dataset in requirements["datasets"]:
        for operation in requirements["operations"]:
            samples = []
            for name in ("baseline", "records"):
                samples.append([row for row in report.get(name, [])
                                if (row.get("dataset"), row.get("operation"), row.get("cells"),
                                    row.get("backend"), row.get("threads"))
                                == (dataset, operation, 1, "rust", 1)])
            if any(len(rows) < 3 for rows in samples):
                errors.append(f"{dataset}/{operation}: missing one-cell baseline or candidate")
                continue
            if any(any(row.get(key) != identity[key] for key in ("source_hash", "extension_hash"))
                   for row in samples[0]):
                errors.append(f"{dataset}/{operation}: scalar baseline source/binary identity mismatch")
                continue
            if any(len({row.get("block") for row in rows}) != len(rows) for rows in samples):
                errors.append(f"{dataset}/{operation}: duplicate scalar process blocks")
            if any(row.get("profile") is not False for rows in samples for row in rows):
                errors.append(f"{dataset}/{operation}: scalar baseline must be unprofiled")
            for name in ("numpy", "numpy_build", "python", "platform", "seed"):
                values = {row.get(name) for rows in samples for row in rows}
                if len(values) != 1 or None in values or "" in values:
                    errors.append(f"{dataset}/{operation}: scalar baseline {name} mismatch")
            candidate = samples[1][0]
            expected = {"dataset": dataset, "operation": operation, "cells": 1, "threads": 1,
                        "numpy": candidate.get("numpy"), "numpy_build": candidate.get("numpy_build"),
                        "source_hash": candidate.get("source_hash"),
                        "extension_hash": candidate.get("extension_hash"), "seed": requirements["seed"]}
            validations = [row for row in report.get("validations", [])
                           if all(row.get(name) == value for name, value in expected.items())]
            if len(validations) != 1 or validations[0].get("passed") is not True:
                errors.append(f"{dataset}/{operation}: missing scalar full-output validation")
            for warmed in (False, True):
                values = [[_duration(row, warmed, requirements) for row in rows] for rows in samples]
                if any(None in row for row in values):
                    errors.append(f"{dataset}/{operation}: invalid scalar durations")
                    continue
                old, new = (statistics.median(row) for row in values)
                if new > old + max(old * 0.1, 0.00025):
                    errors.append(f"{dataset}/{operation}: one-cell regression exceeds 10% or 0.25 ms budget")
    return errors


def _target(operation, cells, threads, requirements):
    """Select the owner-confirmed target without averaging different routes.

    :param operation: Surface or Crown route.
    :param cells: Logical cell count.
    :param threads: Native worker request.
    :param requirements: Numeric targets.
    :return: Required Python/native ratio.
    """
    if cells == 10_000:
        return requirements["targets"]["small_typed" if operation.startswith("typed-") else "small_common"]
    return requirements["targets"]["large_four" if threads == 4 else "large_serial"]


def check_report(report, requirements, *, artifact_root=None):
    """Assess complete evidence without running science or changing a target.

    :param report: Parsed timing, correctness, allocation, and baseline evidence.
    :param requirements: Accepted measurement/target contract and independently declared baseline identity.
    :param artifact_root: Optional execution root for complete captured output arrays.
    :return: JSON-compatible pass/fail, case statistics, and failure explanations.
    """
    from performance_contract import ROOT, contract_errors

    errors = contract_errors(report, requirements, ROOT if artifact_root is None else artifact_root)
    cases = []
    if report.get("errors"):
        errors.append("The campaign contains failed workers or correctness checks")
    if report.get("schema_version") != 1 or requirements.get("schema_version") != 1:
        errors.append("Unsupported or absent performance evidence schema")
    if report.get("profile") is not False or report.get("preset") != "acceptance":
        errors.append("Only unprofiled acceptance evidence can certify performance")
    if type(report.get("physical_cores")) is not int or report["physical_cores"] < 4:
        errors.append("Four-worker claims require a declared host with at least four physical cores")
    rows = report.get("records", [])
    for name in ("numpy", "numpy_build", "python", "platform", "source_hash", "seed"):
        values = {row.get(name) for row in rows}
        if len(values) != 1 or None in values or "" in values:
            errors.append(f"Candidate observations require one nonempty {name}")
    if any(row.get("seed") != requirements["seed"] for row in rows):
        errors.append("Candidate inputs do not use the declared reproducible seed")
    identities = {(row.get("source_hash"), row.get("extension_hash"))
                  for row in rows if row.get("backend") == "rust"}
    if len(identities) != 1 or any(not value for identity in identities for value in identity):
        errors.append("Native observations require one nonempty source/binary identity")
    if errors:
        return {"passed": False, "errors": errors, "cases": cases}
    identity = next(iter(identities))
    for dataset in requirements["datasets"]:
        for operation in requirements["operations"]:
            for cells in requirements["sizes"]:
                for threads in ([1] if cells == 10_000 else [1, 4]):
                    case, failures = _group_errors(rows, (dataset, operation, cells, threads),
                                                  report, requirements, identity)
                    cases.append(case)
                    errors.extend(failures)
    errors.extend(_allocation_errors(report, requirements, identity))
    errors.extend(_scalar_errors(report, requirements))
    return {"passed": not errors, "errors": errors, "cases": cases}


def main():
    """Read report/requirements files and return a failing status for any unmet gate.

    :return: None; prints JSON evidence assessment and exits with its status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=Path(__file__).with_name("performance_requirements.json"))
    parser.add_argument("--baseline-source-hash", help="Independently recorded pre-optimization source SHA-256")
    parser.add_argument("--baseline-extension-hash", help="Independently recorded pre-optimization native-binary SHA-256")
    args = parser.parse_args()
    try:
        requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
        if args.baseline_source_hash is not None or args.baseline_extension_hash is not None:
            requirements["baseline_identity"] = {"source_hash": args.baseline_source_hash,
                                                 "extension_hash": args.baseline_extension_hash}
        result = check_report(json.loads(args.results.read_text(encoding="utf-8")), requirements)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = {"passed": False, "errors": [f"Invalid performance evidence: {error}"], "cases": []}
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
