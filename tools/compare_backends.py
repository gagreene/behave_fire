"""Compare explicit native execution with the frozen, normalized Python oracle."""

import argparse
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())
SUPPORT = Path(__file__).resolve().parents[1] / "tests"
sys.path[:0] = [str(ROOT / "src"), str(SUPPORT)]
import parity_support


def _metrics(actual, expected, summary):
    """Collect strict structural differences and finite numeric error maxima.

    :param actual: Native result or receiver field.
    :param expected: Frozen Python representation.
    :param summary: Mutable per-domain report.
    :return: None; report counters and error maxima are updated.
    """
    if isinstance(actual, dict) and isinstance(expected, dict) and "dtype" not in expected:
        summary["missing_fields"] += len(set(expected) - set(actual))
    if not isinstance(expected, dict) or "dtype" not in expected:
        return
    target = parity_support.decode(expected)
    actual = np.asanyarray(actual)
    summary["array_fields"] += 1
    summary["dtype_differences"] += int(actual.dtype != target.dtype)
    summary["shape_differences"] += int(actual.shape != target.shape)
    if actual.shape != target.shape:
        return
    summary["mask_differences"] += int(not np.array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(target)))
    left = np.asarray(np.ma.getdata(actual), dtype=float)
    right = np.asarray(np.ma.getdata(target), dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.any(finite):
        absolute = np.abs(left[finite] - right[finite])
        summary["maximum_absolute_error"] = max(summary["maximum_absolute_error"], float(np.max(absolute)))
        nonzero = right[finite] != 0
        if np.any(nonzero):
            summary["maximum_relative_error"] = max(summary["maximum_relative_error"], float(np.max(absolute[nonzero] / np.abs(right[finite][nonzero]))))


def main(argv=None):
    """Run every acceptance fixture without rewriting its oracle.

    :param argv: Optional command-line arguments.
    :return: Zero for complete parity; one if any scenario differs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    scenarios = []
    for path in sorted((ROOT / "tests/data/rust_port").glob("*.json")):
        for case in json.loads(path.read_text(encoding="utf-8" )).get("cases", []):
            if case.get("policy", "acceptance") == "acceptance":
                scenarios.append((path.stem, case))
    random.Random(arguments.seed).shuffle(scenarios)
    report = {"seed": arguments.seed, "oracle": "Frozen reference cases with float16/internal-half widening; four exact reviewed invalid-int32-cast inputs use live explicit Python only when the host cast differs; frozen cast differences remain diagnostics", "domains": {}, "failures": [], "platform_cast_diagnostics": []}
    original_assertion = parity_support.assert_result
    for domain, case in scenarios:
        summary = report["domains"].setdefault(domain, dict(scenarios=0, passed=0, array_fields=0, missing_fields=0, dtype_differences=0, shape_differences=0, mask_differences=0, maximum_absolute_error=0.0, maximum_relative_error=0.0))
        summary["scenarios"] += 1

        def observed(actual, expected):
            """Record this scenario's metrics before its ordinary strict assertion.

            :param actual: Native result.
            :param expected: Frozen Python result.
            :return: None if the result agrees; otherwise raise AssertionError.
            """
            _metrics(actual, expected, summary)
            original_assertion(actual, expected)

        parity_support.assert_result = observed
        try:
            from rust_reference_platform import reference_outputs
            outputs = reference_outputs(case, parity_support.encode_value)
            if outputs is not None:
                current, original_cast = outputs
                report["platform_cast_diagnostics"].append({
                    "domain": domain, "scenario": case["id"],
                    "frozen_output": case["output"], "current_python_output": current,
                    "original_cast_replay_matches_frozen": parity_support.contracts_match(original_cast, case["output"], True),
                    "current_python_matches_frozen": parity_support.contracts_match(current, case["output"], True),
                })
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                parity_support.run_case(case)
            summary["passed"] += 1
        except Exception as error:
            report["failures"].append({"domain": domain, "scenario": case["id"], "error": f"{type(error).__name__}: {error}"})
        finally:
            parity_support.assert_result = original_assertion
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    print(text)
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
