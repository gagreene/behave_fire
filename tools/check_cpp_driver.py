"""Exercise driver boundaries without ever allowing an unsafe upstream call."""

import argparse
import json
from pathlib import Path
import subprocess

from cpp_reference import ROOT, _digest, _document, _unique_object, load_reference


def _read_lines(path):
    """Decode protocol records, including null IDs after malformed input.

    :param path: Driver output path.
    :return: Parsed rows in input order.
    """
    return [json.loads(line, object_pairs_hook=_unique_object) for line in path.read_text().splitlines()]


def _run(driver, directory, name, input_path, flag, expected_exit):
    """Run one bounded protocol check and preserve its exact process evidence.

    :param driver: Explicit driver executable.
    :param directory: Scratch evidence directory.
    :param name: Safe evidence stem.
    :param input_path: JSONL input path.
    :param flag: Optional upstream/geometry poison flag.
    :param expected_exit: Expected process status.
    :return: Output path after status verification.
    """
    output = directory / (name + ".jsonl")
    arguments = [str(driver), str(input_path), str(output)] + ([flag] if flag else [])
    process = subprocess.run(arguments, cwd=directory, capture_output=True, check=False)
    (directory / (name + ".stdout")).write_bytes(process.stdout)
    (directory / (name + ".stderr")).write_bytes(process.stderr)
    (directory / (name + ".exit")).write_text(str(process.returncode) + "\n")
    if process.returncode != expected_exit:
        raise ValueError(f"{name}: exit {process.returncode}, expected {expected_exit}")
    return output


def check_driver(driver, directory, root=ROOT):
    """Verify exact replay, guarded rejections, and active entry traps.

    :param driver: Out-of-tree executable supplied explicitly by the operator.
    :param directory: Retained scratch directory.
    :param root: Candidate repository root.
    :return: Control counts without a cross-implementation parity verdict.
    """
    driver, directory, root = Path(driver).resolve(), Path(directory).resolve(), Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data = load_reference(root)
    if _digest(driver) != data["manifest"]["build_inputs"]["driver_sha256"]:
        raise ValueError("Driver binary does not match the recorded build")
    source = root / "tests/data/cpp_reference"
    negative = _run(driver, directory, "negative", source / "negative-inputs.jsonl", "--forbid-upstream", 0)
    actual, expected = _read_lines(negative), _document(source / "negative-expected.json")
    if actual != expected or not actual or any(row["upstream_entries"] != 0 or "output" in row for row in actual):
        raise ValueError("An unsafe request bypassed its guarded rejection")
    cases = source / "surface/cases.jsonl"
    _run(driver, directory, "upstream-positive-trap", cases, "--forbid-upstream", 86)
    if b"UPSTREAM_ENTRY_BLOCKED" not in (directory / "upstream-positive-trap.stderr").read_bytes():
        raise ValueError("Upstream trap was not active")
    unavailable = [case for case, result in zip(data["cases"], data["results"])
                   if result["status"] == "unavailable" or case["fuel_model"] in (0, 91, 255)]
    unavailable_path = directory / "undefined-input.jsonl"
    unavailable_path.write_text("".join(json.dumps(row) + "\n" for row in unavailable), encoding="utf-8")
    output = _run(driver, directory, "geometry-negative", unavailable_path, "--forbid-geometry", 0)
    rows = _read_lines(output)
    if len(rows) != len(unavailable) or any(row["status"] != "unavailable" or row["geometry_entries"] != 0
                                           or row["finding"] != "F-013" or "output" in row for row in rows):
        raise ValueError("A skipped calculation read or invented geometry")
    _run(driver, directory, "geometry-positive-trap", cases, "--forbid-geometry", 87)
    if b"GEOMETRY_ENTRY_BLOCKED" not in (directory / "geometry-positive-trap.stderr").read_bytes():
        raise ValueError("Geometry trap was not active")
    repeated = _run(driver, directory, "repeat", cases, None, 0)
    if repeated.read_bytes() != (source / "surface/results.jsonl").read_bytes():
        raise ValueError("Exact-driver replay differs from stored observations")
    indexed = {row["id"]: row for row in _read_lines(repeated)}
    if (indexed["fm124-state-A1"]["output"] != indexed["fm124-state-A2"]["output"]
            or indexed["fm124-state-A1"]["output"] != indexed["fm124-angle0"]["output"]
            or indexed["fm0-state-unavailable"]["status"] != "unavailable"):
        raise ValueError("State sequence failed to restore the fresh full-run result")
    duplicate = directory / "duplicate-input.jsonl"
    duplicate.write_text((json.dumps(data["cases"][0]) + "\n") * 2, encoding="utf-8")
    duplicate_rows = _read_lines(_run(driver, directory, "duplicate", duplicate, None, 0))
    if (len(duplicate_rows) != 2 or duplicate_rows[0] != data["results"][0]
            or duplicate_rows[1].get("error") != "duplicate_id" or duplicate_rows[1]["upstream_entries"] != 0):
        raise ValueError("Duplicate case ID reached upstream twice")
    summary = dict(negative_cases=len(actual), negative_upstream_entries=0, unavailable_cases=len(rows),
                   unavailable_geometry_entries=0, upstream_trap_exit=86, geometry_trap_exit=87,
                   replay_cases=len(data["cases"]), replay_byte_identical=True, duplicate_ids_rejected=True,
                   state_sequence_checked=True, scientific_parity_approved=False)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    """Run controls only when an explicit executable and evidence path are supplied.

    :return: None; writes evidence and prints bounded counts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_driver(args.driver, args.out), indent=2))


if __name__ == "__main__":
    main()
