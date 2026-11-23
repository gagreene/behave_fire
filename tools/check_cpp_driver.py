"""Exercise driver boundaries without ever allowing an unsafe upstream call."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cpp_mortality as mortality
import cpp_safety as safety
import cpp_spot as spot
from cpp_fire_size import DATASET, load_cases
from cpp_ignite import DATASET as IGNITE_DATASET
from cpp_ignite import load_cases as load_ignite_cases
from cpp_reference import PIN, ROOT, _digest, _document, _prepared_build, _text_digest, _unique_object, load_reference


def _check_custom(driver, directory, data, root):
    """Qualify new custom inputs without weakening any original driver control.

    :param driver: Source-matched executable.
    :param directory: Existing evidence directory.
    :param data: Verified original and custom observations.
    :param root: Source checkout root.
    :return: Explicit custom rejection, trap, duplicate, and state counts.
    """
    inputs = root / "tests/data/cpp_reference/surface_custom/cases.jsonl"
    cases = data["controls"]["cases"]
    base = cases[0]
    invalid = []
    for index, (field, value) in enumerate([
            ("fuel_mode", "automatic"), ("fuel_model", -1), ("fuel_model", 0), ("fuel_model", 1),
            ("fuel_model", 256), ("fuel_model", 2147483647), ("fuel_model", 110.),
            ("moe_percent", 0), ("moe_percent", -1), ("moe_percent", .24), ("moe_percent", 101),
            ("moe_percent", "NaN"), ("moe_percent", "Inf"), ("moe_percent", "-Inf"),
            ("moe_percent", True), ("moe_percent", "24")]):
        row = dict(base, id=f"custom-negative-{index}")
        row[field] = value
        invalid.append(row)
    for field in ("fuel_mode", "moe_percent"):
        row = dict(base, id="custom-missing-" + field)
        del row[field]
        invalid.append(row)
    path = directory / "custom-negative-inputs.jsonl"
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in invalid), encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "custom-negative", path, "--forbid-upstream", 0))
    if len(rows) != 18 or any(row["status"] != "rejected" or row["upstream_entries"] != 0 for row in rows):
        raise ValueError("An unreviewed custom definition reached upstream")
    _run(driver, directory, "custom-entry-trap", inputs, "--forbid-upstream", 86)
    _run(driver, directory, "custom-geometry-trap", inputs, "--forbid-geometry", 87)
    if (b"UPSTREAM_ENTRY_BLOCKED" not in (directory / "custom-entry-trap.stderr").read_bytes()
            or b"GEOMETRY_ENTRY_BLOCKED" not in (directory / "custom-geometry-trap.stderr").read_bytes()):
        raise ValueError("Custom control traps were not active")
    fresh = {row["id"]: row for row in _read_lines(_run(driver, directory, "custom-fresh", inputs, None, 0))}
    duplicate = directory / "custom-duplicate-inputs.jsonl"
    duplicate.write_text((json.dumps(base) + "\n") * 2, encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "custom-duplicate", duplicate, None, 0))
    if len(rows) != 2 or rows[0] != fresh[base["id"]] or rows[1].get("error") != "duplicate_id" or rows[1][
        "upstream_entries"] != 0:
        raise ValueError("Duplicate custom case reached upstream twice")
    other = next(case for case in cases if case["fuel_model"] == 151)
    skipped = next(
        case for case, result in zip(data["cases"], data["results"], strict=False)
        if result["status"] == "unavailable"
    )
    sequence = [dict(base, id="custom-state-A1", session="percent-state"),
                dict(other, id="custom-state-B", session="percent-state"),
                dict(skipped, id="custom-state-unavailable", session="percent-state"),
                dict(base, id="custom-state-A2", session="percent-state")]
    path = directory / "custom-state-inputs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in sequence), encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "custom-state", path, None, 0))
    if (len(rows) != 4 or rows[0]["output"] != rows[3]["output"]
            or rows[0]["output"] != fresh[base["id"]]["output"]
            or rows[2]["status"] != "unavailable" or rows[2]["geometry_entries"] != 0):
        raise ValueError("Custom definition state did not restore a fresh full run")
    return dict(custom_negative_cases=18, custom_negative_upstream_entries=0,
                custom_upstream_trap_exit=86, custom_geometry_trap_exit=87,
                custom_duplicate_ids_rejected=True, custom_state_sequence_checked=True)


def _check_fire_size(driver, directory, root, exact_replay):
    """Check the standalone FireSize whitelist and both entry traps.

    :param driver: Source-matched executable.
    :param directory: Retained scratch evidence directory.
    :param root: Reference checkout root.
    :param exact_replay: Whether to require the original compiler's exact bytes.
    :return: FireSize rejection and execution control counts.
    """
    calm, surface, zero = load_cases(root)
    controls = (
        (dict(calm, wind_mph=5), "unreviewed_fire_size_request"),
        (dict(calm, wind_mph=-0.0), "unreviewed_fire_size_request"),
        (dict(surface, wind_mph=math.nextafter(5, math.inf)), "unreviewed_fire_size_request"),
        (dict(surface, spread_ft_min=-1), "unreviewed_fire_size_request"),
        (dict(zero, elapsed_min=-0.0), "unreviewed_fire_size_request"),
        (dict(calm, mode="Unknown"), "unsupported_fire_size_mode"),
        (dict(surface, wind_mph="NaN"), "numeric_type_wind_mph"),
        (dict(calm, session="state"), "unsupported_field"),
    )
    path = directory / "fire-size-negative-input.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case, _ in controls), encoding="utf-8")
    result_path = _run(driver, directory, "fire-size-negative", path, "--forbid-upstream", 0)
    expected = [dict(id=case["id"], status="rejected", upstream_entries=0, error=error)
                for case, error in controls]
    if _read_lines(result_path) != expected:
        raise ValueError("FireSize mutation escaped guarded rejection")
    cases_path = root / DATASET / "cases.jsonl"
    if exact_replay:
        replay = _run(driver, directory, "fire-size-replay", cases_path, None, 0)
        if replay.read_bytes() != (cases_path.parent / "results.jsonl").read_bytes():
            raise ValueError("Exact FireSize replay differs from stored observations")
    _run(driver, directory, "fire-size-upstream-positive-trap", cases_path, "--forbid-upstream", 86)
    _run(driver, directory, "fire-size-geometry-positive-trap", cases_path, "--forbid-geometry", 87)
    for name in ("upstream", "geometry"):
        message = (directory / f"fire-size-{name}-positive-trap.stderr").read_bytes()
        if f"{name.upper()}_ENTRY_BLOCKED".encode() not in message:
            raise ValueError("FireSize entry trap was not active")
    duplicate = directory / "fire-size-duplicate-input.jsonl"
    duplicate.write_text((json.dumps(calm) + "\n") * 2, encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "fire-size-duplicate", duplicate, None, 0))
    expected_duplicate = dict(id=calm["id"], status="rejected", upstream_entries=0, error="duplicate_id")
    if len(rows) != 2 or rows[0].get("status") != "ok" or rows[1] != expected_duplicate:
        raise ValueError("Duplicate FireSize case reached upstream twice")
    return dict(negative_cases=len(controls), negative_upstream_entries=0, upstream_trap_exit=86,
                geometry_trap_exit=87, duplicate_ids_rejected=True)


def _check_ignite(driver, directory, root, exact_replay):
    """Verify the Ignite input guard, both entry traps, and exact local replay.

    :param driver: Source-matched executable.
    :param directory: Retained scratch evidence directory.
    :param root: Reference checkout root.
    :param exact_replay: Whether to require the original compiler's exact bytes.
    :return: Ignite rejection and execution control counts.
    """
    dry, low, high, wet = load_ignite_cases(root)
    controls = (
        (dict(dry, air_temperature_f=72), "unreviewed_ignite_request"),
        (dict(dry, sun_shade_fraction=0.25), "unreviewed_ignite_request"),
        (dict(dry, moisture_1h=-0.0), "unreviewed_ignite_request"),
        (dict(low, moisture_1h=math.nextafter(0.125, math.inf)), "unreviewed_ignite_request"),
        (dict(high, moisture_1h=0.75), "unreviewed_ignite_request"),
        (dict(wet, moisture_1h="NaN"), "numeric_type_moisture_1h"),
        (dict(dry, moisture_100h=0.125), "unsupported_field"),
        (dict(wet, id="ignite-unreviewed"), "unreviewed_ignite_request"),
    )
    path = directory / "ignite-negative-input.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case, _ in controls), encoding="utf-8")
    negative = _run(driver, directory, "ignite-negative", path, "--forbid-upstream", 0)
    expected = [dict(id=case["id"], status="rejected", upstream_entries=0, error=error)
                for case, error in controls]
    if _read_lines(negative) != expected:
        raise ValueError("Ignite mutation escaped guarded rejection")
    cases_path = root / IGNITE_DATASET / "cases.jsonl"
    if exact_replay:
        replay = _run(driver, directory, "ignite-replay", cases_path, None, 0)
        if replay.read_bytes() != (cases_path.parent / "results.jsonl").read_bytes():
            raise ValueError("Exact Ignite replay differs from stored observations")
    for name, code in (("upstream", 86), ("geometry", 87)):
        stem = f"ignite-{name}-positive-trap"
        _run(driver, directory, stem, cases_path, f"--forbid-{name}", code)
        if f"{name.upper()}_ENTRY_BLOCKED".encode() not in (directory / f"{stem}.stderr").read_bytes():
            raise ValueError("Ignite entry trap was not active")
    duplicate = directory / "ignite-duplicate-input.jsonl"
    duplicate.write_text((json.dumps(dry) + "\n") * 2, encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "ignite-duplicate", duplicate, None, 0))
    expected_duplicate = dict(id=dry["id"], status="rejected", upstream_entries=0, error="duplicate_id")
    if len(rows) != 2 or rows[0].get("status") != "ok" or rows[1] != expected_duplicate:
        raise ValueError("Duplicate Ignite case reached upstream twice")
    return dict(negative_cases=len(controls), negative_upstream_entries=0, upstream_trap_exit=86,
                output_trap_exit=87, duplicate_ids_rejected=True)


def _check_mortality(driver, directory, root, exact_replay):
    """Verify the Mortality input guard, both entry traps, and exact local replay.

    :param driver: Source-matched executable.
    :param directory: Retained scratch evidence directory.
    :param root: Reference checkout root.
    :param exact_replay: Whether to require the original compiler's exact bytes.
    :return: Mortality rejection and execution control counts.
    """
    cases = {case["id"]: case for case in mortality.load_cases(root)}
    zero_calm = cases["scorch-zero-calm-77"]
    one_calm = cases["scorch-one-calm-77"]
    one_wind = cases["scorch-one-wind-77"]
    controls = (
        (dict(zero_calm, intensity_btu_ft_s=1), "unreviewed_mortality_request"),
        (dict(zero_calm, air_temperature_f=14), "unreviewed_mortality_request"),
        (dict(one_wind, wind_mph=0), "unreviewed_mortality_request"),
        (dict(zero_calm, intensity_btu_ft_s=-0.0), "unreviewed_mortality_request"),
        (dict(one_calm, intensity_btu_ft_s=math.nextafter(1, math.inf)), "unreviewed_mortality_request"),
        (dict(one_calm, id="scorch-unreviewed"), "unreviewed_mortality_request"),
        (dict(zero_calm, air_temperature_f="77"), "numeric_type_air_temperature_f"),
        (dict(one_wind, species="PIPO"), "unsupported_field"),
    )
    path = directory / "mortality-negative-input.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case, _ in controls), encoding="utf-8")
    negative = _run(driver, directory, "mortality-negative", path, "--forbid-upstream", 0)
    expected = [dict(id=case["id"], status="rejected", upstream_entries=0, error=error)
                for case, error in controls]
    if _read_lines(negative) != expected:
        raise ValueError("Mortality mutation escaped guarded rejection")
    cases_path = root / mortality.DATASET / "cases.jsonl"
    if exact_replay:
        replay = _run(driver, directory, "mortality-replay", cases_path, None, 0)
        if replay.read_bytes() != (cases_path.parent / "results.jsonl").read_bytes():
            raise ValueError("Exact Mortality replay differs from stored observations")
    for name, code in (("upstream", 86), ("geometry", 87)):
        stem = f"mortality-{name}-positive-trap"
        _run(driver, directory, stem, cases_path, f"--forbid-{name}", code)
        if f"{name.upper()}_ENTRY_BLOCKED".encode() not in (directory / f"{stem}.stderr").read_bytes():
            raise ValueError("Mortality entry trap was not active")
    duplicate = directory / "mortality-duplicate-input.jsonl"
    duplicate.write_text((json.dumps(zero_calm) + "\n") * 2, encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "mortality-duplicate", duplicate, None, 0))
    expected_duplicate = dict(id=zero_calm["id"], status="rejected", upstream_entries=0, error="duplicate_id")
    if len(rows) != 2 or rows[0].get("status") != "ok" or rows[1] != expected_duplicate:
        raise ValueError("Duplicate Mortality case reached upstream twice")
    return dict(negative_cases=len(controls), negative_upstream_entries=0, upstream_trap_exit=86,
                output_trap_exit=87, duplicate_ids_rejected=True)


def _check_safety(driver, directory, root, exact_replay):
    """Verify the Safety input guard, both entry traps, and pinned replay.

    :param driver: Source-matched executable.
    :param directory: Retained scratch evidence directory.
    :param root: Reference checkout root.
    :param exact_replay: Whether to require the original compiler's exact bytes.
    :return: Safety rejection and execution control counts.
    """
    cases = safety.load_cases(root)
    empty, personnel, equipment, both = cases[:4]
    controls = (
        (dict(empty, flame_ft=-0.0), "unreviewed_safety_request"),
        (dict(personnel, personnel=0.5), "unreviewed_safety_request"),
        (dict(equipment, equipment=-1), "unreviewed_safety_request"),
        (dict(both, equipment=2), "unreviewed_safety_request"),
        (dict(cases[4], personnel="0"), "numeric_type_personnel"),
        (dict(cases[5], equipment=True), "numeric_type_equipment"),
        (dict(cases[6], personnel=1), "unreviewed_safety_request"),
        (dict(cases[7], area_per_person_sqft=0), "unsupported_field"),
    )
    path = directory / "safety-negative-input.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case, _ in controls), encoding="utf-8")
    negative = _run(driver, directory, "safety-negative", path, "--forbid-upstream", 0)
    expected = [dict(id=case["id"], status="rejected", upstream_entries=0, error=error)
                for case, error in controls]
    if _read_lines(negative) != expected:
        raise ValueError("Safety mutation escaped guarded rejection")
    cases_path = root / safety.DATASET / "cases.jsonl"
    if exact_replay:
        replay = _run(driver, directory, "safety-replay", cases_path, None, 0)
        if replay.read_bytes() != (cases_path.parent / "results.jsonl").read_bytes():
            raise ValueError("Exact Safety replay differs from stored observations")
    for name, code in (("upstream", 86), ("geometry", 87)):
        stem = f"safety-{name}-positive-trap"
        _run(driver, directory, stem, cases_path, f"--forbid-{name}", code)
        if f"{name.upper()}_ENTRY_BLOCKED".encode() not in (directory / f"{stem}.stderr").read_bytes():
            raise ValueError("Safety entry trap was not active")
    duplicate = directory / "safety-duplicate-input.jsonl"
    duplicate.write_text((json.dumps(empty) + "\n") * 2, encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "safety-duplicate", duplicate, None, 0))
    expected_duplicate = dict(id=empty["id"], status="rejected", upstream_entries=0, error="duplicate_id")
    if len(rows) != 2 or rows[0].get("status") != "ok" or rows[1] != expected_duplicate:
        raise ValueError("Duplicate Safety case reached upstream twice")
    return dict(negative_cases=len(controls), negative_upstream_entries=0, upstream_trap_exit=86,
                output_trap_exit=87, duplicate_ids_rejected=True)


def _check_spot(driver, directory, root, exact_replay):
    """Verify the flat Spot input guard, both entry traps, and exact local replay.

    :param driver: Source-matched executable.
    :param directory: Retained scratch evidence directory.
    :param root: Reference checkout root.
    :param exact_replay: Whether to require the original compiler's exact bytes.
    :return: Spot rejection and execution control counts.
    """
    cases = {case["id"]: case for case in spot.load_cases(root)}
    calm = cases["spot-flat-calm-zero-cover"]
    wind = cases["spot-flat-wind-zero-cover"]
    covered = cases["spot-flat-wind-ten-cover"]
    controls = (
        (dict(calm, flame_ft=4), "unreviewed_spot_request"),
        (dict(wind, wind_mph=math.nextafter(5, math.inf)), "unreviewed_spot_request"),
        (dict(covered, cover_ft=0), "unreviewed_spot_request"),
        (dict(calm, wind_mph=-0.0), "unreviewed_spot_request"),
        (dict(wind, cover_ft=-0.0), "unreviewed_spot_request"),
        (dict(wind, id="spot-unreviewed"), "unreviewed_spot_request"),
        (dict(wind, cover_ft="0"), "numeric_type_cover_ft"),
        (dict(covered, canopy_mode="OPEN"), "unsupported_field"),
    )
    path = directory / "spot-negative-input.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case, _ in controls), encoding="utf-8")
    negative = _run(driver, directory, "spot-negative", path, "--forbid-upstream", 0)
    expected = [dict(id=case["id"], status="rejected", upstream_entries=0, error=error)
                for case, error in controls]
    if _read_lines(negative) != expected:
        raise ValueError("Spot mutation escaped guarded rejection")
    cases_path = root / spot.DATASET / "cases.jsonl"
    if exact_replay:
        replay = _run(driver, directory, "spot-replay", cases_path, None, 0)
        if replay.read_bytes() != (cases_path.parent / "results.jsonl").read_bytes():
            raise ValueError("Exact Spot replay differs from stored observations")
    for name, code in (("upstream", 86), ("geometry", 87)):
        stem = f"spot-{name}-positive-trap"
        _run(driver, directory, stem, cases_path, f"--forbid-{name}", code)
        if f"{name.upper()}_ENTRY_BLOCKED".encode() not in (directory / f"{stem}.stderr").read_bytes():
            raise ValueError("Spot entry trap was not active")
    duplicate = directory / "spot-duplicate-input.jsonl"
    duplicate.write_text((json.dumps(calm) + "\n") * 2, encoding="utf-8")
    rows = _read_lines(_run(driver, directory, "spot-duplicate", duplicate, None, 0))
    expected_duplicate = dict(id=calm["id"], status="rejected", upstream_entries=0, error="duplicate_id")
    if len(rows) != 2 or rows[0].get("status") != "ok" or rows[1] != expected_duplicate:
        raise ValueError("Duplicate Spot case reached upstream twice")
    return dict(negative_cases=len(controls), negative_upstream_entries=0, upstream_trap_exit=86,
                output_trap_exit=87, duplicate_ids_rejected=True)


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


def check_driver(driver, directory, root=ROOT, build_receipt=None, exact_replay=True):
    """Verify exact replay, guarded rejections, and active entry traps.

    :param driver: Out-of-tree executable supplied explicitly by the operator.
    :param directory: Retained scratch directory.
    :param root: Candidate repository root.
    :param build_receipt: Optional identity of another source-matched compiler build.
    :param exact_replay: Require byte identity, or explicitly measure the conditional caps.
    :return: Control counts without a cross-implementation parity verdict.
    """
    driver, directory, root = Path(driver).resolve(), Path(directory).resolve(), Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data = load_reference(root, measurement=True)
    build = _prepared_build(root) if build_receipt is None else build_receipt
    if (_digest(driver) != build["driver_sha256"] or build["upstream_sha"] != PIN
            or (build.get("driver_source_sha256_lf") or build.get("driver_source_sha256"))
            != _text_digest(root / "reference/behave_cpp_overlay/driver.cpp")):
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
    unavailable = [case for case, result in zip(data["cases"], data["results"], strict=False)
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
    byte_identical = repeated.read_bytes() == (source / "surface/results.jsonl").read_bytes()
    if exact_replay and not byte_identical:
        raise ValueError("Exact-driver replay differs from stored observations")
    if not exact_replay:
        from cpp_controls import calibration_policy, check_values

        actual_rows = _read_lines(repeated)
        if [row["id"] for row in actual_rows] != [row["id"] for row in data["results"]]:
            raise ValueError("Compiler replay case identities changed")
        for actual_row, expected_row in zip(actual_rows, data["results"], strict=False):
            if set(actual_row) != set(expected_row) or any(actual_row[key] != expected_row[key]
                                                         for key in expected_row if key != "output"):
                raise ValueError("Compiler replay execution contract changed")
            if expected_row["status"] == "ok":
                check_values(actual_row["output"], expected_row["output"], calibration_policy(root))
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
                   replay_cases=len(data["cases"]), replay_byte_identical=byte_identical, duplicate_ids_rejected=True,
                   state_sequence_checked=True, scientific_parity_approved=False)
    summary.update(_check_custom(driver, directory, data, root))
    summary["fire_size"] = _check_fire_size(driver, directory, root, exact_replay)
    summary["ignite"] = _check_ignite(driver, directory, root, exact_replay)
    summary["mortality"] = _check_mortality(driver, directory, root, exact_replay)
    summary["safety"] = _check_safety(driver, directory, root, exact_replay)
    summary["spot"] = _check_spot(driver, directory, root, exact_replay)
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
