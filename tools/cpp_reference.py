"""Prepare and inspect pinned C++ observations without approving scientific parity."""

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "f3b2943e69abca942e795c5b1b45276f6338abb5"
FIELDS = ("fireline_intensity_btu_ft_s", "flame_length_ft", "heat_per_unit_area_btu_ft2",
          "length_to_width_ratio", "spread_chains_per_hour")
INPUTS = {"id", "operation", "fuel_model", "wind_direction", "m1h", "m10h", "m100h", "mlh", "mlw",
          "wind_speed", "slope", "aspect", "canopy_cover", "canopy_height", "crown_ratio"}
OPTIONAL_INPUTS = {"wind_height", "wind_orientation", "waf_method", "user_waf", "moisture_mode", "session",
                   "fuel_mode", "moe_percent"}
MOE_PERCENT_CONTROLS = {
    110: 24., 111: 24., 150: 20., 151: 13., 152: 15., 153: 15., 154: 25.,
    155: 25., 156: 20., 157: 35., 158: 24., 159: 35., 166: 26., 167: 30.,
    168: 35., 169: 30., 170: 27., 171: 40., 172: 32., 190: 28., 191: 25.,
    192: 45., 193: 26.,
}
CONTROL_MODEL = 250
FIXTURE_BUDGET = {'case_limit': 800,
                  'canonical': {'raw_bytes': 6 * 1024 ** 2, 'gzip_bytes': 512 * 1024},
                  'standalone': {'raw_bytes': 512 * 1024, 'gzip_bytes': 64 * 1024}}


class UnapprovedCppPolicy(ValueError):
    """An observation cannot be called parity before owner approval."""


def _cases():
    """Create deterministic pilot inputs from the existing numbered registry.

    :return: Ordered cases and their applicability profiles.
    """
    from behave_fire.components.fuel_models import FuelModels

    registry = FuelModels().fuel_models_
    numbers = sorted(n for n, record in registry.items() if record["is_defined"])
    if len(numbers) != 83:
        raise ValueError(f"Numbered-model inventory changed: {len(numbers)}")
    records, profiles = [], {}

    def append(number, label, **overrides):
        """Append one explicit-unit case.

        :param number: Registry identifier.
        :param label: Unique profile suffix.
        :param overrides: Reviewed changes to base inputs.
        :return: None; appends a case and its profile.
        """
        case = dict(id=f"fm{number}-{label}", operation="surface", fuel_model=number,
                    wind_direction=0, m1h=6.0, m10h=7.0, m100h=8.0, mlh=60.0, mlw=90.0,
                    wind_speed=5.0, slope=30.0, aspect=0.0, canopy_cover=50.0,
                    canopy_height=30.0, crown_ratio=0.5)
        case.update(overrides)
        records.append(case)
        profiles[case["id"]] = label

    for number in numbers:
        for angle in (0, 90, 180, 270, 360):
            append(number, f"angle{angle}", wind_direction=angle)
    selected = (1, 4, 101, 110, 124, 146, 181, 204)
    for number in selected:
        append(number, "calm-flat", wind_speed=0.0, slope=0.0, canopy_cover=0.0,
               canopy_height=0.0, crown_ratio=0.0)
        append(number, "windy-slope", wind_speed=20.0, slope=60.0, wind_direction=180)
        append(number, "closed-canopy", canopy_cover=100.0, canopy_height=60.0, crown_ratio=1.0)
        for aspect in (90, 180, 270, 360):
            append(number, f"aspect{aspect}", aspect=float(aspect), wind_direction=90)
        extinction = registry[number]["moisture_of_extinction_dead"] * 100.0
        for label, moisture in (("below", math.nextafter(extinction, 0)),
                                ("at", extinction), ("above", math.nextafter(extinction, math.inf))):
            append(number, "extinction-" + label, m1h=moisture, m10h=moisture, m100h=moisture)
        for moisture in (29.99, 30.0, 30.01, 119.99, 120.0, 120.01):
            append(number, "herb-" + str(moisture), mlh=moisture)
    rng = random.Random(20261113)
    for index in range(24):
        append(rng.choice(numbers), f"interior{index}", wind_speed=rng.uniform(0, 20),
               slope=rng.uniform(0, 60), m1h=rng.uniform(2, 20), mlh=rng.uniform(30, 140),
               canopy_cover=rng.uniform(0, 100), canopy_height=rng.uniform(1, 80),
               crown_ratio=rng.uniform(0, 1))
    for number in sorted({0, 255, *(n for n, record in registry.items() if not record["is_defined"])}):
        append(number, "undefined")
    for number in (1, 4, 124):
        for height in ("TwentyFoot", "TenMeter"):
            for orientation in ("North", "Upslope"):
                for label, factor in (("crown", None), ("user0", 0.0), ("user04", 0.4), ("user1", 1.0)):
                    options = dict(wind_height=height, wind_orientation=orientation,
                                   waf_method="CrownRatio" if factor is None else "UserInput",
                                   moisture_mode="BySizeClass", wind_direction=90, aspect=180.0)
                    if factor is not None:
                        options["user_waf"] = factor
                    append(number, f"mode-{height}-{orientation}-{label}", **options)
    append(124, "state-A1", session="state-sequence")
    append(4, "state-B", session="state-sequence", wind_height="TenMeter", wind_orientation="Upslope",
           waf_method="UserInput", user_waf=0.4, moisture_mode="BySizeClass", wind_direction=90)
    append(0, "state-unavailable", session="state-sequence")
    append(124, "state-A2", session="state-sequence")
    return records, profiles


def _custom_cases(cases):
    """Declare a distinct, explicit Percent-unit control for each affected scenario.

    :param cases: Original numbered-model Surface inputs.
    :return: New control cases retaining all original weather and geometry inputs.
    """
    return [dict(case, id="custom-" + case["id"], fuel_mode="percent_unit_custom",
                 moe_percent=MOE_PERCENT_CONTROLS[case["fuel_model"]])
            for case in cases if case["fuel_model"] in MOE_PERCENT_CONTROLS]


def _digest(path):
    """Hash one artifact byte for byte.

    :param path: Artifact path.
    :return: SHA256 hex digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _document(path):
    """Read strict JSON metadata without duplicate keys or nonfinite constants.

    :param path: UTF-8 JSON document.
    :return: Parsed document.
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _json(path, value):
    """Write an explicitly requested candidate artifact.

    :param path: Destination path.
    :param value: JSON-compatible document.
    :return: None; writes UTF-8 with finite JSON numbers.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        newline="\n")


def _prepared_build(root):
    """Read the current prepared driver without rewriting historical build provenance.

    :param root: Repository root.
    :return: Verified current build record, or the original preparation record.
    """
    directory = Path(root) / "tests/data/cpp_reference"
    manifest = _document(directory / "surface/manifest.json")
    path = directory / "prepared-build.json"
    if not path.is_file():
        return manifest["build_inputs"]
    prepared = _document(path)
    if (prepared["upstream_sha"] != PIN
            or prepared["historical_manifest_sha256"] != _digest(directory / "surface/manifest.json")
            or prepared["original_replay_sha256"] != _digest(directory / "surface/results.jsonl")):
        raise ValueError("Prepared build does not preserve historical observations")
    custom_manifest = prepared.get("historical_custom_manifest_sha256")
    if custom_manifest is not None and custom_manifest != _digest(directory / "surface_custom/manifest.json"):
        raise ValueError("Prepared build does not preserve the historical custom manifest")
    return prepared


def _reject_constant(value):
    """Reject non-standard bare JSON float constants.

    :param value: Decoder token.
    :return: Never returns; raises ValueError.
    """
    raise ValueError(f"Bare nonfinite JSON token: {value}")


def _required_artifacts(root):
    """Enumerate mandatory and additional shipped artifacts for closed verification.

    :param root: Repository root.
    :return: Exact repository-relative artifact paths that must be hashed.
    """
    base = root / "tests/data/cpp_reference"
    mandatory = {"tests/data/cpp_reference/" + name for name in (
        "inventory.json", "findings.json", "tolerance-policy.json", "fixture-budget.json", "surface/cases.jsonl",
        "surface/results.jsonl", "surface/manifest.json", "surface/profiles.json", "surface/applicability.json",
        "negative-inputs.jsonl", "negative-expected.json", "prepared-build.json",
        "fire_size/cases.jsonl", "fire_size/manifest.json", "fire_size/results.jsonl", "fire_size/schema.json",
        "ignite/cases.jsonl", "ignite/manifest.json", "ignite/results.jsonl", "ignite/schema.json",
        "mortality/cases.jsonl", "mortality/manifest.json", "mortality/results.jsonl", "mortality/schema.json",
        "spot/cases.jsonl", "spot/manifest.json", "spot/results.jsonl", "spot/schema.json",
        "surface_custom/cases.jsonl", "surface_custom/results.jsonl", "surface_custom/manifest.json")}
    mandatory.update("tools/" + name for name in (
        "cpp_reference.py", "cpp_controls.py", "cpp_fire_size.py", "cpp_ignite.py", "cpp_inventory.py",
        "cpp_mortality.py", "cpp_spot.py", "cpp_calibrate.py", "export_cpp_crate.py", "check_cpp_driver.py"))
    mandatory.update(f"rust/crates/behave-fire/tests/data/cpp_reference/{dataset}/{name}"
                     for dataset in ("surface", "surface_custom", "fire_size", "ignite", "mortality")
                     for name in ("cases.jsonl", "results.jsonl", "manifest.json", "tolerance-policy.json")
                     if dataset != "surface_custom" or name != "tolerance-policy.json")
    mandatory.update(p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file() and p.name != "hashes.json")
    return mandatory


def _rows(path, *, parse_int=int):
    """Read strict JSONL with unique case identifiers.

    :param path: JSONL path.
    :param parse_int: Decoder for integer tokens.
    :return: Ordered case objects.
    """
    rows = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_int=parse_int)
            for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] for row in rows
    ):
        raise ValueError("Empty dataset or invalid case identity")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case ID")
    return rows


def _signed_int(token):
    """Preserve the negative sign when decoding a lexical zero.

    :param token: JSON integer token.
    :return: Integer value or negative binary64 zero for the signed zero token.
    """
    return -0.0 if token == "-0" else int(token)


def _source_digest(path):
    """Normalize Git text-auto line endings without assuming a source encoding.

    :param path: Trusted upstream file.
    :return: SHA256 with CRLF translation only when Git's binary probe finds no NUL.
    """
    payload = Path(path).read_bytes()
    text = b"\0" not in payload[:8000]
    return hashlib.sha256(payload.replace(b"\r\n", b"\n") if text else payload).hexdigest()


def _text_digest(path):
    """Hash source text canonically while retaining raw build hashes separately.

    :param path: UTF-8 source artifact.
    :return: SHA256 after CRLF-to-LF normalization only.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _unique_object(pairs):
    """Reject duplicate JSON object keys.

    :param pairs: Decoder key/value pairs.
    :return: Uniquely keyed dictionary.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _untrusted_changes(status):
    """Ignore only the explicitly permitted untracked IDE directory.

    :param status: Git porcelain output with all untracked paths.
    :return: Remaining source-state records.
    """
    return [line for line in status.splitlines() if line and not line.startswith("?? .idea/")]


def _validate_case(case):
    """Enforce the same bounded preparation schema as the driver.

    :param case: Candidate input row.
    :return: None; raises for a malformed or unsafe input.
    """
    if (not INPUTS <= set(case) <= INPUTS | OPTIONAL_INPUTS or case["operation"] != "surface"
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case["id"])):
        raise ValueError("Case schema changed")
    if type(case["fuel_model"]) is not int or not 0 <= case["fuel_model"] <= 255:
        raise ValueError("Unsafe fuel storage ID")
    mode = case.get("fuel_mode", "numbered")
    if mode not in ("numbered", "percent_unit_custom"):
        raise ValueError("Unsupported fuel mode")
    if mode == "percent_unit_custom":
        value = case.get("moe_percent")
        if (case["fuel_model"] not in MOE_PERCENT_CONTROLS or type(value) not in (int, float)
                or not math.isfinite(value) or value != MOE_PERCENT_CONTROLS[case["fuel_model"]]):
            raise ValueError("Unreviewed custom Percent-unit extinction definition")
    elif "moe_percent" in case:
        raise ValueError("Unused custom extinction quantity")
    if type(case["wind_direction"]) is not int or case["wind_direction"] not in (0, 90, 180, 270, 360):
        raise ValueError("Unsafe wind-direction grid")
    maximum = dict(m1h=300, m10h=300, m100h=300, mlh=300, mlw=300, wind_speed=40,
                   slope=60, aspect=360, canopy_cover=100, canopy_height=200, crown_ratio=1)
    for field, upper in maximum.items():
        value = case[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"Unsafe physical input: {field}")
    if case["aspect"] not in (0, 90, 180, 270, 360):
        raise ValueError("Unsafe aspect grid")
    for field, allowed, default in (
            ("wind_height", {"TwentyFoot", "TenMeter"}, "TwentyFoot"),
            ("wind_orientation", {"North", "Upslope"}, "North"),
            ("waf_method", {"CrownRatio", "UserInput"}, "CrownRatio"),
            ("moisture_mode", {"BySizeClass"}, "BySizeClass")):
        if case.get(field, default) not in allowed:
            raise ValueError(f"Unsupported mode: {field}")
    if case.get("waf_method", "CrownRatio") == "UserInput":
        value = case.get("user_waf")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid user WAF")
    elif "user_waf" in case:
        raise ValueError("Unused user WAF")
    if "session" in case and (not isinstance(case["session"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", case[
        "session"])):
        raise ValueError("Invalid state sequence identity")


def _validate_matrix(cases, results, matrix):
    """Require the measured supported-control grid and explicit state transition.

    :param cases: Canonical Surface inputs.
    :param results: Joined C++ observations.
    :param matrix: Scope and exclusion accounting.
    :return: None; raises for missing modes or a stale state result.
    """
    expected = {(model, height, orientation, method, factor)
                for model in (1, 4, 124) for height in ("TwentyFoot", "TenMeter")
                for orientation in ("North", "Upslope")
                for method, factor in (("CrownRatio", None), ("UserInput", 0.0), ("UserInput", 0.4), ("UserInput", 1.0))
    }
    observed = {(case["fuel_model"], case.get("wind_height", "TwentyFoot"), case.get("wind_orientation", "North"),
                 case.get("waf_method", "CrownRatio"), case.get("user_waf")) for case in cases if "-mode-" in case["id"]
    }
    if observed != expected or matrix["measured_mode_cases"] != len(expected) or not matrix["excluded_cpp_controls"]:
        raise ValueError("Incomplete supported Surface mode matrix")
    by_id = {row["id"]: row for row in results}
    inputs = {row["id"]: row for row in cases}
    state_ids = ("fm124-state-A1", "fm4-state-B", "fm0-state-unavailable", "fm124-state-A2")
    if any(inputs[name].get("session") != "state-sequence" for name in state_ids):
        raise ValueError("State cases do not share the explicit driver session")
    if tuple(case["id"] for case in cases if case.get("session") == "state-sequence") != state_ids:
        raise ValueError("State transition order changed")
    if (by_id[state_ids[0]]["output"] != by_id[state_ids[3]]["output"]
            or by_id[state_ids[0]]["output"] != by_id["fm124-angle0"]["output"]
            or by_id[state_ids[2]]["status"] != "unavailable"):
        raise ValueError("State transition differs from a fresh full run")


def assert_matches_cpp(actual, expected, policy):
    """Compare only under an explicitly approved field policy.

    :param actual: Port outputs in the canonical five units.
    :param expected: Independent C++ output fields.
    :param policy: Owner-approved per-field policy.
    :return: None on agreement; raises for missing approval or a mismatch.
    """
    if policy.get("approved") is not True:
        raise UnapprovedCppPolicy("Scientific tolerances require Phase 3 owner approval")
    if set(actual) != set(FIELDS) or set(expected) != set(FIELDS) or set(policy["fields"]) != set(FIELDS):
        raise ValueError("Incomplete comparison fields or policy")
    for field in FIELDS:
        entry = policy["fields"][field]
        if entry["kind"] != "relative_absolute":
            raise ValueError(f"Unimplemented approved policy kind: {field}")
        relative, absolute = entry["rtol"], entry["atol"]
        if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (relative, absolute)):
            raise ValueError("Invalid approved tolerance")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (actual[field], expected[field])):
            raise ValueError("Finite output classification cannot compare nonfinite values")
        if actual[field] == expected[field] == 0 and math.copysign(1, actual[field]) != math.copysign(1, expected[field]
        ):
            raise AssertionError(f"{field}: signed-zero mismatch")
        if not math.isclose(actual[field], expected[field], rel_tol=relative, abs_tol=absolute):
            raise AssertionError(f"{field}: port={actual[field]!r}, C++={expected[field]!r}")


def check_fixture_budget(root=ROOT):
    """Enforce the owner's case and deterministic compressed/uncompressed fixture ceilings.

    :param root: Checkout containing canonical and standalone copied C++ fixtures.
    :return: Actual case count and byte measurements; raises for drift or unapproved limits.
    """
    root = Path(root)
    canonical = root / 'tests/data/cpp_reference'
    policy = _document(canonical / 'fixture-budget.json')
    if policy.get('approved') is not True or policy.get('limits') != FIXTURE_BUDGET:
        raise ValueError('Unapproved fixture budget limits')
    cases = sum(sum(bool(line.strip()) for line in (canonical / dataset / 'cases.jsonl').read_bytes().splitlines())
                for dataset in ('surface', 'surface_custom', 'fire_size', 'ignite', 'mortality', 'spot'))
    if cases > FIXTURE_BUDGET['case_limit']:
        raise ValueError('C++ fixture case budget exceeded')
    measured = {'cases': cases}
    for name, directory in [('canonical', canonical),
                            ('standalone', root / 'rust/crates/behave-fire/tests/data/cpp_reference')]:
        files = sorted(path for path in directory.rglob('*') if path.is_file())
        payload = b''.join(path.read_bytes() for path in files)
        actual = {'raw_bytes': len(payload), 'gzip_bytes': len(gzip.compress(payload, mtime=0))}
        if any(actual[key] > limit for key, limit in FIXTURE_BUDGET[name].items()):
            raise ValueError(f'{name} fixture budget exceeded: {actual}')
        measured[name] = actual
    return measured


def compare_backends(root, report_dir):
    """Measure raw Python/C++ and Rust/C++ differences without accepting them.

    :param root: Repository root holding the new candidate observations.
    :param report_dir: Ignored destination for full measurement records.
    :return: Summary of exact and nonexact observations; no parity verdict.
    """
    import numpy as np

    from behave_fire import BehaveRun
    from behave_fire.components.fuel_models import FuelModels

    data = load_reference(root, measurement=True)
    mappings = dict(fireline_intensity_btu_ft_s="fireline_intensity", flame_length_ft="flame_length",
                    heat_per_unit_area_btu_ft2="heat_per_unit_area", length_to_width_ratio="fire_length_to_width_ratio",
                    spread_chains_per_hour="spread_rate")
    observations, summaries = [], {}
    for backend in ("python", "rust"):
        runner = BehaveRun(FuelModels(), backend=backend)
        aggregate = {field: dict(count=0, exact_bits=0, max_absolute=0.0, max_relative=0.0,
                                worst_absolute_case=None, worst_relative_case=None,
                                worst_relative_cpp=None, worst_relative_port=None,
                                zero_reference_nonzero_port=0) for field in FIELDS}
        for case, reference in zip(data["cases"], data["results"], strict=False):
            if reference["status"] != "ok" and "session" not in case:
                continue
            arguments = {name: case[name] for name in ("m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed",
                         "wind_direction", "slope", "aspect", "canopy_cover", "canopy_height", "crown_ratio")}
            arguments.update(fuel_model_grid=case["fuel_model"], wind_speed_units=5, slope_units=1,
                             moisture_units=1, canopy_cover_units=1, crown_ratio_units=0,
                             wind_orientation_mode="RelativeToNorth" if case.get("wind_orientation", "North")
                == "North" else "RelativeToUpslope",
                             wind_height_mode=case.get("wind_height", "TwentyFoot"),
                             waf_method="UserInput" if case.get("waf_method") == "UserInput" else "UseCrownRatio",
                             user_waf=case.get("user_waf"), out_units={"spread_rate": 1})
            output = runner.do_surface_run(**arguments)
            if reference["status"] != "ok":
                continue
            for field, port_key in mappings.items():
                actual = float(np.asarray(output[port_key]).reshape(-1)[0])
                expected = reference["output"][field]
                absolute = abs(actual - expected)
                relative = absolute / abs(expected) if expected else (0.0 if absolute == 0 else None)
                if not math.isfinite(actual):
                    raise ValueError(f"Nonfinite port output: {backend}/{case['id']}/{field}")
                exact_bits = struct.pack("!d", actual) == struct.pack("!d", expected)
                observations.append(dict(id=case["id"], backend=backend, field=field, cpp=expected,
                                         port=actual, absolute=absolute, relative=relative, exact_bits=exact_bits))
                stats = aggregate[field]
                stats["count"] += 1
                stats["exact_bits"] += exact_bits
                stats["zero_reference_nonzero_port"] += expected == 0 and actual != 0
                if absolute > stats["max_absolute"]:
                    stats.update(max_absolute=absolute, worst_absolute_case=case["id"])
                if relative is not None and relative > stats["max_relative"]:
                    stats.update(max_relative=relative, worst_relative_case=case["id"],
                                 worst_relative_cpp=expected, worst_relative_port=actual)
        summaries[backend] = aggregate
    summary = dict(status="pending_owner_review", policy_approved=False, backends=summaries,
                   compared_records=len(observations), unavailable=sum(r["status"] != "ok" for r in data["results"]))
    _json(Path(report_dir) / "differences.json", observations)
    _json(Path(report_dir) / "summary.json", summary)
    return summary


def generate_reference(root, driver, attestation, reason):
    """Generate only new pilot files from an explicitly attested external build.

    :param root: Repository candidate destination.
    :param driver: Out-of-tree executable.
    :param attestation: Build/pin/source identity record.
    :param reason: Written reason for initial generation.
    :return: Case and availability counts.
    """
    root, driver = Path(root), Path(driver)
    receipt = _document(attestation)
    if receipt["upstream_sha"] != PIN or receipt["driver_sha256"] != _digest(driver):
        raise ValueError("Untrusted pin or executable")
    overlay = root / "reference/behave_cpp_overlay/driver.cpp"
    expected_source = (receipt['driver_source_sha256_lf'] if 'driver_source_sha256_lf' in receipt
                       else receipt['driver_source_sha256'])
    if expected_source != _text_digest(overlay):
        raise ValueError("Driver source differs from build receipt")
    directory = root / "tests/data/cpp_reference/surface"
    if (directory / "cases.jsonl").exists() or (directory / "results.jsonl").exists():
        raise ValueError("Refusing to overwrite existing observation values")
    cases, profiles = _cases()
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "cases.jsonl"
    inputs.write_text("".join(json.dumps(case, sort_keys=True, allow_nan=False) + "\n" for case in cases),
        encoding="utf-8", newline="\n")
    output = directory / "results.jsonl"
    subprocess.run([str(driver), str(inputs), str(output)], check=True)
    results = _rows(output)
    if [row["id"] for row in cases] != [row["id"] for row in results]:
        raise ValueError("Driver output lost or reordered case identities")
    _json(directory / "profiles.json", profiles)
    policy = dict(approved=False, approval_gate="Phase 3 owner review", fields={
        field: dict(kind="relative_absolute", proposal=None) for field in FIELDS})
    _json(root / "tests/data/cpp_reference/tolerance-policy.json", policy)
    manifest = dict(schema_version=1, dataset="surface", upstream_sha=PIN, case_count=len(cases),
                    result_count=len(results), available=sum(r["status"] == "ok" for r in results),
                    seed=20261113, generation_reason=reason, oracle="pinned C++ external preparation capsule",
                    submodule_checkout_used=False, build_inputs=receipt, fields=list(FIELDS),
                    prepared_overlay_sha256={p.name: _text_digest(p) for p in (root / "reference/behave_cpp_overlay"
                    ).glob("*") if p.is_file()},
                    controls=dict(wind_orientation="North", wind_height="TwentyFoot", waf_method="CrownRatio"),
                    non_goals=["special fuels", "two fuel models", "scientific parity approval"],
                    input_units=dict(moistures="percent", wind="mph", slope="percent", canopy_cover="percent",
                                     canopy_height="feet", crown_ratio="fraction", directions="degrees"))
    _json(directory / "manifest.json", manifest)
    from export_cpp_crate import export

    export(root / "tests/data/cpp_reference", root / "rust/crates/behave-fire/tests/data/cpp_reference/surface")
    sync_hashes(root, reason)
    load_reference(root)
    return {"cases": len(cases), "available": manifest["available"], "unavailable": len(cases)-manifest["available"]}


def load_reference(root=ROOT, *, measurement=False):
    """Verify every shipped artifact and enforce complete independent observations.

    :param root: Repository root; no upstream checkout is read offline.
    :param measurement: Verify reference integrity without requiring previous source qualification.
    :return: Validated artifacts; measurement views always carry an unapproved policy and cannot certify acceptance.
    """
    root = Path(root).resolve()
    directory = root / "tests/data/cpp_reference"
    hashes = _document(directory / "hashes.json")
    if set(hashes) != _required_artifacts(root):
        raise ValueError("Incomplete expected hash list")
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or _digest(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    manifest = _document(directory / "surface/manifest.json")
    cases, results = _rows(directory / "surface/cases.jsonl"), _rows(directory / "surface/results.jsonl")
    if (manifest["upstream_sha"] != PIN or manifest["build_inputs"]["upstream_sha"] != PIN
            or manifest["fields"] != list(FIELDS) or len(cases) != manifest["case_count"]
            or len(results) != manifest["result_count"]):
        raise ValueError("Pin or cardinality mismatch")
    if [r["id"] for r in cases] != [r["id"] for r in results]:
        raise ValueError("Case/result identity mismatch")
    available = 0
    for case, result in zip(cases, results, strict=False):
        _validate_case(case)
        if result["status"] == "ok":
            available += 1
            if set(result) != {"id", "status", "upstream_entries", "geometry_entries", "output"}:
                raise ValueError("Unexpected scientific result schema")
            if type(result["upstream_entries"]) is not int or result["upstream_entries"] != 1 or result[
                "geometry_entries"] != 1:
                raise ValueError("Scientific output lacks the executed calculation")
            if set(result["output"]) != set(FIELDS) or not all(
                    type(v) in (int, float) and math.isfinite(v) for v in result["output"].values()):
                raise ValueError("Missing or nonfinite scientific field")
        elif result["status"] == "unavailable":
            if (set(result) != {"id", "status", "upstream_entries", "geometry_entries", "finding"}
                    or result.get("finding") != "F-013" or result["upstream_entries"] != 1
                    or result["geometry_entries"] != 0):
                raise ValueError("Unsafe unavailable row has invented values")
        else:
            raise ValueError("Unclassified oracle result")
    if available == 0 or available != manifest["available"]:
        raise ValueError("No executable observations or availability drift")
    profiles = _document(directory / "surface/profiles.json")
    if set(profiles) != {case["id"] for case in cases} or not all(isinstance(value, str) and value
        for value in profiles.values()):
        raise ValueError("Incomplete profile identities")
    _validate_matrix(cases, results, _document(directory / "surface/applicability.json"))
    policy = _document(directory / "tolerance-policy.json")
    if (type(policy["approved"]) is not bool or set(policy["fields"]) != set(FIELDS)
            or any(not isinstance(item, dict) or item.get("kind") != "relative_absolute"
                   for item in policy["fields"].values())):
        raise ValueError("Missing field tolerance classification")
    if policy["approved"] and not measurement:
        from cpp_calibrate import validate_qualification

        qualification = directory / "calibration/qualification.json"
        if not qualification.is_file():
            raise ValueError("Active comparison policy lacks complete calibration qualification")
        record = _document(qualification)
        embedded = json.dumps(policy.get("qualification"), sort_keys=True, allow_nan=False)
        canonical = json.dumps(record, sort_keys=True, allow_nan=False)
        if embedded != canonical:
            raise ValueError("Embedded calibration qualification differs from the reviewed record")
        validate_qualification(root, record)
    from cpp_inventory import validate_inventory

    validate_inventory(_document(directory / "inventory.json"), reference_root=directory)
    from cpp_controls import load
    from cpp_fire_size import EXACT_FIELDS, load_observations, validate_policy
    from cpp_ignite import EXACT_FIELDS as IGNITE_FIELDS
    from cpp_ignite import load_observations as load_ignite_observations
    from cpp_ignite import validate_policy as validate_ignite_policy
    from cpp_mortality import EXACT_FIELDS as MORTALITY_FIELDS
    from cpp_mortality import load_observations as load_mortality_observations
    from cpp_mortality import validate_policy as validate_mortality_policy
    from cpp_spot import load_observations as load_spot_observations
    from cpp_spot import validate_measurement as validate_spot_measurement

    controls = load(root, cases)
    qualification = policy.get("qualification") if policy["approved"] else None
    load_observations(root)
    validate_policy(root, policy.get("fire_size"), qualification)
    load_ignite_observations(root)
    validate_ignite_policy(root, policy.get("ignite"), qualification)
    load_mortality_observations(root)
    validate_mortality_policy(root, policy.get("mortality"), qualification)
    load_spot_observations(root)
    validate_spot_measurement(root, policy.get("spot"), None if measurement else qualification)
    _prepared_build(root)
    budget = check_fixture_budget(root)
    if measurement:
        policy = dict(policy, approved=False, fire_size=dict(approved=False, fields=EXACT_FIELDS),
                      ignite=dict(approved=False, fields=IGNITE_FIELDS),
                      mortality=dict(approved=False, fields=MORTALITY_FIELDS))
    return dict(cases=cases, results=results, manifest=manifest, policy=policy, controls=controls, budget=budget)


def main(argv=None):
    """Run explicitly selected preparation or read-only verification.

    :param argv: Optional command-line arguments.
    :return: Zero on requested operation success; scientific approval is separate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--sync-hashes", action="store_true")
    parser.add_argument("--verify-live", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if args.generate:
        if not args.driver or not args.attestation or not args.reason:
            parser.error("Generation requires an attested driver and a written reason")
        print(json.dumps(generate_reference(args.root, args.driver, args.attestation, args.reason)))
    elif args.sync_hashes:
        if not args.reason:
            parser.error("A hash-list update requires a written reason")
        sync_hashes(args.root, args.reason)
    elif args.verify_live:
        print(require_live_checkout(args.root))
    elif args.compare:
        print(json.dumps(compare_backends(args.root, args.compare), indent=2))
    elif args.check:
        data = load_reference(args.root)
        print(
            f"C++ observation integrity passed: {len(data['cases'])} cases; "
            f"parity approval={data['policy']['approved']}")
    else:
        parser.error("Select --generate, --check, --compare, --sync-hashes, or --verify-live")
    return 0


def require_live_checkout(root=ROOT):
    """Require the hard-coded pinned dependency before any live qualification.

    :param root: Repository root.
    :return: Verified dependency path, or raises without fallback.
    """
    path = Path(root) / "reference/behave_cpp"
    if not (path / ".git").exists():
        raise ValueError("Pinned submodule checkout is not prepared; no external fallback is allowed")
    sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if sha != PIN:
        raise ValueError("C++ dependency pin mismatch")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        text=True)
    if _untrusted_changes(dirty):
        raise ValueError("Modified C++ dependency is not a trusted live oracle")
    manifest = _document(Path(root) / "tests/data/cpp_reference/surface/manifest.json")
    prepared = _prepared_build(root)
    expected_source = prepared.get("driver_source_sha256_lf") or prepared["driver_source_sha256"]
    if expected_source != _text_digest(Path(root) / "reference/behave_cpp_overlay/driver.cpp"):
        raise ValueError("Live driver source differs from recorded observations")
    trusted = prepared.get("upstream_source_sha256_portable", manifest["build_inputs"]["upstream_source_hashes"])
    digest = _source_digest if "upstream_source_sha256_portable" in prepared else _digest
    tracked = set(subprocess.check_output(["git", "-C", str(path), "ls-files", "-z"], text=True).split("\0")) - {""}
    if not trusted or set(trusted) != tracked:
        raise ValueError("Incomplete live source hash inventory")
    overlay = prepared.get("prepared_overlay_sha256", manifest["prepared_overlay_sha256"])
    for relative, expected in overlay.items():
        source = Path(root) / "reference/behave_cpp_overlay" / relative
        if source.parent != Path(root) / "reference/behave_cpp_overlay" or _text_digest(source) != expected:
            raise ValueError("Live overlay differs from prepared observations")
    for relative, expected in trusted.items():
        source = (path / relative).resolve()
        if not source.is_relative_to(path.resolve()) or digest(source) != expected:
            raise ValueError(f"Modified trusted C++ input: {relative}")
    return path


def sync_hashes(root, reason):
    """Record reviewed preparation artifact identities without changing observations.

    :param root: Repository root.
    :param reason: Required written reason for metadata synchronization.
    :return: None; writes only the expected artifact hash list.
    """
    if not reason.strip():
        raise ValueError("Hash-list updates need a reason")
    root = Path(root)
    base = root / "tests/data/cpp_reference"
    paths = [root / relative for relative in _required_artifacts(root)]
    _json(base / "hashes.json", {p.relative_to(root).as_posix(): _digest(p) for p in sorted(paths)})


if __name__ == "__main__":
    raise SystemExit(main())
