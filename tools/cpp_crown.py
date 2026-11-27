"""Validate and measure six fixed Scott and Reinhardt Crown runs."""

import hashlib
import json
import math
import struct
from importlib import import_module
from pathlib import Path

import cpp_reference as reference

DATASET = Path("tests/data/cpp_reference/crown")
FIELDS = (
    "surface_ros_ft_min", "crown_ros_ft_min", "crown_fireline_btu_ft_s", "crown_flame_ft",
    "crown_fraction_burned", "fire_type", "final_ros_ft_min", "final_hpua_btu_ft2",
    "final_fireline_btu_ft_s", "final_flame_ft", "crown_length_to_width_ratio",
)
FILES = ("cases.jsonl", "manifest.json", "results.jsonl", "schema.json")
MEASUREMENT_POLICY = dict(approved=False, fields={field: {"kind": "measurement_only"} for field in FIELDS})
OPERATION = "crown"
SOURCES = ("crown.cpp", "crown.h", "crownInputs.cpp", "crownInputs.h", "fireSize.cpp", "fireSize.h",
           "fuelModels.cpp", "fuelModels.h", "surface.cpp", "surface.h", "surfaceFire.cpp", "surfaceFire.h")
REQUESTS = (
    dict(id="calm", operation=OPERATION, wind_fpm=0, canopy_base_ft=15, crown_ratio=0.5),
    dict(id="wind5", operation=OPERATION, wind_fpm=440, canopy_base_ft=15, crown_ratio=0.5),
    dict(id="wind15", operation=OPERATION, wind_fpm=1320, canopy_base_ft=15, crown_ratio=0.5),
    dict(id="torching", operation=OPERATION, wind_fpm=616, canopy_base_ft=5, crown_ratio=25 / 30),
    dict(id="active", operation=OPERATION, wind_fpm=2200, canopy_base_ft=15, crown_ratio=0.5),
    dict(id="zero-base", operation=OPERATION, wind_fpm=0, canopy_base_ft=0, crown_ratio=1.0),
)
FIRE_TYPES = {"calm": 0, "wind5": 0, "wind15": 2, "torching": 1, "active": 3, "zero-base": 1}
PORT_KEYS = {
    "crown_ros_ft_min": "crown_fire_spread_rate",
    "crown_fireline_btu_ft_s": "crown_fire_line_intensity",
    "crown_flame_ft": "crown_flame_length",
    "crown_fraction_burned": "crown_fraction_burned",
    "fire_type": "fire_type",
    "final_ros_ft_min": "final_spread_rate",
    "final_hpua_btu_ft2": "final_heat_per_unit_area",
    "final_fireline_btu_ft_s": "final_fireline_intensity",
    "final_flame_ft": "final_flame_length",
    "crown_length_to_width_ratio": "crown_length_to_width_ratio",
}
GUARDS = dict(negative_cases=11, negative_upstream_entries=0, upstream_trap_exit=86,
              output_trap_exit=87, duplicate_ids_rejected=True)
COMPARISONS = len(REQUESTS) * len(FIELDS) * 16


def _port_observation(module, case):
    """Run fresh public Surface and Crown calls using the fixed FM10 input conventions.

    :param module: Python or native package that supplies BehaveRun and FuelModels.
    :param case: A validated Crown case.
    :return: Canonical Crown fields plus the source Surface spread rate.
    """
    import numpy as np

    from behave_fire.components.behave_units import DensityUnits, FractionUnits, LengthUnits, SlopeUnits, SpeedUnits

    feet = LengthUnits.LengthUnitsEnum.Feet
    fraction = FractionUnits.FractionUnitsEnum
    common = dict(
        fuel_model_grid=10, m1h=6.0, m10h=7.0, m100h=8.0, mlh=60.0, mlw=90.0,
        moisture_units=fraction.Percent, wind_speed=case["wind_fpm"],
        wind_speed_units=SpeedUnits.SpeedUnitsEnum.FeetPerMinute, wind_direction=0.0,
        wind_orientation_mode="RelativeToUpslope", slope=0.0, slope_units=SlopeUnits.SlopeUnitsEnum.Degrees,
        aspect=0.0, canopy_height=30.0, canopy_height_units=feet,
        canopy_base_height=case["canopy_base_ft"], canopy_base_height_units=feet,
    )
    runner = module.BehaveRun(module.FuelModels())
    surface = runner.do_surface_run(
        **common, canopy_cover=50.0, canopy_cover_units=fraction.Percent, crown_ratio=case["crown_ratio"],
        crown_ratio_units=fraction.Fraction, wind_height_mode="TwentyFoot", waf_method="UserInput", user_waf=0.4,
    )
    if np.asarray(surface["midflame_wind_speed"]).item() != case["wind_fpm"] * 0.4:
        raise ValueError("Crown source Surface did not retain the reviewed wind reference")
    crown = runner.do_crown_run(
        **common, surface_results=surface, canopy_bulk_density=0.02,
        canopy_bulk_density_units=DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot, moisture_foliar=100.0,
    )
    output = {field: np.asarray(crown[key]).item() for field, key in PORT_KEYS.items()}
    output["surface_ros_ft_min"] = np.asarray(surface["spread_rate"]).item()
    _validate_output(case["id"], output)
    return dict(id=case["id"], output=output)


def _validate_fields(output):
    """Require complete finite, nonnegative Crown fields or difference maxima.

    :param output: Output or maxima keyed by canonical field name.
    :return: None. Raises for invalid values or missing fields.
    """
    if not isinstance(output, dict) or set(output) != set(FIELDS):
        raise ValueError("Incomplete Crown output")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in output.values()):
        raise ValueError("Invalid Crown output")


def _validate_output(case_id, output):
    """Require the specific fire classification and valid reported geometry and fraction.

    :param case_id: Identity of a reviewed Crown run.
    :param output: C++ or port outputs.
    :return: None. Raises for invalid output or an unexpected classification.
    """
    _validate_fields(output)
    if type(output["fire_type"]) is not int or output["fire_type"] != FIRE_TYPES[case_id]:
        raise ValueError("Unexpected Crown classification")
    if output["crown_fraction_burned"] > 1 or output["crown_length_to_width_ratio"] < 1:
        raise ValueError("Invalid Crown fraction or ellipse")


def artifacts(root):
    """Bind a measurement to the Crown fixture files.

    :param root: Reference checkout root.
    :return: File names mapped to SHA256 digests.
    """
    return {name: reference._digest(Path(root) / DATASET / name) for name in FILES}


def load_cases(root):
    """Require the six reviewed requests in diagnostic order.

    :param root: Reference checkout root.
    :return: Validated case rows.
    """
    rows = reference._rows(Path(root) / DATASET / "cases.jsonl", parse_int=reference._signed_int)
    if json.dumps(rows, sort_keys=True, allow_nan=False) != json.dumps(REQUESTS, sort_keys=True):
        raise ValueError("Crown inputs differ from the approved cases")
    return rows


def load_observations(root):
    """Validate the pinned Crown results, fixture digests and upstream source closure.

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
        raise ValueError("Incomplete Crown observations or provenance")
    return results


def measure_ports(root, backends):
    """Measure public port outputs for the same six guarded requests.

    :param root: Reference checkout root.
    :param backends: Explicit backend names to measure.
    :return: Per-backend observations in approved case order.
    """
    cases = load_cases(root)
    measurements = {}
    for backend in backends:
        module = import_module(f"behave_fire.{backend}")
        measurements[backend] = [_port_observation(module, case) for case in cases]
    return measurements


def measurement_evidence(root, paths, record):
    """Summarize all authenticated C++ and port Crown observations.

    :param root: Checkout holding the approved Crown observations.
    :param paths: Compiler and platform report paths.
    :param record: Source qualification binding the report hashes.
    :return: Comparison counts, differing bits, and maximum absolute differences.
    """
    expected = load_observations(root)
    signature = artifacts(root)
    pending = {name: report["sha256"] for name, report in record["reports"].items()}
    if len(pending) != 9 or {name for name in pending if not name.startswith("ports-")} != {"cpp-GNU", "cpp-MSVC"}:
        raise ValueError("Incomplete Crown compiler or platform evidence")
    count, nonmatching = 0, 0
    maxima = dict.fromkeys(FIELDS, 0.0)
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw, object_pairs_hook=reference._unique_object,
                            parse_constant=reference._reject_constant, parse_int=reference._signed_int)
        kind = report["kind"]
        if kind not in ("cpp", "ports"):
            raise ValueError("Unknown Crown report kind")
        name = "cpp-" + report["compiler"]["id"] if kind == "cpp" else "ports-" + report["platform_key"]
        if pending.pop(name, None) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Crown report differs from the source-qualified evidence")
        measurement = report[OPERATION]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != signature:
            raise ValueError("Crown report used a different dataset")
        if kind == "cpp":
            if json.dumps(report["protocol"][OPERATION], sort_keys=True) != json.dumps(GUARDS, sort_keys=True):
                raise ValueError("Missing Crown entry-guard evidence")
            rows = measurement["observations"]
            validate_cpp_rows(root, rows)
            streams = (rows,)
        else:
            backends = measurement["observations"]
            if not isinstance(backends, dict) or set(backends) != {"python", "rust"} or any(
                not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {"id", "output"}
                                                  for row in rows) for rows in backends.values()
            ):
                raise ValueError("Incomplete Crown backend observations")
            streams = backends.values()
        for rows in streams:
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Incomplete Crown report case identities")
            for row, original in zip(rows, expected, strict=True):
                _validate_output(row["id"], row["output"])
                for field in FIELDS:
                    actual = float(row["output"][field])
                    pinned = float(original["output"][field])
                    nonmatching += struct.pack("!d", actual) != struct.pack("!d", pinned)
                    maxima[field] = max(maxima[field], abs(actual - pinned))
                    count += 1
    if pending or count != COMPARISONS:
        raise ValueError("Incomplete Crown measurement evidence")
    return dict(artifacts=signature, comparisons=count, nonmatching=nonmatching, max_abs=maxima)


def read_cpp_results(root, path):
    """Read guarded C++ results while preserving signed zeros.

    :param root: Reference checkout root.
    :param path: JSONL observation path.
    :return: Validated rows in the reviewed case order.
    """
    rows = reference._rows(path, parse_int=reference._signed_int)
    validate_cpp_rows(root, rows)
    return rows


def validate_cpp_rows(root, rows):
    """Require one upstream and one output entry for every approved case.

    :param root: Reference checkout root.
    :param rows: Parsed raw observations or compiler evidence.
    :return: None. Raises for incomplete or invalid rows.
    """
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != {"id", "status", "upstream_entries", "output_entries", "output"}
        or row["status"] != "ok"
        or any(type(row[key]) is not int or row[key] != 1 for key in ("upstream_entries", "output_entries"))
        for row in rows
    ):
        raise ValueError("Crown row lacks one complete output action")
    if [row["id"] for row in rows] != [case["id"] for case in load_cases(root)]:
        raise ValueError("Crown case/result identity mismatch")
    for row in rows:
        _validate_output(row["id"], row["output"])


def validate_measurement(root, policy, qualification):
    """Verify scope and statistics of the Crown measurement receipt.

    :param root: Reference checkout root.
    :param policy: Crown section of the canonical policy.
    :param qualification: Verified matrix receipt, or None while measuring.
    :return: None. Raises if policy or evidence is invalid.
    """
    if not isinstance(policy, dict) or policy.get("approved") is not False or policy != MEASUREMENT_POLICY:
        raise ValueError("Crown policy must remain measurement-only")
    if qualification is None:
        return
    receipt = qualification.get(OPERATION) if isinstance(qualification, dict) else None
    if (not isinstance(receipt, dict) or set(receipt) != {"artifacts", "comparisons", "nonmatching", "max_abs"}
            or receipt["artifacts"] != artifacts(root) or type(receipt["comparisons"]) is not int
            or receipt["comparisons"] != COMPARISONS or type(receipt["nonmatching"]) is not int):
        raise ValueError("Incomplete or stale Crown measurement receipt")
    maxima = receipt["max_abs"]
    _validate_fields(maxima)
    if maxima["fire_type"] != 0:
        raise ValueError("Inconsistent Crown classification statistics")
    pinned = load_observations(root)
    streams = COMPARISONS // (len(pinned) * len(FIELDS))
    minimum = sum(value > 0 for value in maxima.values())
    # A numerically equal zero may still differ in sign. Classifications are integer categories.
    maximum = streams * sum(
        len(pinned) if maxima[field] > 0 else sum(row["output"][field] == 0 for row in pinned)
        for field in FIELDS if field != "fire_type"
    )
    if not minimum <= receipt["nonmatching"] <= maximum:
        raise ValueError("Inconsistent Crown comparison statistics")
