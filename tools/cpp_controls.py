"""Explicit Percent-unit Surface controls and conditional calibration measurements."""

import argparse
import importlib
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path

import cpp_fine_dead_fuel_moisture as moisture
import cpp_fire_size as fire_size
import cpp_ignite as ignite
import cpp_mortality as mortality
import cpp_reference as reference
import cpp_safety as safety
import cpp_slope_tool as slope_tool
import cpp_spot as spot


def _artifacts(root):
    """Identify numerical inputs whose identities must accompany every calibration.

    :param root: Repository root.
    :return: Repository-relative SHA256 values, excluding mutable approval metadata.
    """
    names = [f"tests/data/cpp_reference/{dataset}/{name}"
             for dataset in ("surface", "surface_custom") for name in ("cases.jsonl", "results.jsonl")]
    return {name: reference._digest(Path(root) / name) for name in names}


def _platform_key():
    """Identify the actual supported host rather than trusting an evidence label.

    :return: Native wheel-platform key; raises for an unsupported host.
    """
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if not arm and machine not in ("amd64", "x86_64", "x64"):
        raise ValueError("Unsupported calibration platform architecture")
    if system == "Windows":
        return "windows-arm64" if arm else "windows-x86_64"
    if system == "Darwin":
        return "macos-arm64" if arm else "macos-x86_64"
    if system == "Linux":
        if Path("/etc/alpine-release").is_file():
            if arm:
                raise ValueError("Unreviewed musl calibration platform")
            return "linux-musl-x86_64"
        return "linux-aarch64" if arm else "linux-x86_64"
    raise ValueError("Unsupported calibration platform")


def _port(case, backend, runners):
    """Evaluate a numbered or explicitly registered definition through its public API.

    :param case: Validated Surface input.
    :param backend: Explicit Python or Rust selection.
    :param runners: Private per-definition runner cache for this measurement.
    :return: Five finite scalar outputs in the declared C++ comparison units.
    """
    import numpy as np

    from behave_fire import BehaveRun

    reference._validate_case(case)
    custom = case.get("fuel_mode") == "percent_unit_custom"
    key = (backend, case["fuel_model"] if custom else None, case.get("moe_percent"))
    if key not in runners:
        module = importlib.import_module("behave_fire.rust.fuel_models" if backend == "rust"
                                         else "behave_fire.components.fuel_models")
        fuels = module.FuelModels()
        if custom:
            fuel = fuels.fuel_models_[case["fuel_model"]]
            accepted = fuels.set_custom_fuel_model(
                reference.CONTROL_MODEL, "UNIT_CONTROL", "Explicit Percent-unit control",
                fuel["fuel_bed_depth"], 0, case["moe_percent"], 1,
                fuel["heat_of_combustion_dead"], fuel["heat_of_combustion_live"], 0,
                fuel["dead_1h"], fuel["dead_10h"], fuel["dead_100h"], fuel["live_herb"], fuel["live_woody"], 0,
                fuel["savr_1h"], fuel["savr_live_herb"], fuel["savr_live_woody"], 0, fuel["is_dynamic"])
            if not accepted:
                raise ValueError("Public custom registration failed")
        runners[key] = BehaveRun(fuels, backend=backend)
    values = {name: case[name] for name in ("m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed",
                                          "wind_direction", "slope", "aspect", "canopy_cover", "canopy_height",
        "crown_ratio")}
    values.update(fuel_model_grid=reference.CONTROL_MODEL if custom else case["fuel_model"],
                  wind_speed_units=5, slope_units=1, moisture_units=1, canopy_cover_units=1,
                  crown_ratio_units=0, wind_height_mode=case.get("wind_height", "TwentyFoot"),
                  wind_orientation_mode="RelativeToNorth" if case.get("wind_orientation", "North") == "North"
                  else "RelativeToUpslope", waf_method="UserInput" if case.get("waf_method") == "UserInput"
                  else "UseCrownRatio", user_waf=case.get("user_waf"), out_units={"spread_rate": 1})
    output = runners[key].do_surface_run(**values)
    names = ("fireline_intensity", "flame_length", "heat_per_unit_area", "fire_length_to_width_ratio", "spread_rate")
    result = {
        field: float(np.asarray(output[name]).reshape(-1)[0])
        for field, name in zip(reference.FIELDS, names, strict=False)
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite port control output")
    return result


def _validate_results(cases, results):
    """Reject incomplete or invented custom-control observations.

    :param cases: Validated control cases.
    :param results: Independently produced C++ rows.
    :return: None; raises for a broken execution or field contract.
    """
    if [case["id"] for case in cases] != [row["id"] for row in results]:
        raise ValueError("Custom control identity mismatch")
    for row in results:
        if (set(row) != {"id", "status", "upstream_entries", "geometry_entries", "output"}
                or row["status"] != "ok" or type(row["upstream_entries"]) is not int
                or row["upstream_entries"] != 1 or type(row["geometry_entries"]) is not int
                or row["geometry_entries"] != 1 or set(row["output"]) != set(reference.FIELDS)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in row["output"].values())):
            raise ValueError("Invalid custom control result")


def calibration_policy(root):
    """Read the owner's conditional caps without activating scientific acceptance.

    :param root: Repository root.
    :return: Field caps only; approval and activation are separate.
    """
    policy = reference._document(Path(root) / "tests/data/cpp_reference/tolerance-policy.json")
    if policy.get("conditional_owner_approval") is not True or set(policy["fields"]) != set(reference.FIELDS):
        raise ValueError("Calibration caps lack conditional owner approval")
    for entry in policy["fields"].values():
        if entry.get("kind") != "relative_absolute" or entry.get("rtol") != 1e-12 or entry.get("atol") != 1e-12:
            raise ValueError("Unapproved calibration cap")
    return policy["fields"]


def check_values(actual, expected, fields):
    """Measure a declared numerical cap without declaring global parity.

    :param actual: Five calculated fields.
    :param expected: Five independent observations.
    :param fields: Conditionally owner-approved comparison bounds.
    :return: Per-field absolute differences; raises on a numerical or schema failure.
    """
    if (set(actual) != set(reference.FIELDS) or set(expected) != set(reference.FIELDS)
            or set(fields) != set(reference.FIELDS)):
        raise ValueError("Incomplete calibration fields")
    differences = {}
    for field in reference.FIELDS:
        left, right = actual[field], expected[field]
        if (type(left) not in (int, float) or type(right) not in (int, float)
                or not math.isfinite(left) or not math.isfinite(right)):
            raise ValueError("Nonfinite calibration field")
        if left == right == 0 and math.copysign(1, left) != math.copysign(1, right):
            raise AssertionError(f"{field}: signed-zero mismatch")
        cap = fields[field]
        if any(type(cap.get(key)) not in (int, float) or cap[key] != 1e-12 for key in ("rtol", "atol")):
            raise ValueError("Invalid or widened calibration cap")
        if not math.isclose(left, right, rel_tol=cap["rtol"], abs_tol=cap["atol"]):
            raise AssertionError(f"{field}: {left!r} versus {right!r}")
        differences[field] = abs(left - right)
    return differences


def generate(root, driver, receipt, output, reason):
    """Generate a separate candidate dataset from the attested qualified driver.

    :param root: Source repository root with original observations.
    :param driver: Explicit out-of-tree executable.
    :param receipt: Current build identity document.
    :param output: New candidate dataset directory, never original observations.
    :param reason: Written generation reason.
    :return: Generated case count; writes only the candidate directory.
    """
    root, driver, output = Path(root), Path(driver), Path(output)
    build = reference._document(receipt)
    if (not reason.strip() or build["upstream_sha"] != reference.PIN
            or build["driver_sha256"] != reference._digest(driver)
            or build["driver_source_sha256_lf"] != reference._text_digest(root
                / "reference/behave_cpp_overlay/driver.cpp")):
        raise ValueError("Untrusted control build")
    if output.exists():
        raise ValueError("Refusing to overwrite control observations")
    cases = reference._custom_cases(reference._rows(root / "tests/data/cpp_reference/surface/cases.jsonl"))
    if len(cases) != 139:
        raise ValueError("Custom-control coverage changed")
    for case in cases:
        reference._validate_case(case)
    output.mkdir(parents=True)
    inputs = output / "cases.jsonl"
    inputs.write_text("".join(json.dumps(case, sort_keys=True, allow_nan=False) + "\n" for case in cases),
                      encoding="utf-8", newline="\n")
    subprocess.run([str(driver.resolve()), str(inputs.resolve()), str((output / "results.jsonl").resolve())], check=True
    )
    results = reference._rows(output / "results.jsonl")
    _validate_results(cases, results)
    manifest = dict(schema_version=1, dataset="surface_custom", upstream_sha=reference.PIN,
                    case_count=139, result_count=139, available=139, fields=list(reference.FIELDS),
                    source_models=sorted(reference.MOE_PERCENT_CONTROLS), registered_model=reference.CONTROL_MODEL,
                    moisture_units="Percent", generation_reason=reason, build=build,
                    source_cases_sha256=reference._digest(root / "tests/data/cpp_reference/surface/cases.jsonl"),
                    files={name: reference._digest(output / name) for name in ("cases.jsonl", "results.jsonl")})
    reference._json(output / "manifest.json", manifest)
    return len(cases)


def load(root, source_cases):
    """Verify custom controls and their exact relationship to original inputs offline.

    :param root: Repository root, requiring no C++ checkout.
    :param source_cases: Original validated numbered-model cases.
    :return: Validated custom cases, observations, and manifest.
    """
    root = Path(root)
    directory = root / "tests/data/cpp_reference/surface_custom"
    manifest = reference._document(directory / "manifest.json")
    cases, results = reference._rows(directory / "cases.jsonl"), reference._rows(directory / "results.jsonl")
    prepared = reference._prepared_build(root)
    historical_build = prepared.get("historical_custom_manifest_sha256") is not None
    if (manifest["upstream_sha"] != reference.PIN or manifest["case_count"] != 139
            or manifest["result_count"] != 139 or manifest["available"] != 139
            or manifest["fields"] != list(reference.FIELDS)
            or manifest["source_models"] != sorted(reference.MOE_PERCENT_CONTROLS)
            or manifest["registered_model"] != reference.CONTROL_MODEL or manifest["moisture_units"] != "Percent"
            or (not historical_build and manifest["build"] != prepared)
            or manifest["source_cases_sha256"] != reference._digest(root
                / "tests/data/cpp_reference/surface/cases.jsonl")
            or cases != reference._custom_cases(source_cases)):
        raise ValueError("Custom-control scope or source inputs changed")
    for name in ("cases.jsonl", "results.jsonl"):
        if manifest["files"][name] != reference._digest(directory / name):
            raise ValueError("Custom-control manifest drift")
    for case in cases:
        reference._validate_case(case)
    _validate_results(cases, results)
    return dict(cases=cases, results=results, manifest=manifest)


def main():
    """Run explicit control generation or proposed-cap measurements.

    :return: None; writes only the operator-selected candidate or report path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=reference.ROOT)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--generate", type=Path)
    parser.add_argument("--reason")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--platform-key")
    args = parser.parse_args()
    if args.generate:
        if not args.driver or not args.receipt or not args.reason:
            parser.error("Generation needs driver, receipt, and reason")
        print(f"Generated {generate(args.root, args.driver, args.receipt, args.generate, args.reason)} controls")
    elif args.report:
        result = measure_ports(args.root, args.report, expected_platform=args.platform_key)
        print(json.dumps(result["counts"]))
    else:
        parser.error("Select --generate or --report")


def measure_ports(root, report, backends=("python", "rust"), expected_platform=None):
    """Calibrate explicit backends and numbered/custom equivalence against the caps.

    :param root: Repository root with verified offline observations.
    :param report: Evidence output path.
    :param backends: Explicit implementations; native metadata is required only when Rust is requested.
    :param expected_platform: Optional matrix identity, checked against the actual host.
    :return: Evidence without activating the production policy.
    """
    import numpy as np

    platform_key = _platform_key()
    if expected_platform is not None and expected_platform != platform_key:
        raise ValueError("Calibration platform identity mismatch")
    root = Path(root).resolve()
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    data = reference.load_reference(root, measurement=True)
    controls = load(root, data["cases"])
    fields = calibration_policy(root)
    numbered = {case["id"]: case for case in data["cases"]}
    raw_results = {row["id"]: row for row in data["results"]}
    records, counts, runners, documented = [], {}, {}, []
    for backend in backends:
        total, paired = 0, 0
        for case, expected in zip(data["cases"], data["results"], strict=False):
            if expected["status"] != "ok" or case["fuel_model"] in reference.MOE_PERCENT_CONTROLS:
                continue
            actual = _port(case, backend, runners)
            differences = check_values(actual, expected["output"], fields)
            records.append(dict(backend=backend, id=case["id"], actual=actual, differences=differences))
            total += 1
        for case, expected in zip(controls["cases"], controls["results"], strict=False):
            actual = _port(case, backend, runners)
            original = _port(numbered[case["id"].removeprefix("custom-")], backend, runners)
            if any(struct.pack("!d", actual[field]) != struct.pack("!d", original[field]) for field in reference.FIELDS
            ):
                raise AssertionError(f"{backend}/{case['id']}: numbered/custom definition differs")
            source_id = case["id"].removeprefix("custom-")
            documented.append(dict(backend=backend, id=source_id, control=case["id"],
                                   reason="numbered C++ record uses inconsistent extinction units",
                                   numbered_cpp=raw_results[source_id]["output"], numbered_port=original))
            differences = check_values(actual, expected["output"], fields)
            records.append(dict(backend=backend, id=case["id"], actual=actual, differences=differences))
            total += 1
            paired += 1
        if total != 583 or paired != 139:
            raise ValueError("Incomplete port calibration")
        counts[backend] = dict(cases=total, fields=total * len(reference.FIELDS), paired_cases=paired,
                               original_unit_difference_cases=139)
    result = dict(schema_version=1, kind="ports", upstream_sha=reference.PIN, passed=True,
                  parity_approved=False, artifacts=_artifacts(root), fields=fields, counts=counts,
                  platform=platform.platform(), system=platform.system(), machine=platform.machine(),
                  platform_key=platform_key, python=sys.version, numpy=np.__version__, observations=records,
                  documented_differences=documented,
                  libc_family="musl" if platform_key == "linux-musl-x86_64" else platform.libc_ver()[0])
    if 'rust' in backends:
        from behave_fire import _rust

        result.update(binding_api_version=_rust.__binding_api_version__,
                      extension_sha256=reference._digest(Path(_rust.__file__)))
    for operation in (moisture, fire_size, ignite, mortality, safety, slope_tool, spot):
        result[operation.DATASET.name] = dict(
            artifacts=operation.artifacts(root), observations=operation.measure_ports(root, backends)
        )
    reference._json(report, result)
    return result


if __name__ == "__main__":
    main()
