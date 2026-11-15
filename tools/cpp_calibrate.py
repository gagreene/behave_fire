"""Explicit cross-compiler qualification and conditional Surface-policy activation."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import cpp_controls as controls
import cpp_reference as reference


def _compiler_environment():
    """Discover existing compiler tools without installing or exposing environment secrets.

    :return: CMake executable, generator arguments, compiler arguments, and child environment.
    """
    environment = dict(os.environ)
    if os.name != "nt":
        cmake, compiler = shutil.which("cmake"), shutil.which("g++")
        if not cmake or not compiler:
            raise ValueError("CMake and an existing GNU C++ compiler are required")
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []
        return cmake, generator, ["-DCMAKE_CXX_COMPILER=" + compiler], environment
    installer = Path(environment.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = subprocess.check_output([str(installer), "-latest", "-products", "*", "-requires",
                                            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                            "-property", "installationPath"], text=True).strip()
    if not installation:
        raise ValueError("MSVC discovery through vswhere failed")
    installation = Path(installation)
    script = installation / "Common7/Tools/VsDevCmd.bat"
    # Capture compiler environment in memory only; never persist or print tokens.
    command = f'call "{script}" -no_logo -arch=x64 -host_arch=x64 >nul && set'
    output = subprocess.check_output(f'cmd.exe /d /s /c "{command}"', text=True)
    for line in output.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            environment[key] = value
    environment = {key.upper(): value for key, value in environment.items()}
    for key in ('CL', '_CL_', 'CXXFLAGS'):
        environment.pop(key, None)
    cmake = installation / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    ninja = installation / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
    return str(cmake), ["-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + str(ninja)], ["-DCMAKE_CXX_COMPILER=cl.exe"], environment

def _context(root):
    """Record the tested source and CI identity without treating it as numerical proof.

    :param root: Source checkout root.
    :return: Commit, manual-run identity, and qualified driver-source metadata.
    """
    commit = os.environ.get("GITHUB_SHA") or subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    ci = os.environ.get("GITHUB_ACTIONS") == "true"
    run = os.environ.get("GITHUB_RUN_ID", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    return dict(source_commit=commit, evidence_origin="github_actions" if ci else "local",
                ci_event=os.environ.get("GITHUB_EVENT_NAME") if ci else None,
                ci_run_url=f"https://github.com/{repository}/actions/runs/{run}" if ci else None,
                source_files=_sources(root),
                overlay_sha256=reference._prepared_build(root)["prepared_overlay_sha256"])

def _requirements(root):
    """Keep the owner-approved compiler/platform matrix from being reduced by metadata.

    :param root: Checkout holding the conditional policy.
    :return: Exact required compiler and native-platform evidence keys.
    """
    policy = reference._document(Path(root) / "tests/data/cpp_reference/tolerance-policy.json")
    expected = dict(compilers=["GNU", "MSVC"], platforms=[
        "linux-x86_64", "linux-aarch64", "linux-musl-x86_64", "macos-x86_64", "macos-arm64",
        "windows-x86_64", "windows-arm64"])
    if (policy.get("activation_requirements") != expected
            or policy.get("documented_unit_source_models") != sorted(reference.MOE_PERCENT_CONTROLS)):
        raise ValueError("Owner-approved calibration matrix or unit disposition changed")
    return {"cpp-" + key for key in expected["compilers"]} | {"ports-" + key for key in expected["platforms"]}

def _run(command, directory, name, environment=None):
    """Run one checked native command while retaining bounded console output in files.

    :param command: Argument vector, without shell interpolation.
    :param directory: Out-of-tree evidence directory.
    :param name: Log stem.
    :param environment: Optional compiler environment passed only to the child.
    :return: None; raises with the log location if the command fails.
    """
    directory = Path(directory)
    with (directory / (name + ".log")).open("wb") as output:
        process = subprocess.run([str(item) for item in command], cwd=directory, env=environment,
                                 stdout=output, stderr=subprocess.STDOUT, check=False)
    (directory / (name + ".exit")).write_text(str(process.returncode) + "\n", encoding="ascii")
    if process.returncode != 0:
        raise ValueError(f"{name} failed with exit {process.returncode}; see {directory / (name + '.log')}")

def _sources(root):
    """Bind calibration to the exact production, build, and measurement sources.

    :param root: Source checkout, with portable normalized line endings.
    :return: Relative-path SHA256 inventory, excluding activation metadata.
    """
    root = Path(root)
    paths = set()
    for pattern in ('src/behave_fire/**/*.py', 'rust/crates/*/src/**/*.rs', 'rust/**/Cargo.toml',
                    'rust/Cargo.lock', '.cargo/*', 'rust/.cargo/*', 'rust/crates/*/.cargo/*',
                    'pyproject.toml', 'uv.lock', 'tools/*.py', '.github/workflows/*.yml'):
        paths.update(path for path in root.glob(pattern) if path.is_file() and 'target' not in path.parts)
    return {path.relative_to(root).as_posix(): reference._text_digest(path) for path in sorted(paths)}


def _validate_observations(root, report):
    """Recheck full evidence rather than accepting a passed flag or a summary alone.

    :param root: Checkout holding the immutable numerical references.
    :param report: One compiler or port report.
    :return: Compact validated count summary.
    """
    data = reference.load_reference(root, measurement=True)
    fields = controls.calibration_policy(root)
    if report.get("kind") == "ports":
        key = report.get("platform_key", "")
        expected_system = "Windows" if key.startswith("windows-") else "Darwin" if key.startswith("macos-") else "Linux"
        arm = key.endswith(("arm64", "aarch64"))
        machines = ("arm64", "aarch64") if arm else ("x86_64", "amd64", "x64")
        if (report.get("system") != expected_system or report.get("machine", "").lower() not in machines
                or (key == "linux-musl-x86_64" and report.get("libc_family") != "musl")
                or report.get("binding_api_version") != 4
                or not re.fullmatch(r"[0-9a-f]{64}", report.get("extension_sha256", ""))):
            raise ValueError("Inconsistent native calibration platform")
        expected = {case["id"]: row["output"] for case, row in zip(data["cases"], data["results"])
                    if row["status"] == "ok" and case["fuel_model"] not in reference.MOE_PERCENT_CONTROLS}
        expected.update({row["id"]: row["output"] for row in data["controls"]["results"]})
        required = {(backend, identity) for backend in ("python", "rust") for identity in expected}
        observed = set()
        for row in report.get("observations", []):
            key = (row.get("backend"), row.get("id"))
            if key in observed or key not in required:
                raise ValueError("Duplicate or unknown calibration observation")
            controls.check_values(row["actual"], expected[row["id"]], fields)
            observed.add(key)
        if observed != required:
            raise ValueError("Incomplete platform calibration observations")
        documented = report.get("documented_differences", [])
        if len(documented) != 278 or any(report["counts"].get(backend) != dict(
                cases=583, fields=2915, paired_cases=139, original_unit_difference_cases=139)
                for backend in ("python", "rust")):
            raise ValueError("Incomplete numbered/custom calibration accounting")
        raw = {row["id"]: row for row in data["results"]}
        actual = {(row["backend"], row["id"]): row["actual"] for row in report["observations"]}
        pairs = set()
        for row in documented:
            key = (row["backend"], row["id"])
            if (key in pairs or row["control"] != "custom-" + row["id"]
                    or row["numbered_cpp"] != raw[row["id"]]["output"]
                    or set(row["numbered_port"]) != set(reference.FIELDS)
                    or any(struct.pack('!d', row['numbered_port'][field]) != struct.pack(
                        '!d', actual[row['backend'], row['control']][field]) for field in reference.FIELDS)):
                raise ValueError("Numbered/custom evidence changed")
            pairs.add(key)
        return dict(cases_per_backend=583, fields_per_backend=2915, paired_cases_per_backend=139)
    if report.get("kind") != "cpp":
        raise ValueError("Unknown calibration report")
    expected = {row["id"]: row for row in [*data["results"], *data["controls"]["results"]]}
    observed = set()
    for row in report.get("observations", []):
        identity = row.get("id")
        if identity in observed or identity not in expected:
            raise ValueError("Duplicate or unknown compiler calibration case")
        wanted, actual = expected[identity], row["actual"]
        if set(actual) != set(wanted) or any(actual[key] != wanted[key] for key in wanted if key != "output"):
            raise ValueError("Compiler calibration status/schema changed")
        if wanted["status"] == "ok":
            controls.check_values(actual["output"], wanted["output"], fields)
        observed.add(identity)
    protocol = report.get("protocol", {})
    if (observed != set(expected) or protocol.get("negative_cases") != 62
            or protocol.get("negative_upstream_entries") != 0 or protocol.get("upstream_trap_exit") != 86
            or protocol.get("geometry_trap_exit") != 87 or protocol.get("duplicate_ids_rejected") is not True
            or protocol.get("state_sequence_checked") is not True
            or protocol.get("custom_negative_cases") != 18 or protocol.get("custom_negative_upstream_entries") != 0
            or protocol.get("custom_upstream_trap_exit") != 86 or protocol.get("custom_geometry_trap_exit") != 87
            or protocol.get("custom_duplicate_ids_rejected") is not True
            or protocol.get("custom_state_sequence_checked") is not True):
        raise ValueError("Incomplete compiler calibration or safety controls")
    return dict(cases=773, available=722, fields=3610, unavailable=51)

def activate(root, paths, reason):
    """Activate only the unchanged owner-approved caps after complete qualification.

    :param root: Source checkout root.
    :param paths: All full reports from the same manual CI run.
    :param reason: Explicit recorded reason for metadata-only activation.
    :return: Compact qualification record; writes reviewed metadata and crate copies.
    """
    from export_cpp_crate import export

    if not reason.strip():
        raise ValueError("Calibration activation needs a reason")
    root = Path(root)
    record = validate_evidence(root, paths)
    validate_qualification(root, record)
    record["activation_reason"] = reason
    reference._json(root / "tests/data/cpp_reference/calibration/qualification.json", record)
    policy_path = root / "tests/data/cpp_reference/tolerance-policy.json"
    policy = reference._document(policy_path)
    policy["approved"] = True
    policy["qualification"] = record
    policy["approval_gate"] = "Conditionally owner-approved caps qualified by complete manual CI evidence"
    reference._json(policy_path, policy)
    for dataset in ("surface", "surface_custom"):
        export(root / "tests/data/cpp_reference", root / "rust/crates/behave-fire/tests/data/cpp_reference" / dataset,
               dataset=dataset)
    reference.sync_hashes(root, reason)
    reference.load_reference(root)
    return record

def build_driver(root, directory, expected_compiler=None):
    """Build the pinned C++ out of tree with an already-installed compiler.

    :param root: Source checkout root with the hard-coded pinned dependency.
    :param directory: Dedicated evidence directory outside the checkout.
    :param expected_compiler: Optional GNU/MSVC matrix label, verified after discovery.
    :return: Built executable and its source/compiler identity record.
    """
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if directory.is_relative_to(root):
        raise ValueError("C++ calibration builds must remain outside the checkout")
    reference.require_live_checkout(root)
    directory.mkdir(parents=True, exist_ok=True)
    cmake, generator, compiler, environment = _compiler_environment()
    build = directory / "build"
    _run([cmake, "-S", root / "reference/behave_cpp_overlay", "-B", build, *generator, *compiler,
          "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=", "-DINCLUDE_EXRATE=ON"], directory, "configure", environment)
    _run([cmake, "--build", build, "--target", "behave_ref_driver", "--parallel", "4"],
         directory, "build", environment)
    metadata = list((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(metadata) != 1:
        raise ValueError("Ambiguous compiler identity")
    source = metadata[0].read_text(encoding="utf-8")
    compiler_id = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^\"]+)"\)', source).group(1)
    version = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^\"]+)"\)', source).group(1)
    if compiler_id not in ("GNU", "MSVC") or expected_compiler not in (None, compiler_id):
        raise ValueError("Unexpected calibration compiler")
    driver = build / ("behave_ref_driver.exe" if os.name == "nt" else "behave_ref_driver")
    record = dict(upstream_sha=reference.PIN, driver_sha256=reference._digest(driver),
                  driver_source_sha256_lf=reference._text_digest(root / "reference/behave_cpp_overlay/driver.cpp"),
                  compiler=dict(id=compiler_id, version=version), **_context(root))
    reference._json(directory / "build-receipt.json", record)
    return driver, record

def main():
    """Run only an explicit compiler, platform, or complete-evidence activation command.

    :return: None; writes operator-selected reports or conditionally approved metadata.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpp", action="store_true")
    group.add_argument("--ports", action="store_true")
    group.add_argument("--activate", action="store_true")
    parser.add_argument("--root", type=Path, default=reference.ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compiler-id", choices=("GNU", "MSVC"))
    parser.add_argument("--platform-key")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.cpp:
        if not args.out:
            parser.error("--cpp requires an out-of-tree --out directory")
        result = measure_cpp(args.root, args.out, args.compiler_id)
        print(json.dumps(result["summary"]))
    elif args.ports:
        if not args.report:
            parser.error("--ports requires --report")
        result = controls.measure_ports(args.root, args.report, expected_platform=args.platform_key)
        result.update(_context(args.root))
        reference._json(args.report, result)
        print(json.dumps(result["counts"]))
    else:
        if not args.evidence or not args.reason:
            parser.error("--activate requires complete --evidence and --reason")
        paths = sorted(args.evidence.rglob("*-calibration.json"))
        result = activate(args.root, paths, args.reason)
        print("Qualified " + str(len(result["reports"])) + " calibration reports")

def measure_cpp(root, directory, expected_compiler=None):
    """Measure compiler reproducibility and all original guarded driver contracts.

    :param root: Source checkout root.
    :param directory: Out-of-tree evidence directory.
    :param expected_compiler: Expected compiler family for this matrix job.
    :return: Complete compiler evidence, without activating parity.
    """
    from check_cpp_driver import check_driver

    root, directory = Path(root).resolve(), Path(directory).resolve()
    driver, build = build_driver(root, directory, expected_compiler)
    data = reference.load_reference(root, measurement=True)
    protocol = check_driver(driver, directory / "protocol", root, build_receipt=build, exact_replay=False)
    observations = []
    for dataset, cases, results in (("surface", data["cases"], data["results"]),
                                    ("surface_custom", data["controls"]["cases"], data["controls"]["results"])):
        output = directory / (dataset + ".jsonl")
        _run([driver, root / "tests/data/cpp_reference" / dataset / "cases.jsonl", output], directory, dataset)
        actual = reference._rows(output)
        if [row["id"] for row in actual] != [case["id"] for case in cases] or len(actual) != len(results):
            raise ValueError("Compiler calibration lost cases")
        observations.extend(dict(id=row["id"], dataset=dataset, actual=row) for row in actual)
    report = dict(schema_version=1, kind="cpp", upstream_sha=reference.PIN, passed=True,
                  parity_approved=False, artifacts=controls._artifacts(root), fields=controls.calibration_policy(root),
                  compiler=build["compiler"], protocol=protocol, observations=observations, **_context(root))
    report["summary"] = _validate_observations(root, report)
    reference._json(directory / (build["compiler"]["id"] + "-calibration.json"), report)
    return report

def validate_evidence(root, paths):
    """Require both C++ compilers and all seven real native-wheel platforms.

    :param root: Checkout holding unchanged inputs and approved conditional caps.
    :param paths: Full evidence files from one manual GitHub Actions run.
    :return: Compact validated qualification record, not an approval bit alone.
    """
    if len(paths) != 9:
        raise ValueError("Complete calibration requires nine reports")
    root = Path(root)
    required = _requirements(root)
    records, commits, runs = {}, set(), set()
    for path in paths:
        report = reference._document(path)
        key = ("cpp-" + report.get("compiler", {}).get("id", "")) if report.get("kind") == "cpp" else "ports-" + report.get("platform_key", "")
        if (key not in required or key in records or report.get("passed") is not True
                or report.get("parity_approved") is not False or report.get("upstream_sha") != reference.PIN
                or report.get("artifacts") != controls._artifacts(root)
                or report.get("fields") != controls.calibration_policy(root)
                or report.get("overlay_sha256") != reference._prepared_build(root)["prepared_overlay_sha256"]
                or report.get("source_files") != _sources(root)
                or report.get("evidence_origin") != "github_actions" or report.get("ci_event") != "workflow_dispatch"
                or not re.fullmatch(r"[0-9a-f]{40}", report.get("source_commit", ""))
                or not re.fullmatch(r"https://github.com/gagreene/behave_fire/actions/runs/[0-9]+", report.get("ci_run_url", ""))):
            raise ValueError("Untrusted or incomplete calibration identity")
        summary = _validate_observations(root, report)
        records[key] = dict(sha256=reference._digest(path), summary=summary)
        commits.add(report["source_commit"])
        runs.add(report["ci_run_url"])
    if set(records) != required or len(commits) != 1 or len(runs) != 1:
        raise ValueError("Mixed or incomplete calibration matrix")
    return dict(schema_version=1, calibration_complete=True, upstream_sha=reference.PIN,
                artifacts=controls._artifacts(root), fields=controls.calibration_policy(root), reports=records,
                source_commit=commits.pop(), ci_run_url=runs.pop(),
                source_files=_sources(root),
                overlay_sha256=reference._prepared_build(root)["prepared_overlay_sha256"])

def validate_qualification(root, record):
    """Verify shipped qualification metadata before allowing active comparisons.

    :param root: Checkout holding current numerical input artifacts.
    :param record: Hash-checked qualification produced from full evidence.
    :return: None; rejects incomplete, stale, or changed bounds and identities.
    """
    required = _requirements(root)
    if record.get("source_files") != _sources(root):
        raise ValueError("Calibration source inventory changed")
    if (record.get("calibration_complete") is not True or record.get("upstream_sha") != reference.PIN
            or record.get("artifacts") != controls._artifacts(root)
            or record.get("fields") != controls.calibration_policy(root)
            or record.get("overlay_sha256") != reference._prepared_build(root)["prepared_overlay_sha256"]
            or set(record.get("reports", {})) != required
            or not re.fullmatch(r"[0-9a-f]{40}", record.get("source_commit", ""))
            or not re.fullmatch(r"https://github.com/gagreene/behave_fire/actions/runs/[0-9]+", record.get("ci_run_url", ""))):
        raise ValueError("Incomplete or stale calibration qualification")
    for key, report in record["reports"].items():
        expected = dict(cases=773, available=722, fields=3610, unavailable=51) if key.startswith("cpp-") else dict(
            cases_per_backend=583, fields_per_backend=2915, paired_cases_per_backend=139)
        if not re.fullmatch(r"[0-9a-f]{64}", report.get("sha256", "")) or report.get("summary") != expected:
            raise ValueError("Invalid calibration summary or report hash")


if __name__ == "__main__":
    main()
