"""Write and verify local C++ comparison receipts bound to a reviewed revision.

A receipt records one platform's full local comparison of the Python and Rust ports with the
pinned Behave C++ outputs. It binds the results to the tested commit and tree, the qualified
source inventory, the fixture inventory, the native extension, the toolchains, and the host,
and it names the archived full reports by SHA-256. Verification reads those archived reports,
re-applies the existing numerical rules (Surface ``1e-12`` caps with the documented load
exceptions, and exact binary64 FireSize, Ignite, and Mortality results), and rejects a claimed
platform without valid evidence.

Receipts live in ``tests/data/cpp_receipts/``, outside ``tests/data/cpp_reference/``: the
reference hasher lists every file under that tree in ``hashes.json``, so a receipt stored there
would change the fixture inventory digest it records. Writing or checking a receipt never
activates a qualification; approval of the platform claims and activation are separate steps.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import cpp_calibrate as calibrate
import cpp_controls as controls
import cpp_fire_size as fire_size
import cpp_ignite as ignite
import cpp_mortality as mortality
import cpp_reference as reference

RECEIPTS = Path("tests/data/cpp_receipts")
CLAIMS = RECEIPTS / "claims.json"
PLATFORMS = ("linux-x86_64", "linux-aarch64", "linux-musl-x86_64", "macos-x86_64", "macos-arm64",
             "windows-x86_64", "windows-arm64")
BACKENDS = ("python", "rust")
EXACT = {"fire_size": (fire_size, 9), "ignite": (ignite, 8), "mortality": (mortality, 6)}
MEASURED = ("crown", "fine_dead_fuel_moisture", "safety", "slope_tool", "spot", "units", "vapor_pressure_deficit")
RUST_ENFORCED = {"CPP_SURFACE_ENFORCED": "compared=2855 documented=60", "CPP_FIRE_SIZE_ENFORCED": "exact_binary64=9",
                 "CPP_IGNITE_ENFORCED": "exact_binary64=8", "CPP_MORTALITY_ENFORCED": "exact_binary64=6"}
RUST_TESTS = {"cpp_surface_observations_follow_the_qualified_policy": 1,
              "explicit_percent_controls_match_numbered_physics_and_consume_all_cpp_fields": 1,
              "cpp_fire_size_observations_follow_the_qualified_policy": 1,
              "cpp_ignite_observations_follow_the_qualified_policy": 1,
              "cpp_mortality_observations_follow_the_qualified_policy": 1,
              "reaction_order_matches_pinned_cpp_near_extinction": 1,
              "perturbed_cpp_copy_fails_the_surface_caps": 1,
              "perturbed_cpp_copy_fails_the_exact_regression": 3}
SCHEMA = 1


def _bound_revision(root, receipt):
    """Require the checkout to be the receipt's commit, apart from receipt files.

    :param root: Repository root.
    :param receipt: Receipt document.
    :return: None; raises when the commit, tree, or later changes do not match.
    """
    commit = receipt["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Receipt commit is not a full SHA-1")
    if _git(root, "rev-parse", commit + "^{tree}") != receipt["source_tree"]:
        raise ValueError("Receipt tree does not match its commit")
    changed = _git(root, "diff", "--name-only", commit).splitlines()
    changed += _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    outside = sorted(name for name in changed if not name.startswith(RECEIPTS.as_posix() + "/"))
    if outside:
        raise ValueError(f"Checkout differs from the receipt commit outside receipts: {outside[:5]}")


def _clean_tree(root):
    """Require a checkout with no tracked or untracked change outside the receipt directory.

    :param root: Repository root.
    :return: None; raises on any other change.
    """
    lines = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    dirty = sorted(line[3:] for line in lines if not line[3:].startswith(RECEIPTS.as_posix() + "/"))
    if dirty:
        raise ValueError(f"Receipts need a clean checkout; changed: {dirty[:5]}")


def _command(*args):
    """Return a tool's version output, or ``None`` when the tool is unavailable.

    :param args: Command and arguments.
    :return: Trimmed standard output, or ``None``.
    """
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git(root, *args):
    """Run a read-only Git command in the checkout.

    :param root: Repository root.
    :param args: Git arguments.
    :return: Trimmed standard output.
    """
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _host():
    """Describe the host without treating it as numerical evidence.

    :return: Operating system, CPU, and selected instruction-set features.
    """
    features = {}
    try:
        from numpy._core import _multiarray_umath

        available = _multiarray_umath.__cpu_features__
        features = {name: bool(available.get(name)) for name in ("FMA3", "AVX2", "AVX512F", "ASIMD")}
    except (ImportError, AttributeError):
        pass
    return dict(system=platform.system(), release=platform.release(), version=platform.version(),
                machine=platform.machine(), processor=platform.processor(), cpu_features=features)


def _read_json(path):
    """Read strict JSON with unique keys and no nonfinite constants.

    :param path: JSON file.
    :return: Decoded document.
    """
    return json.loads(Path(path).read_bytes(), object_pairs_hook=reference._unique_object,
                      parse_constant=reference._reject_constant)


def _toolchains():
    """Record the interpreter, NumPy, and Rust toolchain identities.

    :return: Version strings; unavailable tools are ``None``.
    """
    import numpy

    return dict(python=sys.version, numpy=numpy.__version__, rustc=_command("rustc", "-Vv"),
                cargo=_command("cargo", "-V"), maturin=_command("maturin", "-V"),
                rustflags=os.environ.get("RUSTFLAGS"))


def _validate_cargo_log(text):
    """Require the Rust core comparisons and their deliberate-failure checks to have run and passed.

    :param text: Output of ``cargo test -p behave-fire --locked --no-fail-fast -- --ignored --nocapture``.
    :return: Counts of the enforced Rust assertions.
    """
    results = re.findall(r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", text, flags=re.M)
    if not results or any(status != "ok" or int(failed) for status, _, failed, _ in results):
        raise ValueError("Rust comparison run did not pass")
    passed = {}
    for name in re.findall(r"^test (\w+) \.\.\. ok$", text, flags=re.M):
        passed[name] = passed.get(name, 0) + 1
    if passed != RUST_TESTS:
        raise ValueError(f"Rust comparison tests differ from the required set: {sorted(passed)}")
    enforced = dict(re.findall(r"^(CPP_\w+_ENFORCED) (.+)$", text, flags=re.M))
    if enforced != RUST_ENFORCED:
        raise ValueError(f"Rust comparisons did not enforce the expected counts: {enforced}")
    return dict(tests=sum(RUST_TESTS.values()), enforced=enforced)


def _validate_ports_report(root, report, platform_key):
    """Re-apply the numerical rules to one archived local ports report.

    :param root: Repository root with the pinned observations.
    :param report: Decoded ports report.
    :param platform_key: Platform the receipt claims.
    :return: Compared counts for Surface and the exact modules, and the measurement-only modules.
    """
    data = reference.load_reference(root, measurement=True)
    if (report.get("kind") != "ports" or report.get("platform_key") != platform_key or report.get("passed") is not True
            or report.get("parity_approved") is not False or report.get("upstream_sha") != reference.PIN
            or report.get("artifacts") != controls._artifacts(root)
            or report.get("fields") != controls.calibration_policy(root)
            or report.get("overlay_sha256") != reference._prepared_build(root)["prepared_overlay_sha256"]
            or report.get("source_files") != calibrate._sources(root) or report.get("evidence_origin") != "local"):
        raise ValueError("Ports report identity does not match this revision")
    surface = calibrate._validate_observations(root, report)
    exact = {}
    for name, (module, per_backend) in EXACT.items():
        measurement = report.get(name, {})
        expected = module.load_observations(root)
        policy = data["policy"][name]
        if set(measurement) != {"artifacts", "observations"} or measurement["artifacts"] != module.artifacts(root):
            raise ValueError(f"{name}: report used a different dataset")
        if set(measurement["observations"]) != set(BACKENDS):
            raise ValueError(f"{name}: missing backend observations")
        for backend in BACKENDS:
            rows = measurement["observations"][backend]
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError(f"{name}: incomplete case identities for {backend}")
            compared = sum(module.check_exact(row["output"], original["output"], policy)
                           for row, original in zip(rows, expected, strict=True))
            if compared != per_backend:
                raise ValueError(f"{name}: {compared} exact comparisons for {backend}, expected {per_backend}")
        exact[name] = per_backend
    for name in MEASURED:
        measurement = report.get(name, {})
        if (set(measurement) != {"artifacts", "observations"} or set(measurement["observations"]) != set(BACKENDS)
                or not all(measurement["observations"][backend] for backend in BACKENDS)):
            raise ValueError(f"{name}: missing measurement-only observations")
    return dict(surface=surface, exact_binary64_per_backend=exact, measurement_only=list(MEASURED))


def check_claims(root, archive):
    """Require a valid receipt for every claimed platform, then the owner's approval of the claims.

    :param root: Repository root.
    :param archive: Directory holding the archived full reports.
    :return: Verified summaries per claimed platform; raises on missing evidence or missing approval.
    """
    root = Path(root)
    claims = _read_json(root / CLAIMS)
    if set(claims) != {"schema_version", "approved", "platforms", "archive_location"} or claims["schema_version"] != 1:
        raise ValueError("Malformed platform claims")
    platforms = claims["platforms"]
    if not platforms or len(set(platforms)) != len(platforms) or not set(platforms) <= set(PLATFORMS):
        raise ValueError("Platform claims must name distinct supported platforms")
    missing = [key for key in platforms if not (root / RECEIPTS / f"{key}.json").is_file()]
    if missing:
        raise ValueError(f"Missing platform evidence: {missing}")
    verified = {key: check_receipt(root, root / RECEIPTS / f"{key}.json", archive) for key in platforms}
    if claims["approved"] is not True:
        raise ValueError("Platform claims are not approved")
    return verified


def check_receipt(root, path, archive):
    """Verify one receipt and the archived reports it names.

    :param root: Repository root at the receipt's commit (receipt files may differ).
    :param path: Receipt file.
    :param archive: Directory holding the archived full reports.
    :return: The receipt's verified results.
    """
    root, path, archive = Path(root), Path(path), Path(archive)
    base = (root / "tests/data/cpp_reference").resolve()
    if path.resolve().is_relative_to(base) or any(name.startswith(RECEIPTS.as_posix() + "/")
                                                  for name in _read_json(base / "hashes.json")):
        raise ValueError("Receipts must stay outside the hashed fixture inventory")
    receipt = _read_json(path)
    expected_keys = {"schema_version", "evidence_origin", "platform_key", "created", "operator", "source_commit",
                     "source_tree", "source_files", "fixture_inventory_sha256", "artifacts", "datasets",
                     "upstream_sha", "overlay_sha256", "public_api", "binaries", "toolchains", "host", "reports",
                     "results"}
    if set(receipt) != expected_keys or receipt["schema_version"] != SCHEMA or receipt["evidence_origin"] != "local":
        raise ValueError("Malformed receipt")
    if path.stem != receipt["platform_key"] or receipt["platform_key"] not in PLATFORMS:
        raise ValueError("Receipt platform does not match its file name")
    _bound_revision(root, receipt)
    datasets = {name: module.artifacts(root) for name, (module, _) in EXACT.items()}
    if (receipt["source_files"] != calibrate._sources(root)
            or receipt["fixture_inventory_sha256"] != reference._digest(base / "hashes.json")
            or receipt["artifacts"] != controls._artifacts(root) or receipt["datasets"] != datasets
            or receipt["upstream_sha"] != reference.PIN
            or receipt["overlay_sha256"] != reference._prepared_build(root)["prepared_overlay_sha256"]
            or receipt["public_api"] != calibrate._expected_public_api(root)):
        raise ValueError("Receipt is not bound to the current sources and fixtures")
    if set(receipt["reports"]) != {"ports", "rust"}:
        raise ValueError("Receipt must name the ports report and the Rust comparison log")
    documents = {}
    for kind, entry in receipt["reports"].items():
        file = archive / entry["file"]
        if Path(entry["file"]).name != entry["file"] or not file.is_file():
            raise ValueError(f"Archived {kind} report is missing: {entry['file']}")
        if reference._digest(file) != entry["sha256"]:
            raise ValueError(f"Archived {kind} report differs from the receipt")
        documents[kind] = file
    report = _read_json(documents["ports"])
    if (report.get("source_commit") != receipt["source_commit"]
            or report.get("extension_sha256") != receipt["binaries"]["extension_sha256"]
            or report.get("binding_api_version") != receipt["binaries"]["binding_api_version"]):
        raise ValueError("Ports report was not produced by the receipt's commit and extension")
    results = dict(ports=_validate_ports_report(root, report, receipt["platform_key"]),
                   rust=_validate_cargo_log(documents["rust"].read_text(encoding="utf-8")))
    if json.loads(json.dumps(results)) != receipt["results"]:
        raise ValueError("Receipt results differ from the archived reports")
    return results


def main(argv=None):
    """Write a receipt, check one receipt, or check every claimed platform.

    :param argv: Optional command-line arguments.
    :return: None; ``--write`` writes the archived reports and the receipt file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", type=Path, metavar="RECEIPT")
    group.add_argument("--check-claims", action="store_true")
    parser.add_argument("--root", type=Path, default=reference.ROOT)
    parser.add_argument("--archive", type=Path, required=True, help="directory of archived full reports")
    parser.add_argument("--operator", default="127547449+gagreene@users.noreply.github.com")
    arguments = parser.parse_args(argv)
    if arguments.write:
        print(json.dumps(write_receipt(arguments.root, arguments.archive, arguments.operator)["results"], indent=2))
    elif arguments.check:
        print(json.dumps(check_receipt(arguments.root, arguments.check, arguments.archive), indent=2))
    else:
        print(json.dumps(check_claims(arguments.root, arguments.archive), indent=2))


def write_receipt(root, archive, operator):
    """Run the full local comparison on this host, archive its reports, and write the receipt.

    :param root: Clean repository root at the commit to record, with the native extension built from it.
    :param archive: Directory that receives the full reports (outside the repository).
    :param operator: Identity recorded in the receipt (use the noreply address).
    :return: The receipt; writes the two archived reports and ``tests/data/cpp_receipts/<platform>.json``.
    """
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.is_relative_to(root):
        raise ValueError("Archive the full reports outside the repository")
    _clean_tree(root)
    commit = _git(root, "rev-parse", "HEAD")
    key = controls._platform_key()
    archive.mkdir(parents=True, exist_ok=True)
    ports = archive / f"{key}-{commit}-ports.json"
    rust = archive / f"{key}-{commit}-rust.log"
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    subprocess.run([sys.executable, "-B", str(root / "tools/cpp_calibrate.py"), "--ports", "--root", str(root),
                    "--report", str(ports), "--platform-key", key], check=True, env=environment)
    log = subprocess.run(["cargo", "test", "--manifest-path", str(root / "rust/Cargo.toml"), "-p", "behave-fire",
                          "--locked", "--no-fail-fast", "--", "--ignored", "--nocapture"], capture_output=True,
                         text=True, cwd=root)
    rust.write_text(log.stdout + log.stderr, encoding="utf-8")
    if log.returncode != 0:
        raise ValueError(f"Rust comparison run failed; see {rust}")
    report = _read_json(ports)
    results = dict(ports=_validate_ports_report(root, report, key),
                   rust=_validate_cargo_log(rust.read_text(encoding="utf-8")))
    base = root / "tests/data/cpp_reference"
    receipt = dict(
        schema_version=SCHEMA, evidence_origin="local", platform_key=key,
        created=datetime.now(UTC).isoformat(timespec="seconds"), operator=operator,
        source_commit=commit, source_tree=_git(root, "rev-parse", "HEAD^{tree}"),
        source_files=calibrate._sources(root), fixture_inventory_sha256=reference._digest(base / "hashes.json"),
        artifacts=controls._artifacts(root),
        datasets={name: module.artifacts(root) for name, (module, _) in EXACT.items()},
        upstream_sha=reference.PIN, overlay_sha256=reference._prepared_build(root)["prepared_overlay_sha256"],
        public_api=calibrate._expected_public_api(root),
        binaries=dict(extension_sha256=report["extension_sha256"], binding_api_version=report["binding_api_version"]),
        toolchains=_toolchains(), host=_host(),
        reports=dict(ports=dict(file=ports.name, sha256=reference._digest(ports)),
                     rust=dict(file=rust.name, sha256=reference._digest(rust))),
        results=json.loads(json.dumps(results)))
    target = root / RECEIPTS / f"{key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    check_receipt(root, target, archive)
    return receipt


if __name__ == "__main__":
    main()
