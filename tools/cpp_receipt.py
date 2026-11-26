"""Write and verify local C++ comparison receipts bound to a reviewed revision.

A receipt records one platform's full local comparison of the Python and Rust ports with the
pinned Behave C++ outputs. It binds the results to the tested commit and tree, the qualified
source inventory, the fixture inventory, the native extension, the toolchains, and the host,
and it names the archived full reports by SHA-256. Verification reads those archived reports,
re-applies the existing numerical rules (Surface ``1e-12`` caps for all available cases
and exact binary64 FireSize, Ignite, and Mortality results), and rejects a claimed
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
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener
from zipfile import ZipFile

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
RUST_ENFORCED = {"CPP_SURFACE_ENFORCED": "compared=2915", "CPP_FIRE_SIZE_ENFORCED": "exact_binary64=9",
                 "CPP_IGNITE_ENFORCED": "exact_binary64=8", "CPP_MORTALITY_ENFORCED": "exact_binary64=6"}
RUST_TESTS = {"cpp_surface_observations_follow_the_qualified_policy": 1,
              "explicit_percent_controls_match_numbered_physics_and_consume_all_cpp_fields": 1,
              "cpp_fire_size_observations_follow_the_qualified_policy": 1,
              "cpp_ignite_observations_follow_the_qualified_policy": 1,
              "cpp_mortality_observations_follow_the_qualified_policy": 1,
              "reaction_order_matches_pinned_cpp_near_extinction": 1,
              "perturbed_cpp_copy_fails_the_surface_caps": 1,
              "perturbed_cpp_copy_fails_the_exact_regression": 3}
SCHEMA = 2


class _HttpsRedirect(HTTPRedirectHandler):
    """Keep evidence archive downloads on HTTPS across redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Build an HTTPS redirect request or reject the redirect.

        :param req: Original request.
        :param fp: Response stream.
        :param code: Redirect status code.
        :param msg: Redirect status message.
        :param headers: Redirect response headers.
        :param newurl: Redirect URL resolved by urllib.
        :return: Request for the HTTPS destination.
        """
        target = urlsplit(newurl)
        if target.scheme != "https" or not target.hostname or target.username is not None:
            raise ValueError("Evidence archive redirects require HTTPS without credentials")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _approved_claims(root):
    """Read approved platform claims and require their receipt files.

    :param root: Repository root.
    :return: Claimed platforms and the approved archive location.
    """
    claims = _read_json(root / CLAIMS)
    if set(claims) != {"schema_version", "approved", "platforms", "archive_location"} or claims["schema_version"] != 1:
        raise ValueError("Malformed platform claims")
    platforms = claims["platforms"]
    if not platforms or len(set(platforms)) != len(platforms) or not set(platforms) <= set(PLATFORMS):
        raise ValueError("Platform claims must name distinct supported platforms")
    missing = [key for key in platforms if not (root / RECEIPTS / f"{key}.json").is_file()]
    if missing:
        raise ValueError(f"Missing platform evidence: {missing}")
    if claims["approved"] is not True:
        raise ValueError("Platform claims are not approved")
    return platforms, claims["archive_location"]


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
    changed = _git(root, "diff", "--no-renames", "--name-only", commit).splitlines()
    changed += _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    outside = sorted(name for name in changed if not name.startswith(RECEIPTS.as_posix() + "/"))
    if outside:
        raise ValueError(f"Checkout differs from the receipt commit outside receipts: {outside[:5]}")


def _build_command(target):
    """Return the Cargo command for an isolated native extension build.

    :param target: Rust host target triple.
    :return: Command to run from the extracted source snapshot.
    """
    command = ["cargo", "rustc", "--manifest-path", "rust/Cargo.toml", "-p", "behave-py",
               "--locked", "--release", "--target", target, "--target-dir", "target"]
    if target.endswith("-apple-darwin"):
        command.extend(["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"])
    if target.endswith("-linux-musl"):
        command.extend(["--", "-C", "target-feature=-crt-static"])
    return command


def _build_extension(root, commit, snapshot):
    """Build and stage the extension from the recorded Git revision.

    :param root: Repository holding the Git revision.
    :param commit: Full Git commit ID.
    :param snapshot: Empty temporary directory for sources and build outputs.
    :return: Staged extension, build provenance, toolchains, and Cargo environment.
    """
    source_zip = snapshot / "source.zip"
    subprocess.run(["git", "-C", str(root), "archive", "--format=zip",
                    f"--output={source_zip}", commit], check=True)
    with ZipFile(source_zip) as archive:
        archive.extractall(snapshot)
    environment = dict(os.environ, PYO3_PYTHON=sys.executable, RUSTC_WRAPPER="", RUSTC_WORKSPACE_WRAPPER="")
    sysroot = _command("rustc", "--print", "sysroot", cwd=snapshot, env=environment)
    if not sysroot:
        raise ValueError("Could not locate the Rust toolchain")
    rustc = Path(sysroot) / "bin" / ("rustc.exe" if os.name == "nt" else "rustc")
    environment["RUSTC"] = str(rustc)
    encoded_flags = environment.get("CARGO_ENCODED_RUSTFLAGS")
    if not encoded_flags:
        encoded_flags = "\x1f".join(environment.pop("RUSTFLAGS", "").split()) or "-Cdebuginfo=0"
    environment["CARGO_ENCODED_RUSTFLAGS"] = encoded_flags
    toolchains = _toolchains(snapshot, environment)
    host = re.search(r"^host: (\S+)$", toolchains["rustc"] or "", re.M)
    if not host or not toolchains["cargo"]:
        raise ValueError("Could not identify the Cargo toolchain and its host target")
    target = host[1]
    command = _build_command(target)
    subprocess.run(command, cwd=snapshot, check=True, env=environment)
    if target.endswith("-apple-darwin"):
        library = "lib_rust.dylib"
    elif "-windows-" in target:
        library = "_rust.dll"
    else:
        library = "lib_rust.so"
    extension = snapshot / "src/behave_fire" / ("_rust.pyd" if "-windows-" in target else "_rust.so")
    shutil.copyfile(snapshot / "target" / target / "release" / library, extension)
    provenance = dict(source_tree=_git(root, "rev-parse", commit + "^{tree}"), target=target, command=command)
    return extension, provenance, toolchains, environment


def _clean_tree(root):
    """Require a checkout with no tracked or untracked change outside the receipt directory.

    :param root: Repository root.
    :return: None; raises on any other change.
    """
    lines = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    dirty = sorted(line[3:] for line in lines if not line[3:].startswith(RECEIPTS.as_posix() + "/"))
    if dirty:
        raise ValueError(f"Receipts need a clean checkout; changed: {dirty[:5]}")


def _command(*args, cwd, env):
    """Return a tool's version output, or ``None`` when the tool is unavailable.

    :param args: Command and arguments.
    :param cwd: Directory where the command runs.
    :param env: Environment used to run the command.
    :return: Trimmed standard output, or ``None``.
    """
    try:
        return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _evidence_file(archive, relative):
    """Resolve a file inside one archived comparison attempt.

    :param archive: Approved archive directory.
    :param relative: Path relative to the archive root.
    :return: Resolved file path inside the archive.
    """
    path = Path(relative)
    resolved = (archive / path).resolve()
    if path.anchor or len(path.parts) != 2 or ".." in path.parts or not resolved.is_relative_to(archive):
        raise ValueError(f"Invalid archived evidence path: {relative}")
    return resolved


def _extract_release_archive(path, destination):
    """Extract a ZIP containing only comparison evidence in attempt directories.

    :param path: ZIP archive whose digest has already been verified.
    :param destination: Empty temporary directory for the evidence.
    :return: None; writes verified member paths under the destination.
    """
    with ZipFile(path) as bundle:
        seen = set()
        for member in bundle.infolist():
            name = member.filename.rstrip("/")
            pattern = (r"[a-z0-9_-]+" if member.is_dir()
                       else r"[a-z0-9_-]+/(?:ports\.json|rust\.log|_rust\.(?:pyd|so))")
            if not re.fullmatch(pattern, name) or name in seen:
                raise ValueError(f"Unsafe evidence archive member: {member.filename}")
            seen.add(name)
        bundle.extractall(destination)


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


def _toolchains(snapshot, environment):
    """Record the interpreter, NumPy, and Rust toolchain identities.

    :param snapshot: Source directory used by the build.
    :param environment: Environment passed to Cargo.
    :return: Version strings; unavailable tools are ``None``.
    """
    import numpy

    return dict(python=sys.version, numpy=numpy.__version__,
                rustc=_command(environment["RUSTC"], "-vV", cwd=snapshot, env=environment),
                cargo=_command("cargo", "-V", cwd=snapshot, env=environment),
                cargo_encoded_rustflags=environment["CARGO_ENCODED_RUSTFLAGS"])


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
    measured = (calibrate.crown, calibrate.moisture, calibrate.safety, calibrate.slope_tool, calibrate.spot,
                calibrate.units, calibrate.vpd)
    for module in measured:
        name = module.DATASET.name
        measurement = report.get(name, {})
        if (set(measurement) != {"artifacts", "observations"} or set(measurement["observations"]) != set(BACKENDS)
                or not all(isinstance(rows, list) and rows for rows in measurement["observations"].values())):
            raise ValueError(f"{name}: missing measurement-only observations")
        if measurement["artifacts"] != module.artifacts(root):
            raise ValueError(f"{name}: report used a different dataset")
        cases = [row["id"] for row in module.load_cases(root)]
        for backend, rows in measurement["observations"].items():
            if (any(not isinstance(row, dict) or set(row) != {"id", "output"} for row in rows)
                    or [row["id"] for row in rows] != cases):
                raise ValueError(f"{name}: incomplete case identities for {backend}")
            for row in rows:
                if module is calibrate.crown:
                    module._validate_output(row["id"], row["output"])
                elif module is calibrate.slope_tool:
                    module._validate_output(row["output"], module.CASE_FIELDS[row["id"]])
                else:
                    module._validate_output(row["output"])
    return dict(surface=surface, exact_binary64_per_backend=exact, measurement_only=list(MEASURED))


def _validate_provenance(receipt, report, archive):
    """Validate the recorded build context and archived extension bytes.

    :param receipt: Receipt containing the build and platform context.
    :param report: Archived ports report.
    :param archive: Resolved archive directory.
    :return: None; raises if the build or platform context disagrees.
    """
    binaries = receipt["binaries"]
    build = binaries.get("build", {})
    target = build.get("target", "")
    if (set(build) != {"source_tree", "target", "command"} or not re.fullmatch(r"[\w-]+", target)
            or build["source_tree"] != receipt["source_tree"] or build["command"] != _build_command(target)):
        raise ValueError("Missing or inconsistent extension build provenance")
    toolchains = receipt["toolchains"]
    if (set(toolchains) != {"python", "numpy", "rustc", "cargo", "cargo_encoded_rustflags"}
            or not all(isinstance(toolchains[key], str) and toolchains[key] for key in ("python", "numpy", "rustc",
                                                                                         "cargo"))
            or not re.match(r"rustc \d+\.\d+\.\d+", toolchains["rustc"])
            or not re.match(r"cargo \d+\.\d+\.\d+", toolchains["cargo"])
            or not isinstance(toolchains["cargo_encoded_rustflags"], str)
            or not toolchains["cargo_encoded_rustflags"]
            or f"host: {target}" not in toolchains["rustc"].splitlines()
            or any(toolchains[key] != report.get(key) for key in ("python", "numpy"))):
        raise ValueError("Missing or inconsistent toolchain metadata")
    key = receipt["platform_key"]
    family = {"windows": "-windows-", "macos": "-apple-darwin", "linux": "-linux-"}[key.split("-")[0]]
    arch = "aarch64" if key.endswith(("aarch64", "arm64")) else "x86_64"
    host = receipt["host"]
    if (set(host) != {"system", "release", "version", "machine", "processor", "cpu_features"}
            or not all(isinstance(host[field], str) and host[field] for field in ("system", "release", "version",
                                                                                   "machine"))
            or not isinstance(host["cpu_features"], dict)
            or set(host["cpu_features"]) != {"FMA3", "AVX2", "AVX512F", "ASIMD"}
            or any(type(value) is not bool for value in host["cpu_features"].values())
            or any(host[field] != report.get(field) for field in ("system", "machine"))
            or not target.startswith(arch + "-") or family not in target
            or (key.startswith("linux-") and ("-musl" in target) != ("musl" in key))):
        raise ValueError("Missing or inconsistent host and build target")
    binary = _evidence_file(archive, binaries["file"])
    if not binary.is_file() or reference._digest(binary) != binaries["extension_sha256"]:
        raise ValueError("Archived native extension is missing or differs from the receipt")


def check_claims(root, archive):
    """Check the owner's approval and verify each claimed platform using the approved archive.

    :param root: Repository root.
    :param archive: Directory holding the archived full reports.
    :return: Verified summaries per claimed platform; raises on missing evidence or missing approval.
    """
    root = Path(root)
    platforms, location = _approved_claims(root)
    if not isinstance(location, str) or not Path(location).is_absolute():
        raise ValueError("Approved claims require an absolute archive location")
    approved_archive = Path(location).resolve()
    if approved_archive != Path(archive).resolve() or not approved_archive.is_dir():
        raise ValueError("Evidence directory does not match the approved archive")
    if approved_archive.is_relative_to(root.resolve()):
        raise ValueError("The approved archive must be outside the checkout")
    return {key: check_receipt(root, root / RECEIPTS / f"{key}.json", approved_archive) for key in platforms}


def check_receipt(root, path, archive):
    """Verify one receipt and the archived reports it names.

    :param root: Repository root at the receipt's commit (receipt files may differ).
    :param path: Receipt file.
    :param archive: Directory holding the archived full reports.
    :return: The receipt's verified results.
    """
    root, path, archive = Path(root), Path(path), Path(archive).resolve()
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
        file = _evidence_file(archive, entry["file"])
        if not file.is_file():
            raise ValueError(f"Archived {kind} report is missing: {entry['file']}")
        if reference._digest(file) != entry["sha256"]:
            raise ValueError(f"Archived {kind} report differs from the receipt")
        documents[kind] = file
    binary = _evidence_file(archive, receipt["binaries"]["file"])
    binary_name = "_rust.pyd" if receipt["platform_key"].startswith("windows-") else "_rust.so"
    if (documents["ports"].name != "ports.json" or documents["rust"].name != "rust.log" or binary.name != binary_name
            or len({documents["ports"].parent, documents["rust"].parent, binary.parent}) != 1):
        raise ValueError("Archived reports and extension must come from one comparison attempt")
    report = _read_json(documents["ports"])
    if (report.get("source_commit") != receipt["source_commit"]
            or report.get("extension_sha256") != receipt["binaries"]["extension_sha256"]
            or report.get("binding_api_version") != receipt["binaries"]["binding_api_version"]):
        raise ValueError("Ports report was not produced by the receipt's commit and extension")
    _validate_provenance(receipt, report, archive)
    results = dict(ports=_validate_ports_report(root, report, receipt["platform_key"]),
                   rust=_validate_cargo_log(documents["rust"].read_text(encoding="utf-8")))
    if json.loads(json.dumps(results)) != receipt["results"]:
        raise ValueError("Receipt results differ from the archived reports")
    return results


def check_release_claims(root):
    """Download and verify the approved release archive and every claimed platform.

    :param root: Repository root.
    :return: Verified summaries per platform; raises if release evidence is invalid.
    """
    root = Path(root)
    platforms, location = _approved_claims(root)
    if not isinstance(location, dict) or set(location) != {"url", "sha256"}:
        raise ValueError("Release claims require an approved HTTPS archive and SHA-256")
    url, digest = location["url"], location["sha256"]
    endpoint = urlsplit(url) if isinstance(url, str) else None
    if (endpoint is None or endpoint.scheme != "https" or not endpoint.hostname or endpoint.username is not None
            or endpoint.fragment or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Release claims require an approved HTTPS archive and SHA-256")
    with TemporaryDirectory(prefix="bf-release-evidence-") as temporary:
        download = Path(temporary) / "evidence.zip"
        with build_opener(_HttpsRedirect()).open(url, timeout=30) as source, download.open("wb") as output:
            shutil.copyfileobj(source, output)
        if reference._digest(download) != digest:
            raise ValueError("Downloaded evidence archive SHA-256 differs from approved claims")
        archive = Path(temporary) / "archive"
        _extract_release_archive(download, archive)
        return {key: check_receipt(root, root / RECEIPTS / f"{key}.json", archive) for key in platforms}


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
    group.add_argument("--check-release-claims", action="store_true")
    parser.add_argument("--root", type=Path, default=reference.ROOT)
    parser.add_argument("--archive", type=Path, help="directory of archived full reports")
    parser.add_argument("--operator", default="127547449+gagreene@users.noreply.github.com")
    arguments = parser.parse_args(argv)
    if arguments.check_release_claims:
        if arguments.archive is not None:
            parser.error("--check-release-claims uses the approved archive URL")
        print(json.dumps(check_release_claims(arguments.root), indent=2))
        return
    if arguments.archive is None:
        parser.error("--archive is required for local receipt operations")
    if arguments.write:
        print(json.dumps(write_receipt(arguments.root, arguments.archive, arguments.operator)["results"], indent=2))
    elif arguments.check:
        print(json.dumps(check_receipt(arguments.root, arguments.check, arguments.archive), indent=2))
    else:
        print(json.dumps(check_claims(arguments.root, arguments.archive), indent=2))


def write_receipt(root, archive, operator):
    """Build and compare the recorded source, then publish a verified local receipt.

    :param root: Clean repository root at the commit to record.
    :param archive: Directory that receives the full reports outside the repository.
    :param operator: Identity recorded in the receipt.
    :return: The receipt; keeps archived evidence and replaces the receipt after verification.
    """
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.is_relative_to(root):
        raise ValueError("Archive the full reports outside the repository")
    _clean_tree(root)
    commit = _git(root, "rev-parse", "HEAD")
    key = controls._platform_key()
    archive.mkdir(parents=True, exist_ok=True)
    attempt = Path(mkdtemp(prefix=f"{key}-{commit[:12]}-", dir=archive))
    ports = attempt / "ports.json"
    rust = attempt / "rust.log"
    with TemporaryDirectory(prefix="bf-receipt-") as directory:
        snapshot = Path(directory)
        extension, build, toolchains, environment = _build_extension(root, commit, snapshot)
        binary = attempt / extension.name
        shutil.copyfile(extension, binary)
        bootstrap = (
            "import runpy, sys; from pathlib import Path; "
            "script, source, checkout, extension = sys.argv[1:5]; "
            "sys.path[:0] = [str(Path(script).parent), source, checkout]; "
            "from behave_fire import _rust; "
            "assert Path(_rust.__file__).resolve() == Path(extension).resolve(); "
            "sys.argv = [script, *sys.argv[5:]]; runpy.run_path(script, run_name='__main__')"
        )
        subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap, str(snapshot / "tools/cpp_calibrate.py"),
                        str(snapshot / "src"), str(root / "src"), str(extension), "--ports", "--root",
                        str(root), "--report", str(ports), "--platform-key", key], check=True, cwd=snapshot)
        log = subprocess.run(["cargo", "test", "--manifest-path", "rust/Cargo.toml", "-p", "behave-fire", "--locked",
                              "--target", build["target"], "--target-dir", "target", "--no-fail-fast", "--",
                              "--ignored", "--nocapture"], capture_output=True, text=True, cwd=snapshot,
                             env=environment)
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
        source_commit=commit, source_tree=build["source_tree"],
        source_files=calibrate._sources(root), fixture_inventory_sha256=reference._digest(base / "hashes.json"),
        artifacts=controls._artifacts(root),
        datasets={name: module.artifacts(root) for name, (module, _) in EXACT.items()},
        upstream_sha=reference.PIN, overlay_sha256=reference._prepared_build(root)["prepared_overlay_sha256"],
        public_api=calibrate._expected_public_api(root),
        binaries=dict(extension_sha256=reference._digest(binary), binding_api_version=report["binding_api_version"],
                      file=binary.relative_to(archive).as_posix(), build=build),
        toolchains=toolchains, host=_host(),
        reports={kind: dict(file=file.relative_to(archive).as_posix(), sha256=reference._digest(file))
                 for kind, file in (("ports", ports), ("rust", rust))},
        results=json.loads(json.dumps(results)))
    target = root / RECEIPTS / f"{key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as directory:
        staged = Path(directory) / target.name
        staged.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        check_receipt(root, staged, archive)
        staged.replace(target)
    return receipt


if __name__ == "__main__":
    main()
