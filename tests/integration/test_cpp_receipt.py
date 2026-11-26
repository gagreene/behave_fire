"""Check local C++ receipts: location outside the hashed fixtures, claims, and archived-report verification."""

import json
import math
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import cpp_calibrate as calibrate
import cpp_controls as controls
import cpp_receipt as receipts
import cpp_reference as reference

ROOT = Path(__file__).resolve().parents[2]
CLAIMS = {"schema_version": 1, "approved": False, "platforms": ["windows-x86_64"], "archive_location": None}


def _cargo_log(**changes):
    names = dict(receipts.RUST_TESTS, **changes.get("tests", {}))
    lines = [f"test {name} ... ok" for name, count in names.items() for _ in range(count)]
    lines += [f"{key} {value}" for key, value in dict(receipts.RUST_ENFORCED, **changes.get("enforced", {})).items()
              if value is not None]
    failed = changes.get("failed", 0)
    lines.append(f"test result: {'FAILED' if failed else 'ok'}. {sum(names.values())} passed; {failed} failed; "
                 "0 ignored; 0 measured; 0 filtered out; finished in 0.10s")
    return "\n".join(lines) + "\n"


def _receipt(archive, ports, log_text):
    key = controls._platform_key()
    attempt = archive / "attempt"
    attempt.mkdir()
    ports_copy = attempt / "ports.json"
    ports_copy.write_bytes(ports.read_bytes())
    log = attempt / "rust.log"
    log.write_text(log_text, encoding="utf-8")
    report = json.loads(ports_copy.read_text(encoding="utf-8"))
    results = dict(ports=receipts._validate_ports_report(ROOT, report, key),
                   rust=receipts._validate_cargo_log(log_text))
    base = ROOT / "tests/data/cpp_reference"
    target = {
        "windows": "pc-windows-msvc", "macos": "apple-darwin", "linux": "unknown-linux-gnu",
    }[key.split("-")[0]]
    if "musl" in key:
        target = "unknown-linux-musl"
    arch = "aarch64" if key.endswith(("arm64", "aarch64")) else "x86_64"
    target = f"{arch}-{target}"
    binary_name = "_rust.pyd" if key.startswith("windows-") else "_rust.so"
    extension = attempt / binary_name
    extension.write_bytes(b"receipt test extension")
    report["extension_sha256"] = reference._digest(extension)
    ports_copy.write_text(json.dumps(report), encoding="utf-8")
    return dict(
        schema_version=receipts.SCHEMA, evidence_origin="local", platform_key=key, created="2026-11-26T00:00:00+00:00",
        operator="127547449+gagreene@users.noreply.github.com", source_commit=report["source_commit"],
        source_tree="0" * 40, source_files=calibrate._sources(ROOT),
        fixture_inventory_sha256=reference._digest(base / "hashes.json"), artifacts=controls._artifacts(ROOT),
        datasets={name: module.artifacts(ROOT) for name, (module, _) in receipts.EXACT.items()},
        upstream_sha=reference.PIN, overlay_sha256=reference._prepared_build(ROOT)["prepared_overlay_sha256"],
        public_api=calibrate._expected_public_api(ROOT),
        binaries=dict(extension_sha256=report["extension_sha256"], binding_api_version=report["binding_api_version"],
                      file=f"attempt/{binary_name}",
                      build=dict(source_tree="0" * 40, target=target, command=receipts._build_command(target))),
        toolchains=dict(python=report["python"], numpy=report["numpy"], rustc=f"rustc 1.78.0\nhost: {target}",
                        cargo="cargo 1.78.0", cargo_encoded_rustflags="-Cdebuginfo=0"),
        host=dict(system=report["system"], machine=report["machine"], release="release", version="version",
                  processor="", cpu_features=dict.fromkeys(("FMA3", "AVX2", "AVX512F", "ASIMD"), False)),
        reports=dict(ports=dict(file="attempt/ports.json", sha256=reference._digest(ports_copy)),
                     rust=dict(file="attempt/rust.log", sha256=reference._digest(log))),
        results=json.loads(json.dumps(results)))


def _write_claims(root, **changes):
    path = root / receipts.CLAIMS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(CLAIMS, **changes)), encoding="utf-8")


@pytest.fixture(scope="module")
def ports_report(tmp_path_factory):
    path = tmp_path_factory.mktemp("ports") / "windows-ports.json"
    result = controls.measure_ports(ROOT, path, backends=receipts.BACKENDS)
    result.update(calibrate._context(ROOT))
    reference._json(path, result)
    return path


# ---- archived report verification tests ----

@pytest.mark.rust
@pytest.mark.cpp_comparison
def test_archived_ports_report_reapplies_every_numerical_rule(ports_report):
    report = json.loads(ports_report.read_text(encoding="utf-8"))
    summary = receipts._validate_ports_report(ROOT, report, controls._platform_key())
    assert summary == dict(
        surface=dict(cases_per_backend=710, fields_per_backend=3550, paired_cases_per_backend=139),
        exact_binary64_per_backend={"fire_size": 9, "ignite": 8, "mortality": 6},
        measurement_only=list(receipts.MEASURED))


@pytest.mark.parametrize("tamper,error", [
    ("surface", "fireline_intensity_btu_ft_s"),
    ("exact", "area_ft2"),
    ("backend", "missing backend observations"),
    ("platform", "identity does not match"),
    ("measured", "missing measurement-only observations"),
    ("measured-fields", "Crown output"),
    ("measured-short", "incomplete case identities"),
    ("measured-dataset", "different dataset"),
    ("origin", "identity does not match"),
], ids=["surface", "exact", "backend", "platform", "measured", "measured-fields", "measured-short",
        "measured-dataset", "origin"])
@pytest.mark.rust
@pytest.mark.cpp_comparison
def test_archived_ports_report_rejects_tampering(ports_report, tamper, error):
    report = json.loads(ports_report.read_text(encoding="utf-8"))
    key = controls._platform_key()
    if tamper == "surface":
        row = report["observations"][0]
        row["actual"]["fireline_intensity_btu_ft_s"] *= 1 + 1e-9
    elif tamper == "exact":
        row = report["fire_size"]["observations"]["rust"][0]["output"]
        row["area_ft2"] = math.nextafter(row["area_ft2"], 0.0)
    elif tamper == "backend":
        del report["ignite"]["observations"]["python"]
    elif tamper == "platform":
        key = "linux-x86_64" if key != "linux-x86_64" else "windows-x86_64"
    elif tamper == "measured":
        report["crown"]["observations"]["rust"] = []
    elif tamper == "measured-fields":
        report["crown"]["observations"]["rust"][0]["output"].clear()
    elif tamper == "measured-short":
        report["crown"]["observations"]["rust"].pop()
    elif tamper == "measured-dataset":
        report["crown"]["artifacts"].clear()
    else:
        report["evidence_origin"] = "github_actions"
    with pytest.raises((AssertionError, ValueError), match=error):
        receipts._validate_ports_report(ROOT, report, key)


@pytest.mark.parametrize("tamper,error", [
    (None, None),
    ("report-bytes", "differs from the receipt"),
    ("missing-report", "is missing"),
    ("fixture-digest", "not bound"),
    ("results", "results differ"),
    ("extension", "receipt's commit and extension"),
    ("build-provenance", "build provenance"),
    ("toolchains", "toolchain"),
    ("archived-extension", "Archived native extension"),
    ("platform-name", "file name"),
], ids=["valid", "report-bytes", "missing-report", "fixture-digest", "results", "extension",
        "build-provenance", "toolchains", "archived-extension", "platform-name"])
@pytest.mark.rust
@pytest.mark.cpp_comparison
def test_receipt_verification_reads_the_archived_reports(tmp_path, monkeypatch, ports_report, tamper, error):
    archive = tmp_path / "archive"
    archive.mkdir()
    receipt = _receipt(archive, ports_report, _cargo_log())
    monkeypatch.setattr(receipts, "_bound_revision", lambda *_: None)
    name = receipt["platform_key"]
    if tamper == "report-bytes":
        log = archive / receipt["reports"]["rust"]["file"]
        log.write_text(log.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif tamper == "missing-report":
        (archive / receipt["reports"]["ports"]["file"]).unlink()
    elif tamper == "fixture-digest":
        receipt["fixture_inventory_sha256"] = "0" * 64
    elif tamper == "results":
        receipt["results"]["rust"]["tests"] += 1
    elif tamper == "extension":
        receipt["binaries"]["extension_sha256"] = "0" * 64
    elif tamper == "build-provenance":
        receipt["binaries"]["build"] = {"source_tree": "1" * 40}
    elif tamper == "toolchains":
        receipt["toolchains"] = {}
    elif tamper == "archived-extension":
        (archive / receipt["binaries"]["file"]).write_bytes(b"stale extension")
    elif tamper == "platform-name":
        name = "linux-aarch64" if name != "linux-aarch64" else "windows-x86_64"
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    if error is None:
        assert receipts.check_receipt(ROOT, path, archive) == receipt["results"]
    else:
        with pytest.raises(ValueError, match=error):
            receipts.check_receipt(ROOT, path, archive)


# ---- receipt location and claims tests ----

def test_check_claims_rejects_missing_platform_evidence(tmp_path):
    _write_claims(tmp_path, platforms=["windows-x86_64", "linux-x86_64"], approved=True)
    with pytest.raises(ValueError, match=r"Missing platform evidence: \['windows-x86_64', 'linux-x86_64'\]"):
        receipts.check_claims(tmp_path, tmp_path / "archive")


def test_check_claims_requires_owner_approval_after_valid_evidence(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    _write_claims(root)
    (root / receipts.RECEIPTS / "windows-x86_64.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(receipts, "check_receipt", lambda root, path, archive: {"platform": path.stem})
    with pytest.raises(ValueError, match="not approved"):
        receipts.check_claims(root, tmp_path)
    _write_claims(root, approved=True)
    with pytest.raises(ValueError, match="archive location"):
        receipts.check_claims(root, tmp_path)
    archive = tmp_path / "approved-archive"
    archive.mkdir()
    _write_claims(root, approved=True, archive_location=archive.as_posix())
    with pytest.raises(ValueError, match="approved archive"):
        receipts.check_claims(root, tmp_path)
    assert receipts.check_claims(root, archive) == {"windows-x86_64": {"platform": "windows-x86_64"}}


@pytest.mark.parametrize("claims", [
    dict(CLAIMS, platforms=[]),
    dict(CLAIMS, platforms=["windows-x86_64", "windows-x86_64"]),
    dict(CLAIMS, platforms=["solaris-sparc"]),
    dict(CLAIMS, extra=True),
    dict(CLAIMS, schema_version=2),
])
def test_check_claims_rejects_malformed_claims(tmp_path, claims):
    path = tmp_path / receipts.CLAIMS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(claims), encoding="utf-8")
    with pytest.raises(ValueError, match="claims"):
        receipts.check_claims(tmp_path, tmp_path)


def test_committed_claims_stay_unapproved_and_unpublishable(tmp_path):
    assert json.loads((ROOT / receipts.CLAIMS).read_text(encoding="utf-8")) == CLAIMS
    with pytest.raises(ValueError, match="Missing platform evidence|not approved"):
        receipts.check_claims(ROOT, tmp_path)


def test_receipt_tree_must_match_its_commit(tmp_path):
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    with pytest.raises(ValueError, match="tree does not match"):
        receipts._bound_revision(ROOT, {"source_commit": head, "source_tree": "0" * 40})
    with pytest.raises(ValueError, match="full SHA-1"):
        receipts._bound_revision(ROOT, {"source_commit": head[:12], "source_tree": "0" * 40})


def test_receipts_stay_outside_the_hashed_fixture_inventory():
    assert not (ROOT / receipts.RECEIPTS).resolve().is_relative_to((ROOT / "tests/data/cpp_reference").resolve())
    listed = reference._required_artifacts(ROOT)
    hashed = json.loads((ROOT / "tests/data/cpp_reference/hashes.json").read_text(encoding="utf-8"))
    for names in (listed, hashed):
        assert not any(name.startswith(receipts.RECEIPTS.as_posix() + "/") for name in names)


def test_rust_log_requires_every_comparison_and_enforced_count():
    assert receipts._validate_cargo_log(_cargo_log()) == dict(tests=10, enforced=receipts.RUST_ENFORCED)
    bad = [
        _cargo_log(failed=1),
        _cargo_log(tests={"perturbed_cpp_copy_fails_the_surface_caps": 0}),
        _cargo_log(tests={"perturbed_cpp_copy_fails_the_exact_regression": 2}),
        _cargo_log(enforced={"CPP_SURFACE_ENFORCED": "compared=2854 documented=60"}),
        _cargo_log(enforced={"CPP_IGNITE_ENFORCED": None}),
        "",
    ]
    for text in bad:
        with pytest.raises(ValueError):
            receipts._validate_cargo_log(text)


# ---- receipt writes tests ----

def test_failed_rust_retry_preserves_prior_reports(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    archive = tmp_path / "archive"
    root.mkdir()
    archive.mkdir()
    previous = root / receipts.RECEIPTS / "windows-x86_64.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous receipt", encoding="utf-8")
    commit = "9" * 40
    old_log = archive / f"windows-x86_64-{commit}-rust.log"
    old_log.write_text("previous successful evidence", encoding="utf-8")
    monkeypatch.setattr(receipts, "_clean_tree", lambda *_: None)
    monkeypatch.setattr(receipts, "_git", lambda *_: commit)
    monkeypatch.setattr(controls, "_platform_key", lambda: "windows-x86_64")
    extension = tmp_path / "_rust.pyd"
    extension.write_bytes(b"test extension")
    build = {"target": "x86_64-pc-windows-msvc"}
    monkeypatch.setattr(receipts, "_build_extension", lambda *_: (extension, build, {}, {}))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs:
        subprocess.CompletedProcess([], 1, "", "test failure"))
    with pytest.raises(ValueError, match="Rust comparison run failed"):
        receipts.write_receipt(root, archive, "operator")
    assert old_log.read_text(encoding="utf-8") == "previous successful evidence"
    assert previous.read_text(encoding="utf-8") == "previous receipt"


# ---- release evidence tests ----

@pytest.mark.parametrize("failure,error", [
    (None, None), ("approved", "not approved"), ("archive", "approved HTTPS archive"),
    ("missing", "Missing platform evidence"),
    ("digest", "archive SHA-256"), ("path", "Unsafe evidence archive member"), ("receipt", "bad receipt"),
])
def test_release_claims_fail_closed(tmp_path, monkeypatch, failure, error):
    fixture = BytesIO()
    with ZipFile(fixture, "w") as bundle:
        bundle.writestr("../outside.log" if failure == "path" else "attempt/rust.log", "evidence")
    payload = fixture.getvalue()
    archive = tmp_path / "evidence.zip"
    archive.write_bytes(payload)
    location = {"url": "https://example.test/evidence.zip", "sha256": reference._digest(archive)}
    if failure == "archive":
        location["url"] = "http://example.test/evidence.zip"
    if failure == "digest":
        location["sha256"] = "0" * 64
    _write_claims(tmp_path, approved=failure != "approved", archive_location=location)
    receipt = tmp_path / receipts.RECEIPTS / "windows-x86_64.json"
    if failure != "missing":
        receipt.write_text("{}", encoding="utf-8")

    def verify_receipt(root, path, archive):
        assert (archive / "attempt/rust.log").read_text(encoding="utf-8") == "evidence"
        if failure == "receipt":
            raise ValueError("bad receipt")
        return {"validated": True}

    if failure not in {"approved", "archive", "missing"}:
        monkeypatch.setattr(receipts, "urlopen", lambda *_args, **_kwargs: archive.open("rb"), raising=False)
    monkeypatch.setattr(receipts, "check_receipt", verify_receipt)
    if failure is None:
        assert receipts.check_release_claims(tmp_path) == {"windows-x86_64": {"validated": True}}
    else:
        with pytest.raises(ValueError, match=error):
            receipts.check_release_claims(tmp_path)
