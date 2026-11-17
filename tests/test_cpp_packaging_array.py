"""Keep optional C++ reference sources and their notices outside distributions."""

import hashlib
import importlib.util
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference


ROOT = Path(__file__).parents[1]
FORBIDDEN = (
    "reference/behave_cpp/src/behave/surface.cpp",
    "reference/behave_cpp/LICENSE",
    "reference/behave_cpp_overlay/surface_driver.cpp",
    "behave_fire/reference/behave_cpp/NOTICE",
)


def _crate(tmp_path, extra):
    checker = _load("check_crate_package")
    files = dict.fromkeys(checker.REQUIRED_FILES, b"fixture")
    files["Cargo.toml"] = (
        b'[package]\nname="behave-fire"\nversion="0.1.0"\nlicense="MIT AND BSD-3-Clause"\n'
    )
    files.update(extra)
    return _tar(tmp_path / "behave-fire-0.1.0.crate", "behave-fire-0.1.0", files)


def _load(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tar(path, prefix, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            entry = tarfile.TarInfo(prefix + "/" + name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return path


def test_cpp_reference_exclusions_apply_to_all_maturin_formats():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = configuration["tool"]["maturin"]["exclude"]
    assert "reference/behave_cpp" in exclusions
    assert "reference/behave_cpp/**" in exclusions
    assert "reference/behave_cpp_overlay/**" in exclusions


def test_cpp_reference_exclusions_are_explicit_in_standalone_crate():
    configuration = tomllib.loads((ROOT / "rust/crates/behave-fire/Cargo.toml").read_text(encoding="utf-8"))
    exclusions = configuration["package"]["exclude"]
    assert "reference/behave_cpp/**" in exclusions
    assert "reference/behave_cpp_overlay/**" in exclusions


def test_crate_cpp_fixture_copies_match_canonical_bytes():
    exporter = _load("export_cpp_crate")
    exporter.export(
        ROOT / "tests/data/cpp_reference",
        ROOT / "rust/crates/behave-fire/tests/data/cpp_reference/surface",
        check=True,
    )


@pytest.mark.parametrize("name", FORBIDDEN)
def test_crate_rejects_cpp_reference_sources_and_notices(tmp_path, name):
    checker = _load("check_crate_package")
    with pytest.raises(ValueError, match="C\\+\\+ reference"):
        checker.inspect_crate(_crate(tmp_path, {name: b"synthetic forbidden payload"}))


def test_export_copies_only_explicit_numeric_fixture_inputs(tmp_path):
    exporter = _load("export_cpp_crate")
    source = tmp_path / "canonical"
    (source / "surface").mkdir(parents=True)
    (source / "surface/cases.jsonl").write_text('{"id":"case"}\n', encoding="utf-8")
    (source / "surface/results.jsonl").write_text('{"id":"case","status":"unavailable"}\n', encoding="utf-8")
    (source / "tolerance-policy.json").write_text(json.dumps({"approved": False}), encoding="utf-8")
    (source / "surface/manifest.json").write_text(
        json.dumps({"upstream_sha": "f3b2943e69abca942e795c5b1b45276f6338abb5"}), encoding="utf-8",
    )
    (source / "LICENSE").write_text("synthetic excluded notice", encoding="utf-8")
    destination = tmp_path / "crate"
    exporter.export(source, destination)
    assert sorted(path.name for path in destination.iterdir()) == [
        "cases.jsonl", "manifest.json", "results.jsonl", "tolerance-policy.json",
    ]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_sha"] == "f3b2943e69abca942e795c5b1b45276f6338abb5"
    assert manifest["files"] == {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in ("cases.jsonl", "results.jsonl", "tolerance-policy.json")
    }
    for name in ("cases.jsonl", "results.jsonl"):
        assert (destination / name).read_bytes() == (source / "surface" / name).read_bytes()
    exporter.export(source, destination, check=True)
    (destination / "results.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        exporter.export(source, destination, check=True)


def test_export_fails_before_writing_when_any_canonical_input_is_missing(tmp_path):
    exporter = _load("export_cpp_crate")
    source = tmp_path / "canonical"
    (source / "surface").mkdir(parents=True)
    (source / "surface/cases.jsonl").write_text('{}\n', encoding="utf-8")
    destination = tmp_path / "crate"
    with pytest.raises(FileNotFoundError):
        exporter.export(source, destination)
    assert not destination.exists()


def test_numeric_cpp_fixture_is_allowed_in_crate(tmp_path):
    checker = _load("check_crate_package")
    report = checker.inspect_crate(_crate(tmp_path, {"tests/data/cpp_reference/surface.tsv": b"1\t2\n"}))
    assert report["normal_dependencies"] == 0


def test_sdist_explicitly_includes_untracked_cpp_candidate_inputs(tmp_path):
    checker = _load("check_sdist")
    files = dict.fromkeys(checker.REQUIRED, b"fixture")
    files["tools/cpp_reference.py"] = b"offline reader"
    archive = _tar(tmp_path / "missing-cpp-inputs.tar.gz", "behave_fire", files)
    with pytest.raises(ValueError, match=r"C\+\+ offline"):
        checker.check_archive(archive)
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = configuration["tool"]["maturin"]["include"]
    assert {"path": "tests/data/cpp_reference/**/*", "format": "sdist"} in includes
    assert {"path": "tests/test_cpp*_array.py", "format": "sdist"} in includes
    assert {"path": "tools/*.py", "format": "sdist"} in includes


@pytest.mark.parametrize("name", FORBIDDEN)
def test_sdist_rejects_cpp_reference_sources_and_notices(tmp_path, name):
    checker = _load("check_sdist")
    files = dict.fromkeys(checker.REQUIRED, b"fixture")
    files[name] = b"synthetic forbidden payload"
    archive = _tar(tmp_path / "behave_fire.tar.gz", "behave_fire", files)
    with pytest.raises(ValueError, match="C\\+\\+ reference"):
        checker.check_archive(archive)


@pytest.mark.parametrize("name", FORBIDDEN)
def test_wheel_rejects_cpp_reference_sources_and_notices(tmp_path, name):
    checker = _load("check_sdist")
    path = tmp_path / "behave_fire.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"synthetic forbidden payload")
    with pytest.raises(ValueError, match="C\\+\\+ reference"):
        checker.check_wheel(path)


def test_wheel_without_cpp_payload_passes_inventory_check(tmp_path):
    checker = _load("check_sdist")
    path = tmp_path / "behave_fire.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("behave_fire/__init__.py", b"")
    assert checker.check_wheel(path) == ["behave_fire/__init__.py"]
