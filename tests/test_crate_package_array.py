"""Artifact checks reject incomplete or non-standalone crate archives."""

import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_crate_package.py"
SPEC = importlib.util.spec_from_file_location("check_crate_package", MODULE_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def _archive(tmp_path, changes=None):
    manifest = b'[package]\nname="behave-fire"\nversion="0.1.0"\nlicense="MIT AND BSD-3-Clause"\n'
    files = {name: b"fixture" for name in CHECKER.REQUIRED_FILES}
    files["Cargo.toml"] = manifest
    files["src/lib.rs"] = b'#![doc = include_str!("../README.md")]\n'
    for name, content in (changes or {}).items():
        if content is None:
            files.pop(name, None)
        else:
            files[name] = content
    path = tmp_path / "behave-fire-0.1.0.crate"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo("behave-fire-0.1.0/" + name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return path


def test_archive_budget_is_enforced(tmp_path, monkeypatch):
    path = _archive(tmp_path)
    monkeypatch.setattr(CHECKER, "MAX_COMPRESSED_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        CHECKER.inspect_crate(path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "src\\escape"])
def test_archive_paths_cannot_escape(tmp_path, name):
    path = _archive(tmp_path, {name: b"bad"})
    with pytest.raises(ValueError, match="path"):
        CHECKER.inspect_crate(path)


def test_archive_validates_manifest_identity(tmp_path):
    path = _archive(tmp_path, {"Cargo.toml": b'[package]\nname="other"\nversion="0.1.0"\n'})
    with pytest.raises(ValueError, match="identity"):
        CHECKER.inspect_crate(path)


def test_complete_archive_passes(tmp_path):
    report = CHECKER.inspect_crate(_archive(tmp_path))
    assert report["name"] == "behave-fire"
    assert report["version"] == "0.1.0"
    assert report["compressed_bytes"] < 5 * 1024 * 1024


@pytest.mark.parametrize("source", [b'include_str!("../../outside.json")', b'include!(concat!(env!("OUT_DIR"), "/generated.rs"))', b'include_str !("../../outside.json")', b'include_bytes \n !("../../outside.bin")'])
def test_external_or_generated_includes_fail(tmp_path, source):
    path = _archive(tmp_path, {"src/lib.rs": source})
    with pytest.raises(ValueError, match="include"):
        CHECKER.inspect_crate(path)


def test_missing_embedded_runtime_table_fails(tmp_path):
    path = _archive(tmp_path, {"src/generated/fuel_models.rs": None})
    with pytest.raises(ValueError, match="missing"):
        CHECKER.inspect_crate(path)


def test_missing_svml_license_fails(tmp_path):
    path = _archive(tmp_path, {"LICENSES/Intel-SVML-BSD-3-Clause.txt": None})
    with pytest.raises(ValueError, match="Intel-SVML-BSD-3-Clause"):
        CHECKER.inspect_crate(path)


@pytest.mark.parametrize("extra", [b'[dependencies]\npyo3="0.27"\n', b'[build-dependencies]\ncc="1"\n', b'[target."cfg(windows)".dependencies]\nlibc="0.2"\n'])
def test_runtime_and_build_dependencies_fail(tmp_path, extra):
    path = _archive(tmp_path, {"Cargo.toml": b'[package]\nname="behave-fire"\nversion="0.1.0"\nlicense="MIT AND BSD-3-Clause"\n' + extra})
    with pytest.raises(ValueError, match="dependenc"):
        CHECKER.inspect_crate(path)


def test_runtime_build_script_fails(tmp_path):
    path = _archive(tmp_path, {"build.rs": b"fn main() {}"})
    with pytest.raises(ValueError, match="build script"):
        CHECKER.inspect_crate(path)
