"""Full-suite wheel binaries must match the host and never overwrite local builds."""

import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def _runner():
    path = Path(__file__).parents[1] / "tools/test_native_wheel_suite.py"
    assert path.is_file(), "full native-wheel suite runner is missing"
    spec = importlib.util.spec_from_file_location("native_wheel_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(root, tag, members=None):
    path = root / ("behave_fire-1.0.0b2-" + tag + ".whl")
    with zipfile.ZipFile(path, "w") as archive:
        for name in members or ("behave_fire/_rust.abi3.so",):
            archive.writestr(name, b"binary fixture")
    return path


@pytest.mark.rust
def test_actual_wheel_binary_provenance_needs_no_retired_math_profiles():
    from behave_fire import _rust

    runner = _runner()
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run([sys.executable, '-B', '-c', runner._environment_command(), _rust.__file__],
                             cwd=root, env=dict(os.environ, PYTHONPATH=str(root/'src')),
                             capture_output=True, text=True, check=True)
    evidence = json.loads(process.stdout)
    assert evidence['binding_api_version'] == 4
    assert evidence['native_physical_dtype'] == 'float64'
    assert len(evidence['extension_sha256']) == 64


@pytest.mark.parametrize("members", [("../_rust.abi3.so",), ("behave_fire/_rust.pyd", "behave_fire/_rust.abi3.so")])
def test_ambiguous_or_unsafe_extension_payload_is_rejected(tmp_path, members):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-win_arm64", members)
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        with runner._stage_extension(wheel, package):
            pytest.fail("invalid extension was staged")
    assert not list(package.iterdir())


def test_companion_library_staging_and_cleanup_preserve_loader_layout(tmp_path):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-musllinux_1_2_x86_64",
                   ("behave_fire/_rust.abi3.so", "behave_fire.libs/libgcc_s-hash.so.1"))
    package = tmp_path / "package"
    package.mkdir()
    libraries = package.parent / "behave_fire.libs"
    with runner._stage_extension(wheel, package):
        assert (libraries / "libgcc_s-hash.so.1").read_bytes() == b"binary fixture"
    assert not libraries.exists()
    assert not list(package.iterdir())


def test_companion_library_staging_refuses_existing_directory(tmp_path):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-musllinux_1_2_x86_64",
                   ("behave_fire/_rust.abi3.so", "behave_fire.libs/libgcc_s-hash.so.1"))
    package = tmp_path / "package"
    package.mkdir()
    libraries = package.parent / "behave_fire.libs"
    libraries.mkdir()
    existing = libraries / "local.so"
    existing.write_bytes(b"local library")
    with pytest.raises(FileExistsError):
        with runner._stage_extension(wheel, package):
            pytest.fail("local libraries were replaced")
    assert existing.read_bytes() == b"local library"
    assert not list(package.iterdir())


@pytest.mark.parametrize("member", ["behave_fire.libs/../escape.so", "behave_fire.libs/..\\escape.so"])
def test_companion_library_staging_rejects_path_escape(tmp_path, member):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-musllinux_1_2_x86_64", ("behave_fire/_rust.abi3.so", member))
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        with runner._stage_extension(wheel, package):
            pytest.fail("unsafe library was staged")
    assert not (tmp_path / "escape.so").exists()
    assert not list(package.iterdir())


@pytest.mark.parametrize("suffix", ["pyd", "abi3.so"])
def test_extension_staging_cleans_up_even_after_suite_failure(tmp_path, suffix):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-win_arm64", ("behave_fire/_rust." + suffix,))
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(RuntimeError, match="suite failed"):
        with runner._stage_extension(wheel, package) as extension:
            assert extension.read_bytes() == b"binary fixture"
            raise RuntimeError("suite failed")
    assert not list(package.iterdir())


@pytest.mark.parametrize("name", ["_rust.abi3.so", "_rust.pyd"])
def test_extension_staging_preserves_existing_local_binary(tmp_path, name):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-win_arm64")
    package = tmp_path / "package"
    package.mkdir()
    existing = package / name
    existing.write_bytes(b"local build")
    with pytest.raises(FileExistsError):
        with runner._stage_extension(wheel, package):
            pytest.fail("local binary was overwritten")
    assert existing.read_bytes() == b"local build"


def test_native_selection_excludes_pure_and_other_platform_wheels(tmp_path):
    runner = _runner()
    expected = _wheel(tmp_path, "cp311-abi3-win_arm64")
    _wheel(tmp_path, "cp311-abi3-win_amd64")
    _wheel(tmp_path, "py3-none-any")
    assert runner._select_native_wheel(tmp_path.glob("*.whl"), {"cp311-abi3-win_arm64"}) == expected


def test_native_selection_rejects_missing_or_ambiguous_binaries(tmp_path):
    runner = _runner()
    _wheel(tmp_path, "py3-none-any")
    supported = {"cp311-abi3-win_arm64"}
    with pytest.raises(ValueError, match="exactly one"):
        runner._select_native_wheel(tmp_path.glob("*.whl"), supported)
    _wheel(tmp_path, "cp311-abi3-win_arm64").rename(tmp_path / "behave_fire-1.0.0b2-1-cp311-abi3-win_arm64.whl")
    _wheel(tmp_path, "cp311-abi3-win_arm64")
    with pytest.raises(ValueError, match="exactly one"):
        runner._select_native_wheel(tmp_path.glob("*.whl"), supported)


def test_native_selection_works_without_seeded_pip(tmp_path, monkeypatch):
    runner = _runner()
    wheel = _wheel(tmp_path, "cp311-abi3-win_arm64")
    original = builtins.__import__

    def without_pip(name, *args, **kwargs):
        if name == "pip" or name.startswith("pip."):
            raise ModuleNotFoundError("No module named 'pip'", name="pip")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pip)
    assert runner._select_native_wheel([wheel], {"cp311-abi3-win_arm64"}) == wheel
