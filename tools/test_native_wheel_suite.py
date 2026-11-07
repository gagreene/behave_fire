"""Run the checkout's complete regression suite against a compatible built wheel."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tempfile
import venv
import zipfile


def _packaging_module(name):
    """Load declared packaging tools or the bare interpreter's bundled pip copy.

    :param name: Required packaging submodule.
    :return: Loaded module without installing or changing host dependencies.
    """
    prefix = "packaging" if importlib.util.find_spec("packaging") is not None else "pip._vendor.packaging"
    return importlib.import_module(prefix + "." + name)


def _run(command, root, log, environment):
    """Run a checked child command while retaining its full diagnostic output.

    :param command: Argument vector without a shell.
    :param root: Checkout working directory.
    :param log: Destination for combined output.
    :param environment: Child environment with the source import path.
    :return: None; writes the log and raises for failures or a 30-minute timeout.
    """
    print("+", " ".join(map(str, command)), flush=True)
    try:
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(list(map(str, command)), cwd=root, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        result.check_returncode()
    finally:
        if log.is_file():
            print(log.read_text(encoding="utf-8", errors="replace")[-8000:], flush=True)


def _select_native_wheel(wheels, supported):
    """Choose exactly one compatible project native wheel, never the pure fallback.

    :param wheels: Candidate wheel paths.
    :param supported: Stringified tags accepted by the host interpreter.
    :return: Native wheel path; raises ValueError for missing or ambiguous choices.
    """
    parse_wheel_filename = _packaging_module("utils").parse_wheel_filename
    matches = []
    for wheel in wheels:
        name, _, _, tags = parse_wheel_filename(wheel.name)
        if name == "behave-fire" and any(tag.platform != "any" and str(tag) in supported for tag in tags):
            matches.append(wheel)
    if len(matches) != 1:
        raise ValueError(f"require exactly one compatible native wheel, found {matches}")
    return matches[0]


@contextmanager
def _stage_extension(wheel, package):
    """Expose the shipped binary and bundled libraries without replacing a build.

    :param wheel: Selected built project wheel.
    :param package: Source package directory in a clean checkout.
    :return: Context yielding the binary path and removing all staged files on exit.
    """
    existing = [path for path in package.glob("_rust*") if path.suffix in (".pyd", ".so")]
    if existing:
        raise FileExistsError(f"use a clean checkout; native binaries already exist: {existing}")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist()
                 if re.fullmatch(r"behave_fire/_rust(?:\.[^/]+)?\.(?:pyd|so)", name)]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one project native extension")
        extension = package / PurePosixPath(names[0]).name
        payloads = [(extension, archive.read(names[0]))]
        libraries = package.parent / "behave_fire.libs"
        for item in archive.infolist():
            if item.filename.startswith("behave_fire.libs/") and not item.is_dir():
                name = item.filename.removeprefix("behave_fire.libs/")
                if not re.fullmatch(r"[^/\\:]+", name) or name in (".", ".."):
                    raise ValueError(f"unsafe companion library path: {item.filename}")
                payloads.append((libraries / name, archive.read(item)))
    created = []
    library_directory_created = False
    try:
        if len(payloads) > 1:
            libraries.mkdir()
            library_directory_created = True
        for destination, payload in payloads:
            with destination.open("xb") as output:
                created.append(destination)
                output.write(payload)
        yield extension
    finally:
        for destination in reversed(created):
            destination.unlink()
        if library_directory_created:
            libraries.rmdir()


def main():
    """Install the binary wheel and run all tests plus platform oracle diagnostics.

    :return: None; creates isolated dependencies, reports evidence and runs pytest.
    """
    sys_tags = _packaging_module("tags").sys_tags
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = arguments.report_dir.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    wheel = _select_native_wheel(arguments.wheels.resolve().glob("*.whl"), {str(tag) for tag in sys_tags()})
    floor = "2.3.3" if sys.version_info >= (3, 14) or (platform.system() == "Windows" and platform.machine() == "ARM64") else "2.1.3"
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    with tempfile.TemporaryDirectory(prefix="behave-full-suite-") as temporary:
        target = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run([python, "-m", "pip", "install", "--only-binary=:all:", str(wheel) + "[test,dev]", "numpy==" + floor],
             root, reports / "install.log", environment)
        with _stage_extension(wheel, root / "src/behave_fire") as extension:
            provenance = (
                "import hashlib,json,platform,sys; from pathlib import Path; import numpy; "
                "from behave_fire import _rust; from behave_fire.rust import _bridge; "
                "assert Path(_rust.__file__).resolve()==Path(sys.argv[1]).resolve(); "
                "print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,machine=platform.machine(),"
                "platform=platform.platform(),extension_sha256=hashlib.sha256(Path(_rust.__file__).read_bytes()).hexdigest(),"
                "svml_candidate_available=_rust.atan2_profile_candidates()[3] is not None,"
                "calibrated_atan2_profile=_bridge._atan2_math_profile())))"
            )
            _run([python, "-B", "-c", provenance, extension], root, reports / "environment.json", environment)
            record = json.loads((reports / "environment.json").read_text(encoding="utf-8"))
            record.update(wheel=wheel.name, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
            record["companion_libraries"] = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((root / "src/behave_fire.libs").glob("*")) if path.is_file()
            }
            (reports / "environment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            _run([python, "-B", "tools/probe_platform_oracle.py", "--native"], root,
                 reports / "numpy-platform-oracle.json", environment)
            _run([python, "-B", "tools/probe_direction_oracle.py"], root,
                 reports / "numpy-direction-oracle.json", environment)
            _run([python, "-B", "-m", "pytest", "--require-rust", "-W", "error", "--tb=short",
                  "-p", "no:cacheprovider", "--junitxml", reports / "pytest.xml"],
                 root, reports / "pytest.log", environment)


if __name__ == "__main__":
    main()
