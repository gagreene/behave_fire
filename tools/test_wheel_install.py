"""Run project selection and dependency-install checks in fresh external venvs."""

import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile


def _run(command, cwd, log=None):
    """Run a checked child command while preserving evidence in the CI log.

    :param command: Argument vector without shell interpolation.
    :param cwd: External working directory.
    :param log: Optional file preserving the child output, including failures.
    :return: None; raises on the first failed installation or smoke check.
    """
    print("+", " ".join(map(str, command)), flush=True)
    environment = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    result = subprocess.run(list(map(str, command)), cwd=cwd, env=environment, stdout=subprocess.PIPE if log else None, stderr=subprocess.STDOUT if log else None, text=True)
    if log:
        log.write_text(result.stdout, encoding="utf-8")
        print(result.stdout, flush=True)
    result.check_returncode()


def _version(wheel):
    """Read the exact project version from a wheel.

    :param wheel: Project wheel archive.
    :return: Core metadata Version field.
    """
    with zipfile.ZipFile(wheel) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        return BytesParser().parsebytes(archive.read(name))["Version"]


def main():
    """Check current and minimum binary dependencies for native and pure artifacts.

    :return: None; creates disposable external environments and prints evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--expect-choice", choices=("python", "rust"), required=True)
    parser.add_argument("--identity", choices=("windows-arm64", "musl-x86_64"))
    parser.add_argument("--report-dir", type=Path, required=True)
    arguments = parser.parse_args()
    wheels = list(arguments.wheels.resolve().glob("*.whl"))
    assert wheels, "no project wheels downloaded"
    versions = {_version(wheel) for wheel in wheels}
    assert len(versions) == 1, versions
    version = versions.pop()
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    floor = "2.3.3" if sys.version_info >= (3, 14) or (platform.system() == "Windows" and platform.machine() == "ARM64") else "2.1.3"
    smoke = Path(__file__).with_name("smoke_test_wheel.py")
    for policy in ("floor", "current"):
        for route in ("choice", "pure"):
            expected = arguments.expect_choice if route == "choice" else "python"
            with tempfile.TemporaryDirectory(prefix="behave-wheel-") as temporary:
                root = Path(temporary)
                staging = root / "wheels"
                staging.mkdir()
                for wheel in wheels:
                    if route == "choice" or wheel.name.endswith("-py3-none-any.whl"):
                        shutil.copy2(wheel, staging / wheel.name)
                assert list(staging.glob("*.whl")), "pure wheel missing"
                shutil.copy2(smoke, root / smoke.name)
                for phase in ("selection", "complete"):
                    environment = root / phase
                    venv.EnvBuilder(with_pip=True).create(environment)
                    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                    report = arguments.report_dir.resolve() / f"{policy}-{route}-{phase}.json"
                    if phase == "selection":
                        _run([python, "-m", "pip", "install", "--only-binary=:all:", "numpy==" + floor if policy == "floor" else "numpy>=" + floor], root)
                        _run([python, "-m", "pip", "install", "--no-deps", "--no-index", "--find-links", staging, "--report", report, "behave-fire==" + version], root)
                        data = json.loads(report.read_text())
                        selected = next(item for item in data["install"] if item["metadata"]["name"].replace("_", "-") == "behave-fire")
                        url = selected["download_info"]["url"]
                        assert url.endswith("-py3-none-any.whl") == (expected == "python"), url
                        if expected == "rust":
                            assert "-cp311-abi3-" in url, url
                        selected_wheel = next(wheel for wheel in staging.glob("*.whl") if wheel.name == Path(__import__("urllib.parse", fromlist=["unquote"]).unquote(url)).name)
                    else:
                        command = [python, "-m", "pip", "install", "--only-binary=:all:", "--report", report, selected_wheel]
                        if policy == "floor":
                            command.append("numpy==" + floor)
                        _run(command, root)
                    command = [python, "-I", root / smoke.name, "--expect", expected, "--version", version]
                    if arguments.identity:
                        command += ["--identity", arguments.identity]
                    _run(command, root, arguments.report_dir.resolve() / f"{policy}-{route}-{phase}-smoke.json")
                    _run([python, "-m", "pip", "list", "--format=json"], root)


if __name__ == "__main__":
    main()
