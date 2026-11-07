"""Require native and pure distributions to expose identical project metadata."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import zipfile

from packaging.requirements import Requirement
from packaging.utils import parse_sdist_filename, parse_wheel_filename


def _platform_family(platform):
    """Normalize the supported native tags without counting glibc aliases twice.

    :param platform: One wheel platform tag.
    :return: Supported platform family, or None for an unsupported tag.
    """
    platforms = {
        "manylinux_2_17_x86_64": "linux-x86_64", "manylinux2014_x86_64": "linux-x86_64",
        "manylinux_2_17_aarch64": "linux-aarch64", "manylinux2014_aarch64": "linux-aarch64",
        "musllinux_1_2_x86_64": "linux-musl-x86_64", "win_amd64": "windows-x86_64", "win_arm64": "windows-arm64",
    }
    macos = re.fullmatch(r"macosx_\d+_\d+_(x86_64|arm64)", platform)
    return "macos-" + macos[1] if macos else platforms.get(platform)


def check_distribution_inventory(directory):
    """Require one native wheel per tested family, one pure wheel and one sdist.

    :param directory: Combined project distribution directory.
    :return: Sorted wheel paths; raises ValueError for incomplete or mistagged artifacts.
    """
    wheels = sorted(directory.glob("*.whl"))
    pure = [wheel for wheel in wheels if wheel.name.endswith("-py3-none-any.whl")]
    native = [wheel for wheel in wheels if wheel not in pure]
    if not native:
        raise ValueError("at least one native wheel is required to compare native/pure metadata")
    if len(pure) != 1:
        raise ValueError("exactly one universal pure wheel is required")
    sdists = list(directory.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise ValueError("exactly one source distribution is required")
    project = parse_sdist_filename(sdists[0].name)
    families = dict.fromkeys(("linux-x86_64", "linux-aarch64", "linux-musl-x86_64", "macos-x86_64",
                              "macos-arm64", "windows-x86_64", "windows-arm64"), 0)
    for wheel in wheels:
        name, version, _, tags = parse_wheel_filename(wheel.name)
        if (name, version) != project:
            raise ValueError(f"distribution project/version mismatch: {wheel.name}")
        if wheel in pure:
            continue
        if any(tag.interpreter != "cp311" or tag.abi != "abi3" for tag in tags):
            raise ValueError(f"native wheel must use cp311-abi3: {wheel.name}")
        platforms = {_platform_family(tag.platform) for tag in tags}
        if len(platforms) != 1 or None in platforms:
            raise ValueError(f"unsupported native wheel platform: {wheel.name}")
        families[platforms.pop()] += 1
    if any(count != 1 for count in families.values()):
        raise ValueError(f"require exactly one native wheel per tested platform: {families}")
    return wheels


def main():
    """Check the complete distribution set and compare native/pure metadata.

    :return: None; fails before artifact aggregation on metadata differences.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    wheels = check_distribution_inventory(arguments.directory)
    pure = next(wheel for wheel in wheels if wheel.name.endswith("-py3-none-any.whl"))
    expected = read_metadata(pure)
    for wheel in wheels:
        actual = read_metadata(wheel)
        if actual != expected:
            raise ValueError(f"metadata or license mismatch: {wheel.name}\n{actual[0]}\n!=\n{expected[0]}")
        print("metadata matched:", wheel.name)


def read_metadata(wheel):
    """Read normalized metadata plus actual bundled license bytes.

    :param wheel: Native or pure wheel path.
    :return: Canonical fields, description, and relative license payloads.
    """
    with zipfile.ZipFile(wheel) as archive:
        name = next(item for item in archive.namelist() if item.endswith(".dist-info/METADATA"))
        message = BytesParser().parsebytes(archive.read(name))
        fields = {}
        for key in message.keys():
            # Backend bookkeeping and transport formatting are not project metadata.
            if key in ("Metadata-Version", "Generator", "Dynamic"):
                continue
            values = message.get_all(key)
            if key == "Requires-Dist":
                values = [str(Requirement(value)) for value in values]
            fields[key] = sorted(values)
        prefix = name.rsplit("/", 1)[0] + "/licenses/"
        licenses = {item.removeprefix(prefix): archive.read(item) for item in archive.namelist() if item.startswith(prefix) and not item.endswith("/")}
        assert set(licenses) == {"LICENSE", "NOTICE", "LICENSES/NumPy-BSD-3-Clause.txt", "LICENSES/Intel-SVML-BSD-3-Clause.txt"}, licenses.keys()
        return fields, message.get_payload(decode=True).decode("utf-8").replace("\r\n", "\n").rstrip("\n"), licenses


if __name__ == "__main__":
    main()
