"""Require native and pure distributions to expose identical project metadata."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import zipfile

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename


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


def main():
    """Compare every supplied wheel to the universal fallback artifact.

    :return: None; fails before artifact aggregation on metadata differences.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    wheels = list(arguments.directory.glob("*.whl"))
    native = [wheel for wheel in wheels if any(tag.platform != "any" for tag in parse_wheel_filename(wheel.name)[3])]
    if not native:
        raise ValueError("at least one native wheel is required to compare native/pure metadata")
    pure = next((wheel for wheel in wheels if wheel.name.endswith("-py3-none-any.whl")), None)
    if pure is None:
        raise ValueError("a universal pure wheel is required to compare native/pure metadata")
    expected = read_metadata(pure)
    for wheel in wheels:
        actual = read_metadata(wheel)
        if actual != expected:
            raise ValueError(f"metadata or license mismatch: {wheel.name}\n{actual[0]}\n!=\n{expected[0]}")
        print("metadata matched:", wheel.name)


if __name__ == "__main__":
    main()
