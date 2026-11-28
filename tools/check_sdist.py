"""Inspect source distributions and require the actual tracked build inputs."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED = {
    "pyproject.toml", "README.md", "LICENSE", "NOTICE", "LICENSES/NumPy-BSD-3-Clause.txt",
    "LICENSES/Intel-SVML-BSD-3-Clause.txt",
    "src/behave_fire/__init__.py", "rust/Cargo.toml", "rust/Cargo.lock",
    "rust/crates/behave-fire/Cargo.toml", "rust/crates/behave-fire/src/lib.rs",
    "rust/crates/behave-py/Cargo.toml", "rust/crates/behave-py/src/lib.rs",
}
# Python/Rust acceptance fixtures that the shipped parity checker and tests read.
FIXTURES = "tests/data/rust_port"
# Local C++ comparison receipts and generated example data stay out of every distribution.
LOCAL_EVIDENCE = ("tests/data/cpp_receipts/", "src/examples/data/", "src/examples/results/")


def _check_cpp_inputs(archive, members):
    """Require the offline reader's entire declared payload in its sdist.

    :param archive: Open source archive.
    :param members: Normalized relative names mapped to regular members.
    :return: None; raises for missing, malformed, or changed reader inputs.
    """
    if "tools/cpp_reference.py" not in members:
        return
    name = "tests/data/cpp_reference/hashes.json"
    if name not in members:
        raise ValueError("C++ offline reader is missing its expected hash list")
    hashes = json.loads(archive.extractfile(members[name]).read(), object_pairs_hook=_unique_hashes)
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("C++ offline hash list is empty or malformed")
    for relative, expected in hashes.items():
        if relative not in members:
            raise ValueError(f"C++ offline input is missing: {relative}")
        actual = hashlib.sha256(archive.extractfile(members[relative]).read()).hexdigest()
        if actual != expected:
            raise ValueError(f"C++ offline input changed in the archive: {relative}")


def _check_local_evidence(name):
    """Reject local C++ comparison receipts and generated example data.

    Receipts carry operator, host, and build metadata; the example data and result rasters are local outputs.

    :param name: Distribution-relative POSIX member name.
    :return: None; raises ValueError when a local file is included.
    """
    if name.startswith(LOCAL_EVIDENCE):
        raise ValueError(f"local file in distribution: {name}")


def _check_reference_payload(name):
    """Reject optional C++ reference sources, driver sources, and their notices.

    :param name: Distribution-relative POSIX member name.
    :return: None; raises ValueError when a reference checkout is included.
    """
    if any(part in {"behave_cpp", "behave_cpp_overlay"} for part in PurePosixPath(name).parts):
        raise ValueError(f"C++ reference payload in distribution: {name}")


def _unique_hashes(pairs):
    """Reject duplicate JSON keys in archive integrity metadata.

    :param pairs: JSON object pairs.
    :return: Unique object fields, or raises ValueError.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("C++ offline hash list has duplicate keys")
        result[key] = value
    return result


def check_archive(path, expected=()):
    """Validate the archive inventory and reject unsafe or stale build inputs.

    :param path: Source tarball built by Maturin.
    :param expected: Complete source inventory captured before the build.
    :return: Sorted complete source inventory, after validating both crates.
    """
    with tarfile.open(path) as archive:
        names = set()
        members = {}
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or PurePosixPath(member.name).is_absolute() or ".." in parts or member.issym() or member.islnk(
                ):
                raise ValueError(f"unsafe archive member: {member.name}")
            name = "/".join(parts[1:])
            _check_reference_payload(name)
            _check_local_evidence(name)
            if member.isfile():
                if name in names:
                    raise ValueError(f"duplicate archive member: {name}")
                names.add(name)
                members[name] = member
            if any(item in parts for item in (".git", "target", "__pycache__", ".superpowers")) or name.endswith((
                ".pyd", ".so", ".pyc", ".pdb")):
                raise ValueError(f"build leftover in source distribution: {name}")
        _check_cpp_inputs(archive, members)
    missing = (REQUIRED | set(expected)) - names
    if missing:
        raise ValueError(f"missing source distribution files: {sorted(missing)}")
    for crate in ("behave-fire", "behave-py"):
        if not any(name.startswith(f"rust/crates/{crate}/src/") and name.endswith(".rs") for name in names):
            raise ValueError(f"missing Rust sources for {crate}")
    return sorted(names)


def check_committed_fixtures(path, root):
    """Require the archived acceptance fixtures to equal the committed ones byte for byte.

    :param path: Source tarball built by Maturin.
    :param root: Repository root whose ``HEAD`` commit defines the fixtures.
    :return: Sorted fixture names; raises ValueError for a missing, extra, or changed file.
    """
    listing = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD", "--", FIXTURES],
        check=True, capture_output=True, text=True).stdout.split()
    committed = {name for name in listing if name.endswith(".json")}
    if not committed:
        raise ValueError(f"no committed fixtures under {FIXTURES}")
    with tarfile.open(path) as archive:
        archived = {}
        for member in archive.getmembers():
            name = "/".join(PurePosixPath(member.name).parts[1:])
            if member.isfile() and name.startswith(FIXTURES + "/"):
                archived[name] = archive.extractfile(member).read()
    if set(archived) != committed:
        raise ValueError(f"fixture set differs: missing {sorted(committed - set(archived))}, "
                         f"extra {sorted(set(archived) - committed)}")
    for name in sorted(committed):
        blob = subprocess.run(["git", "-C", str(root), "cat-file", "blob", f"HEAD:{name}"], check=True,
            capture_output=True).stdout
        if archived[name] != blob:
            raise ValueError(f"fixture changed in the archive: {name}")
    return sorted(committed)


def check_tracked(root):
    """Require every declared build input to be tracked before source packaging.

    :param root: Repository root whose Git index defines the source archive.
    :return: Sorted tracked inventory; missing inputs raise ValueError.
    """
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True, text=True)
    tracked = set(result.stdout.split("\0"))
    inputs = set(REQUIRED)
    for crate in ("behave-fire", "behave-py"):
        inputs.update(path.relative_to(root).as_posix() for path in (root / f"rust/crates/{crate}").rglob("*")
            if path.is_file() and "target" not in path.parts and path.suffix not in {".pyc", ".crate"})
    inputs.update(path.relative_to(root).as_posix() for path in (root / "src/behave_fire").rglob("*") if path.is_file()
        and "__pycache__" not in path.parts and path.suffix.lower() not in {".pyc", ".pyd", ".so", ".dylib", ".pdb"})
    inputs.update(path.relative_to(root).as_posix() for path in (root / FIXTURES).glob("*.json"))
    missing = inputs - tracked
    if missing:
        raise ValueError(f"sdist requires tracked inputs: {sorted(missing)}")
    return sorted(inputs)


def check_wheel(path):
    """Check a native or pure wheel for optional reference checkout payloads.

    :param path: Wheel ZIP archive to inspect without extracting it.
    :return: Sorted member names; raises ValueError for a forbidden payload.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            _check_reference_payload(name)
    return sorted(names)


def main():
    """Check an archive or its pre-build tracked input inventory.

    :return: None; prints the verified inventory and fails on missing files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--check-tracked", type=Path)
    parser.add_argument("--committed-fixtures", type=Path, metavar="ROOT",
        help="with ARCHIVE, require its acceptance fixtures to equal ROOT's HEAD commit")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--wheel", type=Path)
    arguments = parser.parse_args()
    if arguments.check_tracked:
        inventory = check_tracked(arguments.check_tracked.resolve())
        if arguments.inventory:
            arguments.inventory.write_text("\n".join(inventory) + "\n", encoding="utf-8")
    if arguments.archive:
        expected = arguments.inventory.read_text(encoding="utf-8").splitlines() if arguments.inventory else ()
        print("\n".join(check_archive(arguments.archive, expected)))
        if arguments.committed_fixtures:
            check_committed_fixtures(arguments.archive, arguments.committed_fixtures.resolve())
    if arguments.wheel:
        print("\n".join(check_wheel(arguments.wheel)))
    if not arguments.check_tracked and not arguments.archive and not arguments.wheel:
        parser.error("provide archive, --wheel WHEEL, or --check-tracked ROOT")


if __name__ == "__main__":
    main()
