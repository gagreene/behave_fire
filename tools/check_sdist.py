"""Inspect source distributions and require the actual tracked Rust inputs."""

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

REQUIRED = {
    "pyproject.toml", "README.md", "LICENSE", "NOTICE", "LICENSES/NumPy-BSD-3-Clause.txt",
    "LICENSES/Intel-SVML-BSD-3-Clause.txt",
    "src/behave_fire/__init__.py", "rust/Cargo.toml", "rust/Cargo.lock",
    "rust/crates/behave-fire/Cargo.toml", "rust/crates/behave-fire/src/lib.rs",
    "rust/crates/behave-py/Cargo.toml", "rust/crates/behave-py/src/lib.rs",
}


def check_archive(path, expected=()):
    """Validate the archive inventory and reject unsafe or stale build inputs.

    :param path: Source tarball built by Maturin.
    :param expected: Complete source inventory captured before the build.
    :return: Sorted complete source inventory, after validating both crates.
    """
    with tarfile.open(path) as archive:
        names = set()
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or PurePosixPath(member.name).is_absolute() or ".." in parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
            name = "/".join(parts[1:])
            if member.isfile():
                names.add(name)
            if any(item in parts for item in (".git", "target", "__pycache__", ".superpowers")) or name.endswith((".pyd", ".so", ".pyc", ".pdb")):
                raise ValueError(f"build leftover in source distribution: {name}")
    missing = (REQUIRED | set(expected)) - names
    if missing:
        raise ValueError(f"missing source distribution files: {sorted(missing)}")
    for crate in ("behave-fire", "behave-py"):
        if not any(name.startswith(f"rust/crates/{crate}/src/") and name.endswith(".rs") for name in names):
            raise ValueError(f"missing Rust sources for {crate}")
    return sorted(names)


def check_tracked(root):
    """Require every source input selected for git-mode source packaging.

    :param root: Repository root whose Git index defines the source archive.
    :return: Sorted tracked inventory; missing inputs raise ValueError.
    """
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True, text=True)
    tracked = set(result.stdout.split("\0"))
    inputs = set(REQUIRED)
    for crate in ("behave-fire", "behave-py"):
        inputs.update(path.relative_to(root).as_posix() for path in (root / f"rust/crates/{crate}").rglob("*") if path.is_file() and "target" not in path.parts and path.suffix not in {".pyc", ".crate"})
    inputs.update(path.relative_to(root).as_posix() for path in (root / "src/behave_fire").rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() not in {".pyc", ".pyd", ".so", ".dylib", ".pdb"})
    missing = inputs - tracked
    if missing:
        raise ValueError(f"git-mode sdist requires tracked inputs: {sorted(missing)}")
    return sorted(inputs)


def main():
    """Check an archive or its pre-build tracked input inventory.

    :return: None; prints the verified inventory and fails on missing files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--check-tracked", type=Path)
    parser.add_argument("--inventory", type=Path)
    arguments = parser.parse_args()
    if arguments.check_tracked:
        inventory = check_tracked(arguments.check_tracked.resolve())
        if arguments.inventory:
            arguments.inventory.write_text("\n".join(inventory) + "\n", encoding="utf-8")
    if arguments.archive:
        expected = arguments.inventory.read_text(encoding="utf-8").splitlines() if arguments.inventory else ()
        print("\n".join(check_archive(arguments.archive, expected)))
    if not arguments.check_tracked and not arguments.archive:
        parser.error("provide archive or --check-tracked ROOT")


if __name__ == "__main__":
    main()
