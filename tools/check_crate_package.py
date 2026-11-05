"""Check a standalone behave-fire archive without extracting or executing it."""

import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import tarfile
import tomllib

MAX_COMPRESSED_BYTES = 5 * 1024 * 1024
REQUIRED_FILES = frozenset({
    "Cargo.toml", "Cargo.lock", "README.md", "LICENSE", "NOTICE",
    "LICENSES/NumPy-BSD-3-Clause.txt", "LICENSES/Intel-SVML-BSD-3-Clause.txt",
    "examples/standalone.rs",
    "src/lib.rs", "src/model.rs", "src/grid.rs", "src/math.rs", "src/units.rs",
    "src/fuel_models.rs", "src/species.rs", "src/surface/mod.rs", "src/crown.rs",
    "src/mortality.rs", "src/spot.rs", "src/ignite.rs", "src/safety.rs",
    "src/vapor_pressure_deficit.rs", "src/fine_dead_fuel_moisture.rs",
    "src/fire_size.rs", "src/slope.rs", "src/contain.rs",
    "src/generated/fuel_models.rs", "src/generated/species.rs",
    "src/generated/fine_dead_fuel_moisture.rs",
    "tests/data/fuels.json", "tests/data/species.json", "tests/data/surface.json",
    "tests/data/crown.json", "tests/data/effects.json", "tests/data/containment.json",
    "tests/data/tools.json", "tests/data/units.json",
})


def _check_includes(files):
    """Require Rust include macros to resolve within the archive.

    :param files: Archive-relative file names and bytes.
    :return: None; raises ValueError for missing, external, or generated includes.
    """
    pattern = re.compile(r'\binclude(?:_str|_bytes)?\s*!\s*\((.*?)\)', re.DOTALL)
    for name, data in files.items():
        if not name.endswith(".rs"):
            continue
        source = data.decode("utf-8")
        for match in pattern.finditer(source):
            literal = re.fullmatch(r'\s*"([^"\\]+)"\s*,?\s*', match.group(1))
            if literal is None:
                raise ValueError(f"generated or nonliteral include in {name}")
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), literal.group(1)))
            if target.startswith("../") or target.startswith("/") or target not in files:
                raise ValueError(f"external or missing include in {name}: {target}")


def _check_manifest(files, prefix):
    """Validate normalized standalone package identity and dependency policy.

    :param files: Archive-relative file names and bytes.
    :param prefix: Single top-level archive directory.
    :return: Parsed normalized Cargo manifest.
    """
    manifest = tomllib.loads(files["Cargo.toml"].decode("utf-8"))
    package = manifest.get("package", {})
    name, version = package.get("name"), package.get("version")
    if name != "behave-fire" or not isinstance(version, str) or prefix != f"{name}-{version}":
        raise ValueError("crate identity does not match archive directory")
    if package.get("license") != "MIT AND BSD-3-Clause":
        raise ValueError("crate license expression must retain MIT and NumPy BSD notices")
    if "build.rs" in files or package.get("build", False):
        raise ValueError("standalone core must not require a build script")
    sections = [manifest, *manifest.get("target", {}).values()]
    if any(section.get(key) for section in sections for key in ("dependencies", "build-dependencies")):
        raise ValueError("standalone core must have no runtime or build dependencies")
    if "workspace" in manifest or any(isinstance(value, dict) and value.get("workspace") for value in package.values()):
        raise ValueError("normalized crate manifest still depends on a workspace")
    return manifest


def inspect_crate(path):
    """Verify package size, contents, metadata, and archive-local Rust includes.

    :param path: Path to a cargo-generated .crate archive.
    :return: JSON-compatible package verification record; raises ValueError on failure.
    """
    path = Path(path)
    compressed = path.stat().st_size
    if compressed > MAX_COMPRESSED_BYTES:
        raise ValueError(f"compressed crate exceeds {MAX_COMPRESSED_BYTES}-byte budget: {compressed}")
    files = {}
    prefix = None
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            parts = entry.name.rstrip("/").split("/")
            if entry.name.startswith("/") or "\\" in entry.name or any(not part or part in ("..", ".") or ":" in part for part in parts):
                raise ValueError(f"unsafe archive path: {entry.name}")
            if len(parts) < 2 or not parts[0]:
                raise ValueError(f"missing package directory in archive path: {entry.name}")
            if prefix is None:
                prefix = parts[0]
            if prefix != parts[0]:
                raise ValueError("multiple package directories in archive paths")
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError(f"nonregular archive path: {entry.name}")
            name = PurePosixPath(*parts[1:]).as_posix()
            if name in files:
                raise ValueError(f"duplicate archive path: {name}")
            if name.endswith((".py", ".pyd", ".so", ".dylib", ".dll", ".exe")) or "rust_port" in parts:
                raise ValueError(f"non-core artifact in archive path: {name}")
            with archive.extractfile(entry) as stream:
                files[name] = stream.read()
    missing = REQUIRED_FILES.difference(files)
    if missing:
        raise ValueError(f"missing required crate files: {', '.join(sorted(missing))}")
    manifest = _check_manifest(files, prefix)
    _check_includes(files)
    return {
        "name": manifest["package"]["name"],
        "version": manifest["package"]["version"],
        "compressed_bytes": compressed,
        "file_count": len(files),
        "normal_dependencies": 0,
    }


def main():
    """Check a built archive and print its verification record.

    :return: None; exits nonzero for an invalid or incomplete package.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="cargo-generated .crate file")
    args = parser.parse_args()
    try:
        report = inspect_crate(args.archive)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"crate check failed: {error}\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
