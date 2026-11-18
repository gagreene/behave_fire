"""Independent temporary input snapshots for C++ evidence rejection tests."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath


def _snapshot_paths(root, names):
    """Resolve all names within their root before any input read or consumer write.

    :param root: Source or destination snapshot root.
    :param names: Relative manifest names.
    :return: Immutable name/path pairs; raises ValueError for root escapes.
    """
    root = Path(root).resolve()
    paths = []
    parents = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError('Snapshot names must be relative paths within their root')
        posix, windows = PurePosixPath(name), PureWindowsPath(name)
        if posix.is_absolute() or windows.anchor or '..' in posix.parts or '..' in windows.parts:
            raise ValueError('Snapshot names must be relative paths within their root')
        target = root / name
        if target.parent not in parents:
            parent = target.parent.resolve()
            if not parent.is_relative_to(root):
                raise ValueError('Snapshot path escapes its root')
            parents[target.parent] = parent
        path = target.resolve() if target.is_symlink() or target.exists() else parents[target.parent] / target.name
        if path == root or not path.is_relative_to(root):
            raise ValueError('Snapshot path escapes its root')
        paths.append((name, path))
    return tuple(paths)


def capture_snapshot(source_root):
    """Capture source inputs for later independent snapshot consumers.

    :param source_root: Pristine checkout holding artifacts and qualified sources.
    :return: Immutable tuples of relative paths and captured bytes.
    """
    source_root = Path(source_root).resolve()
    metadata = dict(_snapshot_paths(source_root, (
        'tests/data/cpp_reference/hashes.json', 'tests/data/cpp_reference/calibration/qualification.json')))
    hashes = json.loads(metadata['tests/data/cpp_reference/hashes.json'].read_text())
    qualification = metadata['tests/data/cpp_reference/calibration/qualification.json']
    sources = json.loads(qualification.read_text()).get('source_files', {}) if qualification.is_file() else {}
    paths = _snapshot_paths(source_root, sorted({*hashes, *sources, 'tests/data/cpp_reference/hashes.json'}))
    return tuple((name, path.read_bytes()) for name, path in paths)


def write_snapshot(snapshot, destination):
    """Write independent snapshot files beneath the requested destination.

    :param snapshot: Previously captured snapshot description.
    :param destination: Temporary consumer root.
    :return: Destination root, after writing the copied files.
    """
    destination = Path(destination).resolve()
    snapshot = tuple(snapshot)
    paths = _snapshot_paths(destination, (name for name, _ in snapshot))
    for parent in {path.parent for _, path in paths}:
        parent.mkdir(parents=True, exist_ok=True)
    for (_, path), (_, payload) in zip(paths, snapshot, strict=True):
        path.write_bytes(payload)
    return destination
