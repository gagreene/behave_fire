"""Build isolated timed/instrumented binaries and collect scoped allocation evidence.

Run with the configured Python interpreter. The current package extension is
never replaced. All observations run in fresh child processes, and this tool
never emits performance timings or certifies an unmeasured encoder.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from assemble_allocation_evidence import _hash
from performance_contract import requested_outputs, save_outputs

ROOT = Path(__file__).resolve().parents[1]


def _build(target, output, instrumented):
    """Compile one release artifact and preserve its exact bytes.

    :param target: Dedicated Cargo target directory.
    :param output: New artifact directory.
    :param instrumented: Whether to enable the private counting feature.
    :return: Path of the preserved importable extension.
    """
    output.mkdir(parents=True, exist_ok=False)
    command = ["cargo", "build", "--manifest-path", str(ROOT / "rust/Cargo.toml"),
               "-p", "behave-py", "--release", "--locked", "--verbose", "--no-default-features",
               "--target-dir", str(target)]
    if instrumented:
        command.extend(["--features", "allocation-probe"])
    environment = dict(os.environ, PYO3_PYTHON=sys.executable)
    with (output / "build.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    library = "_rust.dll" if os.name == "nt" else ("lib_rust.dylib" if sys.platform == "darwin" else "lib_rust.so")
    extension = output / ("_rust.pyd" if os.name == "nt" else "_rust.so")
    shutil.copyfile(target / "release" / library, extension)
    return extension


def _child(arguments):
    """Run a new worker and require complete JSON output.

    :param arguments: Child command arguments.
    :return: Parsed worker report.
    """
    completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *map(str, arguments)],
                               cwd=ROOT, text=True, capture_output=True, check=True, timeout=600)
    return json.loads(completed.stdout)


def _inventory():
    """Fingerprint source/build inputs, excluding generated artifacts.

    :return: Relative filenames and SHA-256 digests.
    """
    paths = []
    for directory in (ROOT / "src/behave_fire", ROOT / "rust/crates", ROOT / "tools"):
        paths.extend(p for p in directory.rglob("*") if p.is_file()
                     and not {"target", "__pycache__"}.intersection(p.relative_to(directory).parts)
                     and p.suffix in (".py", ".rs", ".toml", ".lock", ".S", ".s", ".c", ".h"))
    paths.extend(ROOT / name for name in ("rust/Cargo.toml", "rust/Cargo.lock", "pyproject.toml"))
    for directory in (ROOT / ".cargo", ROOT / "rust/.cargo"):
        paths.extend(directory.glob("*.toml"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(set(paths))}


def _load(extension):
    """Load only the chosen binary before importing the public Rust backend.

    :param extension: Preserved timed or instrumented native binary.
    :return: Native module and matching benchmark driver.
    """
    sys.path.insert(0, str(ROOT / "src"))
    import behave_fire
    if "behave_fire._rust" in sys.modules:
        raise RuntimeError("Native module was imported before diagnostic selection")
    spec = importlib.util.spec_from_file_location("behave_fire._rust", extension)
    native = importlib.util.module_from_spec(spec)
    sys.modules["behave_fire._rust"] = native
    spec.loader.exec_module(native)
    behave_fire._rust = native
    import benchmark_backends
    return native, benchmark_backends


def _save(path, value):
    """Persist JSON evidence without non-finite numbers.

    :param path: Destination path.
    :param value: Evidence to preserve.
    :return: None; writes the file.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _worker(args):
    """Measure actual production kernel scopes and independently validate outputs.

    :param args: Child route, sample, artifact, and manifest selectors.
    :return: Raw identity or scoped observation.
    """
    native, driver = _load(args.extension.resolve())
    controls = SimpleNamespace(cells=args.cells, operation=args.operation, dataset=args.dataset,
                               seed=20261105, threads=args.workers)
    identity = driver._provenance(controls)
    identity["extension_hash"] = hashlib.sha256(args.extension.read_bytes()).hexdigest()
    identity["allocation_probe"] = hasattr(native, "_allocation_start")
    identity['binding_api_version'] = native.__binding_api_version__
    if args.identity:
        return identity
    if not identity["allocation_probe"]:
        raise RuntimeError("Selected binary has no allocation probe")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (identity["extension_hash"] != manifest["instrumented"]["extension_hash"]
            or identity["source_hash"] != manifest["instrumented"]["source_hash"]
            or _inventory() != manifest["instrumented"]["source_files"]):
        raise RuntimeError("Instrumented source or binary changed since build")
    call = driver._model_call(controls, "rust")
    call()  # Complete one bounded warm-up before the measured call.
    native._allocation_start()
    try:
        actual = call()
    finally:
        counters = native._allocation_stop()
    expected = driver._model_call(controls, "python", normalize_reference=True)()
    driver.assert_outputs(actual, expected)
    scopes = {}
    for index, name in ((1, "kernel"), (2, "encoder")):
        requests, byte_count, calls, workers = counters[index]
        scopes[name] = ({"requests": requests, "bytes": byte_count, "calls": calls, "workers": workers,
                         "definition": name + "-loop-v1"} if calls else None)
    path = args.manifest.parent/'outputs'/f'{args.dataset}-{args.operation}-{args.cells}-{args.workers}-{args.repeat}.npz'
    fields = requested_outputs(args.operation)
    return {**call.evidence, 'fields': fields, 'compared_fields': fields, 'output_fields': len(fields),
            'independent_python': True, 'correctness_contract': 'f64-nan-v1',
            'output_artifact': save_outputs(path, actual, expected, root=ROOT),
            "dataset": args.dataset, "operation": args.operation, "cells": args.cells,
            "workers": args.workers, "repeat": args.repeat, "process_id": os.getpid(),
            "instrumented_source_hash": identity["source_hash"],
            "instrumented_extension_hash": identity["extension_hash"],
            "manifest_sha256": _hash(manifest), "passed": True, "scopes": scopes,
            "outside_scopes": {"requests": counters[0][0], "bytes": counters[0][1]},
            "python": identity["python"], "numpy": identity["numpy"], "numpy_build": identity["numpy_build"]}


def main():
    """Build and measure, or execute one isolated observation.

    :return: None; writes manifest/raw evidence or prints one worker result.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["uniform", "heterogeneous"])
    parser.add_argument("--operations", nargs="+", default=["surface", "crown", "typed-crown", "typed-crown-b"])
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset", default="uniform")
    parser.add_argument("--operation", default="surface")
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    if args.extension:
        print(json.dumps(_worker(args), allow_nan=False))
        return
    if args.output_dir is None or args.target_dir is None:
        parser.error("Require --output-dir and --target-dir for controlled builds")
    output, target = args.output_dir.resolve(), args.target_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = _inventory()
    compiler = subprocess.check_output(["rustc", "-vV"], text=True)
    target_name = next(line.split(": ", 1)[1] for line in compiler.splitlines() if line.startswith("host:"))
    builds = {}
    binaries = {}
    for name, instrumented in (("timed", False), ("instrumented", True)):
        extension = _build(target, output / name, instrumented)
        binaries[name] = extension
        identity = _child(["--extension", extension, "--identity"])
        if identity["allocation_probe"] != instrumented or identity['binding_api_version'] != 4 or _inventory() != before:
            raise RuntimeError("Build feature mismatch or source changed during builds")
        builds[name] = {'binding_api_version': identity['binding_api_version'], "source_hash": identity["source_hash"], "extension_hash": identity["extension_hash"],
                        "source_files": before, "compiler": compiler, "target": target_name, "profile": "release",
                        "features": ["allocation-probe"] if instrumented else [],
                        "build_environment": {k: v for k, v in os.environ.items() if k in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET",
                            "CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
                            "CARGO_PROFILE_RELEASE_OPT_LEVEL", "CARGO_PROFILE_RELEASE_DEBUG",
                            "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")}}
    manifest = {"schema_version": 1, 'binding_api_version': 4, **builds, "source_differences": [],
                "patch_sha256": hashlib.sha256(b"").hexdigest()}
    manifest_path = output / "manifest.json"
    _save(manifest_path, manifest)
    raw = []
    evidence = {"manifest": manifest, "manifest_sha256": _hash(manifest), "raw": raw}
    for dataset, operation, cells, workers, repeat in itertools.product(
            args.datasets, args.operations, (1000, 10000), (1, 4), range(3)):
        row = _child(["--extension", binaries["instrumented"], "--manifest", manifest_path,
                      "--dataset", dataset, "--operation", operation, "--cells", cells,
                      "--workers", workers, "--repeat", repeat])
        raw.append(row)
        evidence["raw_sha256"] = _hash(raw)
        _save(output / "evidence.json", evidence)
        print(f"{dataset}/{operation}/{cells}/{workers}/{repeat}: {row['scopes']}", flush=True)


if __name__ == "__main__":
    main()
