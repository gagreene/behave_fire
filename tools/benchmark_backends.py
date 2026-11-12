"""Measure explicit Python and release Rust in paired fresh processes.

Correctness checks use separate workers, keeping oracle arrays out of timed
peak memory. Profiling is opt-in and cannot certify performance. The first call
warms the implementation for the subsequent warmed observations.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

from performance_cases import OPERATIONS, assert_outputs, build_inputs, paired_schedule, supplied_surface
from performance_contract import input_identity, requested_outputs, save_outputs


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _array_bytes(value):
    """Count array payload bytes through the native result's nested schema.

    :param value: An array, nested schema mapping, or scalar schema value.
    :return: Array storage bytes, excluding Python objects and scalar descriptors.
    """
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item) for item in value)
    return int(getattr(value, "nbytes", 0))

def _campaign(arguments):
    """Run separate validation and shuffled timing blocks, retaining partial failures.

    :param arguments: Parsed campaign controls.
    :return: Complete or partial JSON-compatible measurement report.
    """
    report = {"schema_version": 1, "preset": arguments.preset, "profile": arguments.profile,
              "physical_cores": arguments.physical_cores or _physical_cores(),
              "records": [], "validations": [], "errors": [],
              "baseline": _read_rows(arguments.baseline, "records"),
              "allocations": [row for path in arguments.allocations for row in _read_rows(path, "allocations")]}
    for dataset in arguments.datasets:
        for cells in arguments.sizes:
            for operation in arguments.operations:
                validation_failed = False
                for threads in arguments.worker_threads:
                    command = _command(arguments, cells, operation, dataset, "rust", threads, True)
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=arguments.timeout)
                        if completed.returncode:
                            raise RuntimeError(completed.stderr or completed.stdout)
                        result = json.loads(completed.stdout)
                        report["validations"].append(result)
                        if not result.get("passed"):
                            report["errors"].append(result)
                            validation_failed = True
                    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
                        report["errors"].append({"kind": "validation", "dataset": dataset, "cells": cells,
                                                  "operation": operation, "threads": threads, "error": str(error)})
                        validation_failed = True
                    _save(report, arguments.output)
                if arguments.validate_only or validation_failed:
                    continue
                for block, backend, threads in paired_schedule(arguments.repetitions, arguments.seed,
                                                               arguments.worker_threads):
                    command = _command(arguments, cells, operation, dataset, backend, threads, False)
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=arguments.timeout)
                        if completed.returncode:
                            raise RuntimeError(completed.stderr or completed.stdout)
                        result = json.loads(completed.stdout)
                        result["block"] = block
                        report["records"].append(result)
                        print(json.dumps(result, allow_nan=False), flush=True)
                    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
                        report["errors"].append({"kind": "timing", "dataset": dataset, "cells": cells,
                                                  "operation": operation, "backend": backend,
                                                  "threads": threads, "block": block, "error": str(error)})
                    _save(report, arguments.output)
    return report


def _command(arguments, cells, operation, dataset, backend, threads, validation):
    """Construct a worker argument vector without invoking a shell.

    :param arguments: Campaign configuration.
    :param cells: Logical cell count.
    :param operation: Route to measure or validate.
    :param dataset: Deterministic input dataset.
    :param backend: Explicit implementation selector.
    :param threads: Native worker request.
    :param validation: Whether to compare results instead of timing calls.
    :return: Worker command using the current configured interpreter.
    """
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--backend", backend,
               "--threads", str(threads), "--cells", str(cells), "--operation", operation,
               "--dataset", dataset, "--seed", str(arguments.seed), "--warm-calls", str(arguments.warm_calls)]
    if validation:
        command.append("--validate-only")
        output = getattr(arguments, 'output', None)
        if output is not None:
            artifact = output.parent/(output.stem+'-outputs')/f'{dataset}-{operation}-{cells}-{threads}.npz'
            command.extend(['--artifacts', str(artifact)])
    elif arguments.profile:
        command.append("--profile")
    return command


def _memory():
    """Read current and peak resident bytes through native operating-system APIs.

    :return: Current and lifetime peak resident bytes; current is None on POSIX.
    """
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            """Windows PROCESS_MEMORY_COUNTERS structure."""
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return None, peak if sys.platform == "darwin" else peak * 1024


def _model_call(arguments, backend, *, normalize_reference=False):
    """Prepare a model and identical inputs outside the timed interval.

    :param arguments: Worker route, dataset, size, seed, and worker count.
    :param backend: Explicit Python or Rust implementation.
    :param normalize_reference: Correctness-only widening; never enabled by timed workers.
    :return: Callable retaining its explicit model and prepared inputs.
    """
    import numpy as np
    if normalize_reference and backend != 'python':
        raise ValueError('Only the independent Python correctness oracle can normalize reference inputs')
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    inputs = build_inputs(arguments.cells, arguments.dataset, arguments.seed)
    if backend == "rust":
        from behave_fire.rust import BehaveRun
        model = BehaveRun(FuelModels(), threads=arguments.threads)
    else:
        model = PythonRun(FuelModels())
    original = dict(inputs)
    if arguments.operation == "surface":
        def call():
            """Invoke Surface with normalized shape-matching input arrays.

            :return: Complete Surface output dictionary.
            """
            if normalize_reference:
                from float64_model_reference import model_reference
                return model_reference(model.do_surface_run, inputs)
            return model.do_surface_run(**inputs)
    else:
        surface = supplied_surface(inputs, arguments.operation, PythonRun(FuelModels()),
                                   uniform=arguments.dataset == "uniform")
        canopy = dict(canopy_base_height=4.0, canopy_height=30.0, canopy_bulk_density=.01, moisture_foliar=100.0)
        if arguments.dataset == "heterogeneous":
            phase = ((np.arange(arguments.cells) + arguments.seed) % 17) / 16
            canopy = dict(canopy_base_height=1 + 5 * phase, canopy_height=15 + 25 * phase,
                          canopy_bulk_density=.005 + .02 * phase, moisture_foliar=80 + 40 * phase)
        original.update(surface_results=surface, **canopy)

        def call():
            """Invoke Crown with independently typed supplied Surface fields.

            :return: Complete Crown output dictionary.
            """
            if normalize_reference:
                from float64_model_reference import model_reference
                return model_reference(model.do_crown_run, dict(surface_results=surface, **inputs, **canopy))
            return model.do_crown_run(surface_results=surface, **inputs, **canopy)
    actual_backend = ('rust' if type(model).__module__.startswith('behave_fire.rust.') else
                      'python' if type(model).__module__ == 'behave_fire.behave' else 'unknown')
    call.evidence = dict(binding_api_version=4, actual_backend=actual_backend,
                        input_sha256=input_identity(original),
                        requested_outputs=requested_outputs(arguments.operation),
                        eligible_cells=arguments.cells, eligibility={'kind': 'all'},
                        python_timing_original_inputs=actual_backend == 'python' and not normalize_reference)
    return call


def _physical_cores():
    """Read physical core availability without substituting logical CPU count.

    :return: Physical core count when verifiable, otherwise None.
    """
    try:
        if os.name == "nt":
            command = ["powershell.exe", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"]
            return int(subprocess.check_output(command, text=True, timeout=30).strip())
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True, timeout=30))
        records = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")
        available = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        cores = set()
        for record in records:
            fields = dict(line.split(":", 1) for line in record.splitlines() if ":" in line)
            fields = {key.strip(): value.strip() for key, value in fields.items()}
            if "physical id" in fields and "core id" in fields:
                if available is None or int(fields["processor"]) in available:
                    cores.add((fields["physical id"], fields["core id"]))
        return len(cores) or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _profile_batch(original, samples, *, timers=True):
    """Instrument the current native batch entry point for diagnostics only.

    :param original: Unmodified native batch function.
    :param samples: Destination for per-boundary diagnostic samples.
    :param timers: Whether this boundary supports optional internal timing fields.
    :return: Instrumented callable preserving ordinary batch results.
    """
    def profile(*args, **kwargs):
        """Collect diagnostic timers around one native boundary call.

        :param args: Native positional arguments.
        :param kwargs: Native keyword arguments.
        :return: Ordinary result with optional diagnostic metadata.
        """
        if timers:
            kwargs["profile"] = True
        started = time.perf_counter()
        result = original(*args, **kwargs)
        sample = {key.removeprefix("__timing."): float(value.item())
                  for key, value in result.items() if key.startswith("__timing.")}
        sample["native_boundary_total_seconds"] = time.perf_counter() - started
        sample["native_output_bytes"] = sum(_array_bytes(value) for key, value in result.items()
                                            if not key.startswith("__timing."))
        sample["native_metadata_bytes"] = sum(_array_bytes(value) for key, value in result.items()
                                              if key.startswith("__") and not key.startswith("__timing."))
        sample["operation"] = args[0]
        samples.append(sample)
        return result
    return profile


def _provenance(arguments):
    """Hash sources, native binary, and NumPy build after measurements finish.

    :param arguments: Worker input identity.
    :return: Version and hash fields without developer-machine absolute paths.
    """
    import numpy as np
    paths = sorted((ROOT / "src/behave_fire").rglob("*.py"))
    paths += sorted((ROOT / "rust/crates").glob("*/src/**/*.rs"))
    paths += [ROOT / "rust/Cargo.toml", ROOT / "rust/crates/behave-fire/Cargo.toml",
              ROOT / "rust/crates/behave-py/Cargo.toml", ROOT / "rust/Cargo.lock",
              ROOT / "pyproject.toml", Path(__file__).resolve(), ROOT / "tools/performance_cases.py"]
    paths += [ROOT/'tools'/name for name in ('performance_contract.py', 'check_performance.py',
              'float64_model_reference.py', 'core_f64_reference.py', 'assemble_allocation_evidence.py',
              'measure_native_allocations.py', 'performance_requirements.json')]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    candidates = [ROOT / "src/behave_fire" / ("_rust" + suffix) for suffix in EXTENSION_SUFFIXES]
    extension = next((path for path in candidates if path.is_file()), None)
    with contextlib.redirect_stdout(io.StringIO()) as build:
        np.show_config()
    return {"python": sys.version, "numpy": np.__version__,
            "numpy_build": hashlib.sha256(build.getvalue().encode()).hexdigest(),
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "source_hash": digest.hexdigest(),
            "extension_hash": hashlib.sha256(extension.read_bytes()).hexdigest() if extension else None,
            "seed": arguments.seed}


def _read_rows(path, field):
    """Load report rows or probe JSON/JSONL without inventing absent evidence.

    :param path: Optional evidence file.
    :param field: Report field to extract.
    :return: Existing rows, or an empty list when no file is supplied.
    """
    if path is None:
        return []
    text = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    return value[field] if field in value else [value]


def _save(report, destination):
    """Persist partial evidence after every observation.

    :param report: JSON-compatible campaign evidence.
    :param destination: Optional output path.
    :return: None; writes the report when a destination is supplied.
    """
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def _validation_worker(arguments):
    """Compare full outputs separately from all timing and memory observations.

    :param arguments: Operation, dtype layout, dataset, seed, and worker request.
    :return: Validation status and matching input/source provenance.
    """
    expected = _model_call(arguments, 'python', normalize_reference=True)()
    from behave_fire import _rust
    calls = []
    originals = {name: getattr(_rust, name) for name in ('component_batch', 'model_batch', 'model_groups')}

    def instrument(name, original):
        """Record an untimed real native boundary without changing its operands.

        :param name: Native boundary identity.
        :param original: Unmodified native callable.
        :return: Transparent diagnostic wrapper.
        """
        def invoke(*args, **kwargs):
            """Record and execute exactly one native operation.

            :param args: Original positional arguments.
            :param kwargs: Original keyword arguments.
            :return: Unmodified native result.
            """
            calls.append(name)
            return original(*args, **kwargs)
        return invoke

    try:
        for name, function in originals.items():
            setattr(_rust, name, instrument(name, function))
        call = _model_call(arguments, "rust")
        actual = call()
    finally:
        for name, function in originals.items():
            setattr(_rust, name, function)
    result = dict(kind="validation", dataset=arguments.dataset, operation=arguments.operation,
                  cells=arguments.cells, threads=arguments.threads, output_fields=len(expected), passed=False,
                  correctness_contract='f64-nan-v1', independent_python=True,
                  python_timing_original_inputs=True, fields=sorted(expected), native_boundary_calls=calls)
    result.update(call.evidence)
    try:
        assert_outputs(actual, expected)
        result['compared_fields'] = sorted(actual)
        artifact = getattr(arguments, 'artifacts', None)
        result['output_artifact'] = save_outputs(artifact, actual, expected, root=ROOT) if artifact else None
        result["passed"] = True
    except AssertionError as error:
        result["error"] = str(error)
    result.update(_provenance(arguments))
    return result


def _worker(arguments):
    """Measure first and warmed calls without retaining separate oracle arrays.

    :param arguments: Backend, route, size, worker count, and profiling controls.
    :return: Legacy first-call keys, repeated observations, and exact provenance.
    """
    import_started = time.perf_counter()
    import numpy as np
    from behave_fire.python import BehaveRun as PythonRun
    if arguments.backend == "rust":
        from behave_fire.rust import _bridge
    import_seconds = time.perf_counter() - import_started
    samples = []
    if arguments.backend == "rust" and arguments.profile:
        _bridge._native.component_batch = _profile_batch(_bridge._native.component_batch, samples, timers=False)
        _bridge._native.model_batch = _profile_batch(_bridge._native.model_batch, samples)
    created = time.perf_counter()
    call = _model_call(arguments, arguments.backend)
    construction_seconds = time.perf_counter() - created
    samples.clear()
    before_current, before_peak = _memory()
    stop = threading.Event()
    sampled = [before_current or 0]

    def sample_memory():
        """Sample resident memory until all worker calls finish.

        :return: None; appends current-working-set measurements when available.
        """
        while not stop.wait(.005):
            sampled.append(_memory()[0] or 0)

    sampler = threading.Thread(target=sample_memory)
    sampler.start()
    durations = []
    calls = []
    try:
        for _ in range(arguments.warm_calls + 1):
            result = None
            samples.clear()
            started = time.perf_counter()
            result = call()
            durations.append(time.perf_counter() - started)
            calls.append(list(samples))
        current, peak = _memory()
    finally:
        stop.set()
        sampler.join()
    first_samples = calls[0]
    sums = {key: sum(sample.get(key, 0) for sample in first_samples)
            for key in ("owned_copy_seconds", "detached_seconds", 'kernel_seconds', 'encoder_seconds', "concatenate_seconds",
                        "numpy_transfer_seconds", "appended_bytes")} if arguments.profile else None
    native_total = sum(sample["native_boundary_total_seconds"] for sample in first_samples)
    record = dict(kind="timing", backend=arguments.backend, threads=arguments.threads,
                  dataset=arguments.dataset, cells=arguments.cells, operation=arguments.operation,
                  first_seconds=durations[0], total_seconds=durations[0], warm_seconds=durations[1:],
                  profile=arguments.profile, import_seconds=import_seconds,
                  python_timing_original_inputs=arguments.backend == 'python',
                  actual_backend=arguments.backend,
                  construction_and_inputs_seconds=construction_seconds,
                  python_marshalling_and_output_seconds=(durations[0] - native_total) if arguments.profile else None,
                  native_stages=sums, native_calls=first_samples, warm_native_calls=calls[1:],
                  output_bytes=sum(value.nbytes for value in result.values()),
                  working_set_before_bytes=before_current, lifetime_peak_before_bytes=before_peak,
                  lifetime_peak_after_bytes=peak, peak_bytes=peak,
                  sampled_peak_during_call_bytes=max(sampled + [current or 0]))
    record.update(_provenance(arguments))
    record.update(call.evidence)
    return record


def main():
    """Run paired campaigns or one worker using the original CLI selectors.

    :return: None; prints JSON measurements and exits nonzero on worker failures.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10_000, 1_000_000])
    parser.add_argument("--worker-threads", type=int, nargs="+", default=[1, 4, 0])
    parser.add_argument("--operations", choices=OPERATIONS, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=["python", "rust"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cells", type=int)
    parser.add_argument("--operation", choices=OPERATIONS, default="surface")
    parser.add_argument("--dataset", choices=["uniform", "heterogeneous", "continuous"], default="uniform")
    parser.add_argument("--datasets", choices=["uniform", "heterogeneous", "continuous"], nargs="+")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--preset", choices=["screen", "acceptance"], default="screen")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warm-calls", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--allocations", type=Path, nargs="*", default=[])
    parser.add_argument("--physical-cores", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    args.repetitions = args.repetitions if args.repetitions is not None else (7 if args.preset == "acceptance" else 3)
    args.warm_calls = args.warm_calls if args.warm_calls is not None else (5 if args.preset == "acceptance" else 1)
    args.datasets = args.datasets or (["uniform", "heterogeneous"] if args.preset == "acceptance" else ["uniform"])
    args.operations = args.operations or (list(OPERATIONS) if args.preset == "acceptance" else ["surface", "crown"])
    if args.repetitions < 1 or args.warm_calls < 0 or any(size < 0 for size in args.sizes):
        parser.error("Require positive repetitions and nonnegative sizes/warmed-call counts")
    if args.worker:
        if args.cells is None or args.cells < 0 or args.backend is None or args.threads < 0:
            parser.error("A worker requires nonnegative --cells/--threads and an explicit --backend")
        result = _validation_worker(args) if args.validate_only else _worker(args)
        print(json.dumps(result, allow_nan=False), flush=True)
    else:
        if args.output is not None and args.output.exists():
            parser.error('Refuse to overwrite an existing campaign; use a new output path')
        result = _campaign(args)
        _save(result, args.output)
        if result["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
