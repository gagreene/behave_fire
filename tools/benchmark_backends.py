"""Measure explicit Python and release Rust backends in independent processes.

Run after ``maturin develop --release``. Timings are diagnostics, not correctness
assertions. Peak working set includes the interpreter, inputs, and model state.
Native stage timings include real owned copies and chunk concatenation.
Detached intervals also include native setup, allocation, and result metadata work.
Use --operations typed-crown to measure the typed supplied-surface fallback separately.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


def _inputs(cells):
    """Build identical normalized one-dimensional input arrays.

    :param cells: Logical cell count.
    :return: Surface keyword arguments containing matching moisture arrays.
    """
    import numpy as np
    return dict(fuel_model_grid=np.full(cells, 10, dtype=np.int32),
                m1h=np.full(cells, .06), m10h=np.full(cells, .07),
                m100h=np.full(cells, .08), mlh=np.full(cells, .6),
                mlw=np.full(cells, .9), wind_speed=np.full(cells, 8.0),
                wind_speed_units=0, wind_direction=np.full(cells, 35.0),
                wind_orientation_mode="RelativeToUpslope", slope=np.full(cells, 12.0))


def _memory():
    """Read current and peak process resident bytes with native OS APIs.

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


def _worker(arguments):
    """Measure one isolated backend invocation and emit a structured record.

    :param arguments: Parsed backend, size, worker request, and operation.
    :return: JSON-compatible measurement record, including native copy timings.
    """
    import numpy as np
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    inputs = _inputs(arguments.cells)
    native_samples = []
    if arguments.backend == "rust":
        from behave_fire.rust import BehaveRun, _bridge
        original = _bridge._native.batch

        def profile(*args, **kwargs):
            """Collect private native timers around the ordinary model boundary.

            :param args: Native batch positional arguments.
            :param kwargs: Native batch keyword arguments.
            :return: Ordinary batch result with optional diagnostic metadata.
            """
            kwargs["profile"] = True
            started = time.perf_counter()
            result = original(*args, **kwargs)
            sample = {key.removeprefix("__timing."): float(value.item())
                      for key, value in result.items() if key.startswith("__timing.")}
            sample["native_boundary_total_seconds"] = time.perf_counter() - started
            sample["native_output_bytes"] = sum(value.nbytes for key, value in result.items() if not key.startswith("__timing."))
            sample["native_metadata_bytes"] = sum(value.nbytes for key, value in result.items() if key.startswith("__") and not key.startswith("__timing."))
            sample["operation"] = args[0]
            native_samples.append(sample)
            return result

        _bridge._native.batch = profile
        model = BehaveRun(FuelModels(), threads=arguments.threads)
    else:
        model = PythonRun(FuelModels())
    if arguments.operation in ("crown", "typed-crown"):
        prototype = PythonRun(FuelModels()).do_surface_run(**_inputs(1))
        surface = {key: np.full(arguments.cells, value.item(), dtype=value.dtype)
                   for key, value in prototype.items()}
        if arguments.operation == "typed-crown":
            surface["spread_rate"] = surface["spread_rate"].astype(np.float32)
            surface["fireline_intensity"] = np.full(arguments.cells, 200, dtype=np.uint16)
            surface["heat_per_unit_area"] = np.ma.array(surface["heat_per_unit_area"], mask=np.arange(arguments.cells) % 7 == 0)
        def call():
            """Invoke the crown pipeline with identical supplied surface values.

            :return: Crown result arrays from the selected explicit backend.
            """
            return model.do_crown_run(surface_results=surface, **inputs,
                                      canopy_base_height=4.0, canopy_height=30.0,
                                      canopy_bulk_density=.01, moisture_foliar=100.0)
    else:
        def call():
            """Invoke the surface pipeline with matching moisture array shapes.

            :return: Surface result arrays from the selected explicit backend.
            """
            return model.do_surface_run(**inputs)
    native_samples.clear()
    before_current, before_peak = _memory()
    stop = threading.Event()
    sampled = [before_current or 0]

    def sample_memory():
        """Sample current working set until the invocation finishes.

        :return: None; appends process memory measurements.
        """
        while not stop.wait(.005):
            sampled.append(_memory()[0] or 0)

    sampler = threading.Thread(target=sample_memory)
    sampler.start()
    started = time.perf_counter()
    try:
        result = call()
        total = time.perf_counter() - started
        current, peak = _memory()
    finally:
        stop.set()
        sampler.join()
    native_total = sum(sample["native_boundary_total_seconds"] for sample in native_samples)
    sums = {key: sum(sample[key] for sample in native_samples)
            for key in ("owned_copy_seconds", "detached_seconds", "concatenate_seconds",
                        "numpy_transfer_seconds", "appended_bytes")}
    output_bytes = sum(value.nbytes for value in result.values())
    return dict(backend=arguments.backend, threads=arguments.threads, cells=arguments.cells,
                operation=arguments.operation, total_seconds=total,
                python_marshalling_and_output_seconds=total-native_total,
                native_stages=sums, native_calls=native_samples, output_bytes=output_bytes,
                working_set_before_bytes=before_current, lifetime_peak_before_bytes=before_peak,
                lifetime_peak_after_bytes=peak, sampled_peak_during_call_bytes=max(sampled+[current or 0]),
                python=sys.version, numpy=np.__version__, platform=platform.platform(),
                cpu_count=os.cpu_count())


def main():
    """Run isolated measurements or emit one worker record.

    :return: None; writes JSON measurements to standard output and optional file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10_000, 1_000_000])
    parser.add_argument("--worker-threads", type=int, nargs="+", default=[1, 4, 0])
    parser.add_argument("--operations", nargs="+", default=["surface", "crown"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=["python", "rust"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cells", type=int)
    parser.add_argument("--operation", choices=["surface", "crown", "typed-crown"], default="surface")
    arguments = parser.parse_args()
    if arguments.worker:
        print(json.dumps(_worker(arguments)), flush=True)
        return
    records = []
    for cells in arguments.sizes:
        for operation in arguments.operations:
            for backend, workers in [("python", 1)] + [("rust", value) for value in arguments.worker_threads]:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--backend", backend,
                           "--threads", str(workers), "--cells", str(cells), "--operation", operation]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                if completed.returncode:
                    record = dict(backend=backend, threads=workers, cells=cells, operation=operation,
                                  error=completed.stderr, returncode=completed.returncode)
                else:
                    record = json.loads(completed.stdout)
                records.append(record)
                print(json.dumps(record), flush=True)
                if arguments.output:
                    arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if any("error" in record for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
