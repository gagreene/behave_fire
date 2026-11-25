"""Time a standard Surface and Crown run on each available backend.

The output is a report for one machine. Run times depend on hardware, load, and
build, so this script never compares them with stored values and never passes or
fails on time.

Run from the repository root::

    python tools/time_standard_runs.py [--cells N] [--repeats N] [--threads N] [--json PATH]
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from raster_cases import model_call  # noqa: E402

MODELS = ("surface", "crown")


def _host():
    """Describe the machine and software that produced a report.

    :return: CPU, logical core count, platform, and Python, NumPy, and package versions.
    """
    import numpy as np

    import behave_fire
    return dict(cpu=platform.processor() or platform.machine(), logical_cores=os.cpu_count(),
                platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                behave_fire=behave_fire.__version__)


def _time(call, repeats):
    """Time repeated calls after one untimed warm-up call.

    :param call: Zero-argument model call with its inputs already prepared.
    :param repeats: Number of timed calls.
    :return: Median and minimum wall time in seconds.
    """
    call()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return float(statistics.median(elapsed)), float(min(elapsed))


def format_table(report):
    """Format a report as plain text.

    :param report: Result of :func:`run`.
    :return: One header line with host and settings, then one line per model and backend.
    """
    host, settings = report["host"], report["settings"]
    lines = [f"behave_fire {host['behave_fire']} | {host['cpu']} | {host['logical_cores']} logical cores | "
             f"{host['platform']} | Python {host['python']} | NumPy {host['numpy']} | "
             f"{settings['cells']} cells, {settings['repeats']} repeats, {settings['threads']} thread(s)",
             f"{'model':<8}  {'backend':<8}  {'median s':>10}  {'min s':>10}"]
    for row in report["rows"]:
        lines.append(f"{row['model']:<8}  {row['backend']:<8}  {row['median_s']:>10.4f}  {row['min_s']:>10.4f}")
    return "\n".join(lines)


def main(argv=None):
    """Print a timing report and optionally save it as JSON.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    :return: ``0`` after a completed run; writes the JSON report when ``--json`` is given.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = run(cells=args.cells, repeats=args.repeats, threads=args.threads)
    print(format_table(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def run(cells=100_000, repeats=5, threads=1, backends=None, seed=20261105):
    """Time the standard Surface and Crown runs.

    :param cells: Raster cells per run.
    :param repeats: Timed calls per model and backend, after one warm-up call.
    :param threads: Native worker count for the Rust backend.
    :param backends: Backends to time; ``None`` means Python plus Rust when it is usable.
    :param seed: Stable phase offset for the heterogeneous inputs.
    :return: Host details, settings, and one row per model and backend with median and
        minimum seconds.
    """
    import behave_fire
    if backends is None:
        backends = ("python", "rust") if "rust" in behave_fire.available_backends() else ("python",)
    rows = []
    for model in MODELS:
        for backend in backends:
            call = model_call(model, "heterogeneous", cells, backend, threads=threads, seed=seed)
            median, minimum = _time(call, repeats)
            rows.append(dict(model=model, backend=backend, median_s=median, min_s=minimum))
    settings = dict(cells=cells, repeats=repeats, threads=threads, backends=list(backends),
                    dataset="heterogeneous", seed=seed)
    return dict(host=_host(), settings=settings, rows=rows)


if __name__ == "__main__":
    raise SystemExit(main())
