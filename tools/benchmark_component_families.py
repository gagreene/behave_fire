"""Before/after helper diagnostics; allocation profiling is outside timed workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, 'tools')
from benchmark_backends import _memory

# The reused helper module adds the checkout to sys.path. Restore the explicitly
# selected immutable package before importing any project code in a worker.
if 'BEHAVE_PROBE_SITE' in os.environ:
    sys.path.insert(0, os.environ['BEHAVE_PROBE_SITE'])

def _bytes(value):
    """Count actual result payload storage outside the measured call.

    :param value: Scientific result or nested result mapping.
    :return: Array and mask payload bytes; scalar None has zero storage.
    """
    if isinstance(value, dict):
        return sum(_bytes(item) for item in value.values())
    if value is None:
        return 0
    array = np.asanyarray(value)
    return array.nbytes + (np.ma.getmaskarray(array).nbytes if np.ma.isMaskedArray(array) else 0)


def _cases(count):
    """Construct repeatable calls spanning every component family.

    :param count: Requested vector cell count; scalar state tools run only at one.
    :return: Ordered family names mapped to prepared public calls.
    """
    from behave_fire import rust as r

    values = np.linspace(1., 2., count, dtype=np.float64)
    f32 = values.astype(np.float32)
    indices = np.zeros(count, dtype=np.int32)
    coefficients = r.mortality.build_mortality_lookup()
    cases = {
        'surface.geometry': lambda: r.surface.calculate_eccentricity(f32),
        'surface.direction': lambda: r.surface.calculate_direction_of_max_spread(f32, f32 * 0.5, f32 * 2., 'RelativeToUpslope', 0.),
        'crown.component': lambda: r.crown.calculate_crown_heat_and_intensity(1000., values * 5., 3., 30., 0.02),
        'units': lambda: r.behave_units.LengthUnits.toBaseUnits(values, 1),
        'mortality': lambda: r.mortality.calculate_crown_scorch_mortality(values * 20., 80., 0.5, 12., indices + 1, coefficients),
        'vpd': lambda: r.vapor_pressure_deficit_calculator.calculate_vpd(values + 25., 0, 0.4, 0),
        'fdfm': lambda: r.fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture(indices, 0, 0, 0, 0, 0, 0, 0),
        'containment': lambda: r.contain.run_contain_sim_array(values * 5., 2., 1., 120., values, max_steps=32),
        'ignite': lambda: r.ignite.calculate_firebrand_ignition_probability(values + 70., 0.5, 0.06),
        'safety': lambda: r.safety.calculate_safety_zone(values * 10., 3., 50., 1., 300.),
        'spot': lambda: r.spot.calculate_spotting_from_surface_fire(values * 5., 5., 20.),
    }
    if count == 1:
        size = r.FireSize()
        size.calculate_fire_basic_dimensions(False, 5., 3, 10., 0)
        cases['fire_size.scalar'] = lambda: size.get_fire_area(False, 0, 10., 0)
        slope = r.SlopeTool()
        cases['slope.scalar'] = lambda: slope.calculate_horizontal_distance(100., 0, 10., 0)
    return cases


def _worker(count, profile):
    """Measure one explicitly selected immutable package in a fresh process.

    :param count: Requested cell count.
    :param profile: Whether this worker collects allocations instead of timings.
    :return: Timing or allocation rows, memory diagnostics, and environment identity.
    """
    from behave_fire import _rust

    expected = Path(os.environ['BEHAVE_PROBE_SITE']).resolve() / 'behave_fire'
    assert Path(_rust.__file__).resolve().is_relative_to(expected), _rust.__file__
    assert hasattr(_rust, '_allocation_start') == profile

    rows = []
    for name, call in _cases(count).items():
        before, peak_before = _memory()
        if profile:
            _rust._allocation_start()
            value = call()
            allocations = _rust._allocation_stop()
            elapsed = []
        else:
            allocations = None
            elapsed = []
            for _ in range(4):
                start = time.perf_counter_ns()
                value = call()
                elapsed.append((time.perf_counter_ns() - start) / 1e9)
        after, peak_after = _memory()
        rows.append(dict(family=name, cells=count, first_seconds=elapsed[0] if elapsed else None,
                         warm_seconds=elapsed[1:], output_bytes=_bytes(value), allocations=allocations,
                         before_rss=before, after_rss=after, lifetime_peak_before=peak_before,
                         lifetime_peak_after=peak_after))
    return dict(rows=rows, python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                extension_sha256=hashlib.sha256(Path(_rust.__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--site')
    parser.add_argument('--probe-site')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(_worker(args.worker, args.profile), allow_nan=False))
    else:
        observations = []
        for cells in (1, 10000, 100000):
            for sample in range(3):
                selected = str(Path(args.site).resolve())
                env = dict(os.environ, PYTHONPATH=selected, BEHAVE_PROBE_SITE=selected)
                record = json.loads(subprocess.check_output([sys.executable, __file__, '--worker', str(cells)], text=True, env=env))
                record.update(sample=sample, profile=False)
                observations.append(record)
                Path(args.output).write_text(json.dumps(dict(status='partial', observations=observations), indent=2) + '\n', encoding='utf-8')
                print(f'timed cells={cells} sample={sample} families={len(record["rows"])}', flush=True)
            selected = str(Path(args.probe_site).resolve())
            env = dict(os.environ, PYTHONPATH=selected, BEHAVE_PROBE_SITE=selected)
            record = json.loads(subprocess.check_output([sys.executable, __file__, '--worker', str(cells), '--profile'], text=True, env=env))
            record.update(profile=True)
            observations.append(record)
            Path(args.output).write_text(json.dumps(dict(status='partial', observations=observations), indent=2) + '\n', encoding='utf-8')
            print(f'profile cells={cells} families={len(record["rows"])}', flush=True)
        report = dict(protocol='Three fresh timed processes per size; first plus three warm calls. Separate allocation-only process per size. Serial workers. Scalar-only state tools at one cell; vector geometry also measured.',
                      observations=observations)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
