"""Assess bounded f64 core headroom, never end-to-end Python speed certification."""
import argparse
import json
import math
from pathlib import Path
import statistics

from check_performance import _bootstrap


def _check_case(case, report, requirements, pairs, warm_calls):
    """Validate one full-output case and compute its paired speed endpoints.

    :param case: Dataset, operation, cell count, samples, and independent validation.
    :param report: Campaign source and native binary identity.
    :param requirements: Unchanged public targets and bootstrap configuration.
    :param pairs: Required independent block count.
    :param warm_calls: Required warmed observations per process.
    :return: Case endpoint records and errors, retaining any missed target.
    """
    key = f"{case.get('dataset')}/{case.get('operation')}/{case.get('cells')}"
    errors = []
    validation = case.get('validation', {})
    if (any(validation.get(name) is not True for name in
            ('complete', 'independent_python', 'all_fields', 'typed_codes'))
            or validation.get('truth_contract') != 'f64-nan-v1'):
        errors.append(f'{key}: missing independent complete f64/NoData validation')
    samples = {}
    for backend in ('python', 'core-rust'):
        rows = sorted((r for r in case.get('records', []) if r.get('backend') == backend),
                      key=lambda r: r.get('block', -1))
        blocks = [r.get('block') for r in rows]
        if len(rows) != pairs or blocks != list(range(pairs)):
            errors.append(f'{key}: missing or duplicate {backend} process blocks')
        for row in rows:
            if row.get('profile') is not False or row.get('source_hash') != report.get('source_hash'):
                errors.append(f'{key}: profiled or mixed-source sample')
            if backend == 'core-rust' and row.get('binary_hash') != report.get('binary_hash'):
                errors.append(f'{key}: wrong core binary identity')
            if backend == 'python' and row.get('python_timing_original_inputs') is not True:
                errors.append(f'{key}: normalized correctness oracle used as Python timing baseline')
            values = [row.get('first_seconds')] + row.get('warm_seconds', [])
            if (len(values) != warm_calls + 1 or any(type(v) not in (int, float)
                    or not math.isfinite(v) or v <= 0 for v in values)):
                errors.append(f'{key}: invalid or missing first/warm durations')
        samples[backend] = rows
    if len(samples['python']) != len(samples['core-rust']):
        errors.append(f'{key}: unpaired process counts')
    endpoints = []
    if not errors:
        target = _target(case['operation'], case['cells'], requirements)
        for phase in ('first', 'warm'):
            values = {backend: [r['first_seconds'] if phase == 'first' else statistics.median(r['warm_seconds'])
                               for r in rows] for backend, rows in samples.items()}
            endpoint = dict(dataset=case['dataset'], operation=case['operation'], cells=case['cells'],
                            phase=phase, target=target,
                            **_bootstrap(values['python'], values['core-rust'], requirements))
            endpoints.append(endpoint)
            if endpoint['lower'] < target:
                errors.append(f"{key}/{phase}: core lower95% {endpoint['lower']:.3f}x below {target:.3f}x")
    return endpoints, errors


def _target(operation, cells, requirements):
    """Leave measured headroom above each owner-approved public target.

    :param operation: Common or mixed supplied-input workload.
    :param cells: Logical raster cell count.
    :param requirements: Owner-approved public target values.
    :return: Core target, exactly 1.25 times the public serial target.
    """
    name = ('small_typed' if operation.startswith('typed-') else 'small_common') if cells == 10000 else 'large_serial'
    return 1.25 * requirements['targets'][name]


def check_core_screen(report, requirements=None):
    """Check one independent core campaign without promoting it to public speed evidence.

    :param report: Parsed raw core/public-Python pairs and output validation.
    :param requirements: Optional unchanged target/bootstrap configuration.
    :return: Pass/fail, all endpoints/errors, and an explicit core-only scope marker.
    """
    if requirements is None:
        requirements = json.loads(Path(__file__).with_name('performance_requirements.json').read_text(encoding='utf-8'))
    errors = []
    if report.get('schema_version') != 1 or report.get('contract_version') != 4:
        errors.append('Missing f64 contract/campaign schema')
    if report.get('scope') != 'f64-raster-core-with-storage' or report.get('phase') not in ('screen', 'confirmation'):
        errors.append('Wrong measurement scope or phase')
    if report.get('errors'):
        errors.append('Campaign has failed workers or validations')
    for name in ('source_hash', 'binary_hash'):
        value = report.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            errors.append(f'Missing/invalid {name}')
    pairs, warm_calls = (7, 5) if report.get('phase') == 'confirmation' else (3, 4)
    expected = {(d, o, n) for d in ('uniform', 'heterogeneous')
                for o in ('surface', 'crown', 'typed-crown', 'typed-crown-b') for n in (10000, 100000)}
    keys = [(c.get('dataset'), c.get('operation'), c.get('cells')) for c in report.get('cases', [])]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        errors.append('Missing, duplicate, or unexpected required cases')
    endpoints = []
    for case in report.get('cases', []):
        result, problems = _check_case(case, report, requirements, pairs, warm_calls)
        endpoints.extend(result)
        errors.extend(problems)
    return dict(passed=not errors, scope='core-only', public_speed_certified=False,
                phase=report.get('phase'), endpoints=endpoints, errors=errors)


def main():
    """Read a frozen campaign and exit nonzero when its core gate fails.

    :return: None; prints explicit core-only assessment JSON and sets the exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    result = check_core_screen(json.loads(args.results.read_text(encoding='utf-8')))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
