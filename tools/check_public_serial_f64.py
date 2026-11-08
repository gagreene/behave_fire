"""Check frozen public serial f64 campaigns against unchanged owner targets."""
import argparse
import json
import math
from pathlib import Path
import statistics

from check_performance import _bootstrap

OPERATIONS = ('surface', 'crown', 'typed-crown', 'typed-crown-b', 'masked-crown',
              'masked-false-crown', 'typed-crown-c')


def _target(operation, cells):
    """Choose the approved route/size target without changing missed thresholds.

    :param operation: Public raster route.
    :param cells: Logical cell count.
    :return: Required lower confidence bound for Python time divided by Rust time.
    """
    return 2. if cells == 100000 else 1. if operation.startswith('typed-') else 1.25


def check_public_screen(report):
    """Reject partial, mixed-source, normalized-timing, and replayed sample evidence.

    :param report: Frozen public screen or independent confirmation campaign.
    :return: All endpoints and errors, with an explicit serial-only certification scope.
    """
    requirements = json.loads(Path(__file__).with_name('performance_requirements.json').read_text(encoding='utf-8'))
    errors, endpoints = [], []
    if (report.get('schema_version') != 1 or report.get('contract_version') != 4
            or report.get('scope') != 'public-serial-f64' or report.get('phase') not in ('screen', 'confirmation')):
        errors.append('Wrong public f64 campaign schema/scope/phase')
    if report.get('errors'):
        errors.append('Campaign contains worker or validation failures')
    for key in ('source_hash', 'extension_hash', 'harness_hash'):
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            errors.append(f'Missing/invalid {key}')
    pairs, warm = (7, 5) if report.get('phase') == 'confirmation' else (3, 4)
    expected = {(d, o, n) for d in ('uniform', 'heterogeneous') for o in OPERATIONS for n in (10000, 100000)}
    keys = [(c.get('dataset'), c.get('operation'), c.get('cells')) for c in report.get('cases', [])]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        errors.append('Missing/duplicate/unexpected public cases')
    for case in report.get('cases', []):
        start = len(errors)
        key = (case.get('dataset'), case.get('operation'), case.get('cells'))
        validation = case.get('validation', {})
        if (validation.get('passed') is not True or validation.get('independent_python') is not True
                or validation.get('correctness_contract') != 'f64-nan-v1'
                or sorted(validation.get('fields', [])) != output_fields(case.get('operation'))):
            errors.append(f'{key}: missing complete independent normalized/NaN output validation')
        for name in ('source_hash', 'extension_hash'):
            if validation.get(name) != report.get(name):
                errors.append(f'{key}: wrong validation {name}')
        samples = {}
        for backend in ('python', 'rust'):
            rows = sorted((r for r in case.get('records', []) if r.get('backend') == backend),
                          key=lambda r: r.get('block', -1))
            if len(rows) != pairs or [r.get('block') for r in rows] != list(range(pairs)):
                errors.append(f'{key}: missing/duplicate/pooled {backend} process blocks')
            for row in rows:
                if row.get('actual_backend') != backend or row.get('profile') is not False or row.get('threads') != 1:
                    errors.append(f'{key}: wrong actual backend, profiling, or thread count')
                if backend == 'python' and row.get('python_timing_original_inputs') is not True:
                    errors.append(f'{key}: normalized oracle used as Python timing baseline')
                for name in ('source_hash', 'extension_hash'):
                    if row.get(name) != report.get(name):
                        errors.append(f'{key}: mixed {name}')
                if row.get('numpy') != report.get('environment', {}).get('numpy'):
                    errors.append(f'{key}: changed NumPy timing baseline')
                values = [row.get('first_seconds')] + row.get('warm_seconds', [])
                if len(values) != warm + 1 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                    errors.append(f'{key}: invalid first/warm durations')
            samples[backend] = rows
        if len(errors) != start:
            continue
        for phase in ('first', 'warm'):
            values = {backend: [row['first_seconds'] if phase == 'first' else statistics.median(row['warm_seconds'])
                                for row in rows] for backend, rows in samples.items()}
            endpoint = dict(dataset=key[0], operation=key[1], cells=key[2], phase=phase,
                target=_target(key[1], key[2]), **_bootstrap(values['python'], values['rust'], requirements))
            endpoints.append(endpoint)
            if endpoint['lower'] < endpoint['target']:
                errors.append(f"{key}/{phase}: lower95% {endpoint['lower']:.3f}x below {endpoint['target']:.3f}x")
    return dict(passed=not errors, scope='public-serial-only', phase=report.get('phase'),
                public_serial_certified=not errors and report.get('phase') == 'confirmation',
                endpoints=endpoints, errors=errors)


def main():
    """Assess a saved immutable public campaign.

    :return: None; prints the complete assessment and exits nonzero on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    result = check_public_screen(json.loads(args.results.read_text(encoding='utf-8')))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


def output_fields(operation):
    """Return the complete public output inventory for one measured model.

    :param operation: Surface or one of the Crown input-layout routes.
    :return: Sorted complete field names, including categorical fire_type.
    """
    surface = ('spread_rate', 'backing_spread_rate', 'flanking_spread_rate', 'flame_length',
        'fireline_intensity', 'heat_per_unit_area', 'effective_wind_speed', 'fire_length_to_width_ratio',
        'eccentricity', 'direction_of_max_spread', 'residence_time', 'reaction_intensity',
        'midflame_wind_speed', 'no_wind_no_slope_spread_rate')
    crown = ('crown_fire_spread_rate', 'crown_flame_length', 'crown_fire_line_intensity',
        'crown_fire_heat_per_unit_area', 'canopy_heat_per_unit_area',
        'crown_critical_surface_fire_line_intensity', 'crown_critical_fire_spread_rate',
        'crown_fire_transition_ratio', 'crown_fire_active_ratio', 'crown_length_to_width_ratio',
        'fire_type', 'surface_fire_critical_spread_rate', 'crowning_surface_fire_spread_rate',
        'crown_fraction_burned', 'passive_crown_fire_spread_rate', 'passive_crown_fire_heat_per_unit_area',
        'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length', 'final_spread_rate',
        'final_heat_per_unit_area', 'final_fireline_intensity', 'final_flame_length')
    return sorted(surface if operation == 'surface' else crown)


if __name__ == '__main__':
    main()
