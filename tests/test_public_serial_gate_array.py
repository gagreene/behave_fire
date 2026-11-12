"""Public speed evidence cannot substitute normalized timing or omit outputs."""
from copy import deepcopy

import pytest
from parity_support import ROOT


def _report():
    from check_public_serial_f64 import OPERATIONS, output_fields

    report = dict(schema_version=1, contract_version=4, scope='public-serial-f64', phase='screen',
                  source_hash='a' * 64, extension_hash='b' * 64, harness_hash='c' * 64,
                  environment=dict(numpy='2.1.3'),
                  errors=[], cases=[])
    for dataset in ('uniform', 'heterogeneous'):
        for operation in OPERATIONS:
            for cells in (10000, 100000):
                records = []
                for backend in ('python', 'rust'):
                    for block in range(3):
                        records.append(dict(backend=backend, actual_backend=backend, block=block,
                            source_hash=report['source_hash'], extension_hash=report['extension_hash'],
                            numpy='2.1.3', threads=1, profile=False,
                            python_timing_original_inputs=backend == 'python',
                            first_seconds=10. if backend == 'python' else 1.,
                            warm_seconds=[10. if backend == 'python' else 1.] * 4))
                report['cases'].append(dict(dataset=dataset, operation=operation, cells=cells, records=records,
                    validation=dict(passed=True, independent_python=True, correctness_contract='f64-nan-v1',
                                    source_hash=report['source_hash'], extension_hash=report['extension_hash'],
                                    fields=output_fields(operation))))
    return report


def test_complete_public_evidence_passes_unchanged_targets():
    from check_public_serial_f64 import check_public_screen

    result = check_public_screen(_report())
    assert result['passed']
    assert len(result['endpoints']) == 56


@pytest.mark.parametrize('problem', ['failed', 'changed', 'reused'])
def test_confirmation_reassesses_screen_and_rejects_changed_identity(problem):
    from benchmark_public_serial_f64 import validate_confirmation

    screen = _report()
    identity = {name: deepcopy(screen[name]) for name in ('source_hash', 'extension_hash', 'harness_hash', 'environment')}
    screen['assessment'] = {'passed': True}
    if problem == 'failed':
        screen['cases'][0]['validation']['passed'] = False
    elif problem == 'changed':
        identity['source_hash'] = 'd' * 64
    else:
        screen['phase'] = 'confirmation'
    with pytest.raises(ValueError):
        validate_confirmation(screen, identity)


@pytest.mark.parametrize('problem', ['normalized-timing', 'missing-output', 'wrong-backend', 'pooled-blocks'])
def test_public_gate_rejects_invalid_evidence(problem):
    from check_public_serial_f64 import check_public_screen

    report = _report()
    case = report['cases'][0]
    if problem == 'normalized-timing':
        case['records'][0]['python_timing_original_inputs'] = False
    elif problem == 'missing-output':
        case['validation']['fields'] = case['validation']['fields'][:-1]
    elif problem == 'wrong-backend':
        case['records'][-1]['actual_backend'] = 'python'
    else:
        case['records'].append(deepcopy(case['records'][0]))
    assert not check_public_screen(report)['passed']
