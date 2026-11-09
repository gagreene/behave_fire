"""Final performance evidence proves the actual contract, inputs, and complete outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_performance_harness_array import _evidence, _tool


def _current_evidence(tmp_path):
    _tool('performance_contract')
    from performance_contract import requested_outputs, save_outputs

    report, requirements = _evidence()
    requirements['binding_api_version'] = 4
    requirements['require_allocations'] = False
    for row in report['records']:
        identity = json.dumps([row[name] for name in ('dataset', 'operation', 'cells')]).encode()
        row.update(binding_api_version=4, actual_backend=row['backend'],
                   python_timing_original_inputs=row['backend'] == 'python',
                   input_sha256=hashlib.sha256(identity).hexdigest(),
                   requested_outputs=requested_outputs(row['operation']),
                   eligible_cells=row['cells'], eligibility={'kind': 'all'})
    for index, row in enumerate(report['validations']):
        timing = next(r for r in report['records'] if r['backend'] == 'rust'
                      and all(r[k] == row[k] for k in ('dataset', 'operation', 'cells', 'threads')))
        fields = requested_outputs(row['operation'])
        actual = {name: np.zeros(1, dtype=np.int32 if name == 'fire_type' else np.float64) for name in fields}
        row.update(binding_api_version=4, actual_backend='rust', input_sha256=timing['input_sha256'],
                   requested_outputs=fields, fields=fields, compared_fields=fields,
                   output_fields=len(fields), correctness_contract='f64-nan-v1', independent_python=True,
                   eligible_cells=row['cells'], eligibility={'kind': 'all'},
                   output_artifact=save_outputs(tmp_path/f'{index}.npz', actual, actual, root=tmp_path))
    return report, requirements


def test_complete_artifacts_and_original_public_baselines_pass(tmp_path):
    report, requirements = _current_evidence(tmp_path)
    result = _tool('check_performance').check_report(report, requirements, artifact_root=tmp_path)
    assert result['passed'], result['errors']


def test_current_allocation_claim_requires_versioned_dual_build_manifest():
    report, requirements = _evidence()
    requirements['binding_api_version'] = 4
    errors = _tool('check_performance')._allocation_errors(report, requirements, ('a'*64, 'b'*64))
    assert errors


@pytest.mark.rust
@pytest.mark.parametrize('operation', ['surface', 'crown', 'typed-crown', 'typed-crown-b'])
@pytest.mark.parametrize('dataset', ['uniform', 'heterogeneous'])
def test_explicit_workers_share_original_inputs_and_report_their_actual_backend(operation, dataset):
    driver = _tool('benchmark_backends')
    args = SimpleNamespace(cells=16, dataset=dataset, operation=operation, seed=20261105, threads=1)
    python = driver._model_call(args, 'python').evidence
    native = driver._model_call(args, 'rust').evidence
    assert python['actual_backend'] == 'python' and native['actual_backend'] == 'rust'
    assert python['input_sha256'] == native['input_sha256']
    assert python['requested_outputs'] == native['requested_outputs']
    assert python['python_timing_original_inputs']


@pytest.mark.parametrize('fault', ['fallback', 'normalized-timing', 'missing-comparison', 'request-set',
                                   'category', 'input-identity', 'api', 'archive', 'eligibility'])
def test_incomplete_or_misrepresented_current_contract_cannot_certify(tmp_path, fault):
    report, requirements = _current_evidence(tmp_path)
    row = next(r for r in report['records'] if r['backend'] == 'rust')
    validation = next(r for r in report['validations'] if r['operation'] == 'crown')
    if fault == 'fallback':
        row['actual_backend'] = 'python'
    elif fault == 'normalized-timing':
        next(r for r in report['records'] if r['backend'] == 'python')['python_timing_original_inputs'] = False
    elif fault == 'missing-comparison':
        validation['compared_fields'] = []
    elif fault == 'request-set':
        row['requested_outputs'] = ['spread_rate']
    elif fault == 'category':
        validation['fields'] = [f for f in validation['fields'] if f != 'fire_type']
    elif fault == 'input-identity':
        row['input_sha256'] = 'f'*64
    elif fault == 'api':
        row['binding_api_version'] = 3
    elif fault == 'archive':
        (tmp_path/validation['output_artifact']['path']).write_bytes(b'not measured outputs')
    elif fault == 'eligibility':
        row['eligible_cells'] = 0
    assert not _tool('check_performance').check_report(report, requirements, artifact_root=tmp_path)['passed']


def test_original_input_identity_preserves_dtype_mask_payload_and_values():
    _tool('performance_contract')
    from performance_contract import input_identity

    value = np.ma.array([1., 2.], mask=[False, True])
    original = input_identity({'x': value})
    assert original != input_identity({'x': value.filled(np.nan)})
    assert original != input_identity({'x': value.astype(np.float32)})
    value.data[1] = 9.
    assert original != input_identity({'x': value})


@pytest.mark.rust
def test_profile_keeps_separate_kernel_encoder_and_transfer_measurements():
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run([sys.executable, '-B', 'tools/benchmark_backends.py', '--worker', '--backend',
        'rust', '--cells', '32', '--operation', 'crown', '--warm-calls', '0', '--profile'],
        cwd=root, env=dict(os.environ, PYTHONPATH=str(root/'src')), capture_output=True, text=True, check=True)
    row = json.loads(process.stdout)
    assert row['profile'] is True and len(row['native_calls']) == 1
    assert row['native_stages']['kernel_seconds'] > 0
    assert row['native_stages']['encoder_seconds'] > 0
    assert row['native_stages']['appended_bytes'] == row['native_stages']['concatenate_seconds'] == 0


@pytest.mark.rust
def test_real_validation_worker_saves_every_actual_and_independent_array(tmp_path, monkeypatch):
    driver = _tool('benchmark_backends')
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, '_provenance', lambda args: {})
    arguments = SimpleNamespace(cells=16, dataset='heterogeneous', seed=20261105,
                                operation='typed-crown-b', threads=4, artifacts=tmp_path/'full.npz')
    row = driver._validation_worker(arguments)
    assert row['passed'] and row['binding_api_version'] == 4 and row['actual_backend'] == 'rust'
    assert row['compared_fields'] == row['requested_outputs'] == row['fields']
    with np.load(tmp_path/row['output_artifact']['path'], allow_pickle=False) as archive:
        assert set(archive.files) == {prefix+name for prefix in ('actual__', 'expected__') for name in row['fields']}


@pytest.mark.parametrize('initialized', [True, False])
def test_skipped_cells_require_independent_validity_and_initialized_outputs(tmp_path, initialized):
    contract = _tool('performance_contract')
    fields = contract.requested_outputs('surface')
    outputs = {name: np.array([1., np.nan if initialized else 0., 2.]) for name in fields}
    row = dict(dataset='uniform', operation='surface', cells=3, binding_api_version=4,
               actual_backend='rust', input_sha256='c'*64, requested_outputs=fields, fields=fields,
               compared_fields=fields, output_fields=len(fields), correctness_contract='f64-nan-v1',
               independent_python=True, eligible_cells=2, eligibility={'kind': 'explicit', 'validity': [True, False, True]},
               output_artifact=contract.save_outputs(tmp_path/'skip.npz', outputs, outputs, root=tmp_path))
    errors = contract.contract_errors({'records': [], 'validations': [row]}, {'binding_api_version': 4}, tmp_path)
    assert bool(errors) is not initialized


def test_skipped_scientific_scope_cannot_be_faked_as_a_complete_dense_loop():
    assembler = _tool('assemble_allocation_evidence')
    row = dict(cells=3, workers=1, binding_api_version=4, eligible_cells=2,
               eligibility={'kind': 'explicit', 'validity': [True, False, True]},
               scopes={'kernel': dict(requests=0, bytes=0, calls=2, workers=1, definition='kernel-loop-v1')})
    assert assembler._scope(row, 'kernel')['calls'] == 2
    row['eligibility'] = {'kind': 'all'}
    with pytest.raises(ValueError):
        assembler._scope(row, 'kernel')
