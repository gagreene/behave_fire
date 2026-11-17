"""Conditional C++ bounds must not become active from an approval bit alone."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from test_cpp_reference_array import _snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import cpp_reference as reference


def test_active_policy_requires_complete_calibration_metadata(tmp_path):
    root = _snapshot(tmp_path)
    relative = 'tests/data/cpp_reference/tolerance-policy.json'
    path = root / relative
    policy = json.loads(path.read_text())
    policy['approved'] = True
    policy.pop('qualification', None)
    reference._json(path, policy)
    hash_path = root / 'tests/data/cpp_reference/hashes.json'
    hashes = json.loads(hash_path.read_text())
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference._json(hash_path, hashes)
    with pytest.raises(ValueError, match='calibration'):
        reference.load_reference(root)


def test_empty_or_partial_evidence_cannot_qualify_the_matrix(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    for paths in ([], [tmp_path / 'missing.json']):
        with pytest.raises(ValueError, match='calibration'):
            module.validate_evidence(ROOT, paths)


def test_reducing_the_required_matrix_cannot_activate_a_partial_record(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    controls = importlib.import_module('cpp_controls')
    root = _snapshot(tmp_path)
    policy_path = root / 'tests/data/cpp_reference/tolerance-policy.json'
    policy = reference._document(policy_path)
    policy['activation_requirements'] = {'compilers': ['MSVC'], 'platforms': ['windows-x86_64']}
    reference._json(policy_path, policy)
    record = dict(calibration_complete=True, upstream_sha=reference.PIN,
                  artifacts=controls._artifacts(root), fields=controls.calibration_policy(root),
                  overlay_sha256=reference._prepared_build(root)['prepared_overlay_sha256'],
                  source_commit='0' * 40, ci_run_url='https://github.com/gagreene/behave_fire/actions/runs/1',
                  reports={
                      'cpp-MSVC': dict(sha256='1' * 64, summary=dict(cases=773, available=722, fields=3610, unavailable=51)),
                      'ports-windows-x86_64': dict(sha256='2' * 64, summary=dict(cases_per_backend=583,
                          fields_per_backend=2915, paired_cases_per_backend=139))})
    with pytest.raises(ValueError, match='calibration'):
        module.validate_qualification(root, record)


@pytest.mark.parametrize('mutation', ['change', 'add', 'delete'])
def test_source_changes_invalidate_otherwise_complete_qualification(tmp_path, mutation):
    module = importlib.import_module('cpp_calibrate')
    controls = importlib.import_module('cpp_controls')
    root = _snapshot(tmp_path)
    source = root / 'src/behave_fire/science.py'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('VALUE = 1\n')
    keys = ['cpp-GNU', 'cpp-MSVC', 'ports-linux-x86_64', 'ports-linux-aarch64',
            'ports-linux-musl-x86_64', 'ports-macos-x86_64', 'ports-macos-arm64',
            'ports-windows-x86_64', 'ports-windows-arm64']
    record = dict(calibration_complete=True, upstream_sha=reference.PIN,
                  artifacts=controls._artifacts(root), fields=controls.calibration_policy(root),
                  overlay_sha256=reference._prepared_build(root)['prepared_overlay_sha256'],
                  source_commit='0' * 40, ci_run_url='https://github.com/gagreene/behave_fire/actions/runs/1',
                  source_files=(module._sources(root) if hasattr(module, '_sources') else {}),
                  reports={key: dict(sha256='1' * 64, summary=(dict(cases=773, available=722,
                      fields=3610, unavailable=51) if key.startswith('cpp-') else dict(
                      cases_per_backend=583, fields_per_backend=2915, paired_cases_per_backend=139))) for key in keys})
    module.validate_qualification(root, record)
    policy_path = root / 'tests/data/cpp_reference/tolerance-policy.json'
    policy = reference._document(policy_path)
    policy['approval_gate'] = 'Metadata-only activation does not change tested implementation'
    reference._json(policy_path, policy)
    module.validate_qualification(root, record)
    if mutation == 'change':
        source.write_text('VALUE = 2\n')
    elif mutation == 'add':
        source.with_name('extra.py').write_text('VALUE = 1\n')
    else:
        source.unlink()
    with pytest.raises(ValueError, match='source'):
        module.validate_qualification(root, record)


@pytest.mark.rust
def test_unit_pair_revalidation_preserves_signed_zero(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    controls = importlib.import_module('cpp_controls')
    report = controls.measure_ports(ROOT, tmp_path / 'ports.json')
    module._validate_observations(ROOT, report)
    row, field = next((row, field) for row in report['documented_differences']
                      for field, value in row['numbered_port'].items() if value == 0.)
    row['numbered_port'][field] = -0.
    with pytest.raises(ValueError, match='Numbered/custom'):
        module._validate_observations(ROOT, report)
