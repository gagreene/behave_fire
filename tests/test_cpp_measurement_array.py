"""Collect new evidence without granting stale-source acceptance or requiring native Python runs."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_cpp_reference_array import _snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import cpp_reference as reference

pytestmark = pytest.mark.cpp_reference


def _stale_snapshot(tmp_path):
    root = _snapshot(tmp_path)
    path = root / 'tests/data/cpp_reference/tolerance-policy.json'
    policy = reference._document(path)
    policy['approved'] = True
    policy['qualification'] = reference._document(root / 'tests/data/cpp_reference/calibration/qualification.json')
    reference._json(path, policy)
    hashes_path = root / 'tests/data/cpp_reference/hashes.json'
    hashes = reference._document(hashes_path)
    hashes['tests/data/cpp_reference/tolerance-policy.json'] = reference._digest(path)
    reference._json(hashes_path, hashes)
    source = root / 'src/behave_fire/components/surface.py'
    source.write_bytes(source.read_bytes() + b'\n# Harmless source change needing replacement qualification.\n')
    return root


def test_compiler_observation_validation_does_not_require_previous_source_qualification(tmp_path):
    import cpp_calibrate

    root = _stale_snapshot(tmp_path)
    rows = [*reference._rows(root / 'tests/data/cpp_reference/surface/results.jsonl'),
            *reference._rows(root / 'tests/data/cpp_reference/surface_custom/results.jsonl')]
    protocol = dict(negative_cases=62, negative_upstream_entries=0, upstream_trap_exit=86,
                    geometry_trap_exit=87, duplicate_ids_rejected=True, state_sequence_checked=True,
                    custom_negative_cases=18, custom_negative_upstream_entries=0,
                    custom_upstream_trap_exit=86, custom_geometry_trap_exit=87,
                    custom_duplicate_ids_rejected=True, custom_state_sequence_checked=True)
    report = dict(kind='cpp', observations=[dict(id=row['id'], actual=row) for row in rows], protocol=protocol)
    assert cpp_calibrate._validate_observations(root, report) == dict(cases=773, available=722, fields=3610, unavailable=51)
    with pytest.raises(ValueError, match='source'):
        reference.load_reference(root)


@pytest.mark.parametrize('source_key', ['driver_source_sha256_lf', 'driver_source_sha256'])
def test_generation_accepts_current_and_legacy_receipts_before_preservation_guard(tmp_path, source_key):
    driver = tmp_path / 'driver.bin'
    driver.write_bytes(b'synthetic build artifact')
    overlay = tmp_path / 'reference/behave_cpp_overlay/driver.cpp'
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b'// synthetic source\n')
    receipt = dict(upstream_sha=reference.PIN, driver_sha256=reference._digest(driver),
                   **{source_key: reference._text_digest(overlay)})
    attestation = tmp_path / 'receipt.json'
    reference._json(attestation, receipt)
    cases = tmp_path / 'tests/data/cpp_reference/surface/cases.jsonl'
    cases.parent.mkdir(parents=True)
    cases.write_bytes(b'protected observation input\n')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        reference.generate_reference(tmp_path, driver, attestation, 'Test receipt compatibility')
    assert cases.read_bytes() == b'protected observation input\n'


def test_python_only_comparison_never_imports_the_native_extension(tmp_path):
    script = '''
import pathlib, sys
root, report = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(root / 'src'), str(root / 'tools')]
class BlockNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'behave_fire._rust':
            raise AssertionError('Native extension import during Python-only comparison')
sys.meta_path.insert(0, BlockNative())
import cpp_controls
data = cpp_controls.measure_ports(root, report, backends=('python',))
assert data['counts'] == {'python': dict(cases=583, fields=2915, paired_cases=139, original_unit_difference_cases=139)}
assert 'extension_sha256' not in data and 'binding_api_version' not in data
assert data['parity_approved'] is False
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT), str(tmp_path / 'python.json')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.rust
def test_replacement_port_measurement_keeps_stale_acceptance_closed(tmp_path):
    from behave_fire import _rust

    root = _stale_snapshot(tmp_path)
    binary = Path(_rust.__file__)
    shutil.copyfile(binary, root / 'src/behave_fire' / binary.name)
    libraries = ROOT / 'src/behave_fire.libs'
    if libraries.is_dir():
        shutil.copytree(libraries, root / 'src/behave_fire.libs')
    script = '''
import pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(root / 'src'), str(root / 'tools')]
import cpp_calibrate, cpp_controls, cpp_reference
try:
    cpp_reference.load_reference(root)
except ValueError as error:
    assert 'source' in str(error)
else:
    raise AssertionError('Stale qualification accepted')
report = cpp_controls.measure_ports(root, output)
assert report['parity_approved'] is False
assert cpp_calibrate._validate_observations(root, report)['cases_per_backend'] == 583
assert cpp_reference.compare_backends(root, output.parent / 'raw-differences')['policy_approved'] is False
data = cpp_reference.load_reference(root, measurement=True)
assert data['policy']['approved'] is False
row = next(row['output'] for row in data['results'] if row['status'] == 'ok')
try:
    cpp_reference.assert_matches_cpp(row, row, data['policy'])
except cpp_reference.UnapprovedCppPolicy:
    pass
else:
    raise AssertionError('Observation-only view certified acceptance')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root), str(tmp_path / 'ports.json')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
