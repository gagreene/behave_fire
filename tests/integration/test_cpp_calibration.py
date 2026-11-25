"""Conditional C++ bounds must not become active from an approval bit alone."""

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

from cpp_snapshot_support import write_snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import cpp_reference as reference

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ('rust/Cargo.toml', 'rust/Cargo.lock', 'rust/crates/behave-fire/Cargo.toml',
             'rust/crates/behave-py/Cargo.toml')


def _copy_manifests(tmp_path):
    root = tmp_path / 'root'
    for relative in MANIFESTS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes().replace(b'\r\n', b'\n'))
    return root


def _edit(root, relative, old, new, count=1):
    """Replace text in a copied manifest; ``<core>`` and ``<py>`` stand for the current versions."""
    import tomllib
    core = tomllib.loads((root / 'rust/crates/behave-fire/Cargo.toml').read_text())['package']['version']
    py = tomllib.loads((root / 'rust/crates/behave-py/Cargo.toml').read_text())['package']['version']
    old, new = (value.replace('<core>', core).replace('<py>', py) for value in (old, new))
    path = root / relative
    text = path.read_bytes().decode()
    assert text.count(old) == count, (relative, old)
    path.write_bytes(text.replace(old, new).encode())


# ---- qualification evidence tests ----

@pytest.mark.parametrize("mutation", ["missing", "embedded_numeric_type"])
def test_active_policy_requires_complete_calibration_metadata(tmp_path, cpp_snapshot_template, mutation):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    relative = 'tests/data/cpp_reference/tolerance-policy.json'
    path = root / relative
    policy = json.loads(path.read_text())
    policy['approved'] = True
    hash_path = root / 'tests/data/cpp_reference/hashes.json'
    hashes = json.loads(hash_path.read_text())
    if mutation == "missing":
        policy.pop("qualification", None)
        error = "calibration"
    else:
        record_name = "tests/data/cpp_reference/calibration/qualification.json"
        record_path = root / record_name
        record = reference._document(record_path)
        record["fire_size"] = {"comparisons": 144}
        reference._json(record_path, record)
        hashes[record_name] = hashlib.sha256(record_path.read_bytes()).hexdigest()
        policy["qualification"] = dict(record, fire_size={"comparisons": 144.0})
        error = "Embedded calibration qualification differs"
    reference._json(path, policy)
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference._json(hash_path, hashes)
    with pytest.raises(ValueError, match=error):
        reference.load_reference(root)


def test_approved_exact_compiler_replay_rejects_numeric_drift():
    calibration = importlib.import_module('cpp_calibrate')
    ignite = importlib.import_module('cpp_ignite')
    observations = ignite.load_observations(ROOT)
    report = {'ignite': {'observations': observations}}
    policy = {'ignite': dict(approved=True, fields=ignite.EXACT_FIELDS)}
    calibration._check_approved_exact_replay(ROOT, report, policy)
    first = observations[0]
    field = 'fuel_temperature_f'
    changed = dict(first, output=dict(first['output']))
    changed['output'][field] = math.nextafter(first['output'][field], math.inf)
    report['ignite']['observations'] = [changed, *observations[1:]]
    with pytest.raises(ValueError, match=first['id']):
        calibration._check_approved_exact_replay(ROOT, report, policy)

def test_empty_or_partial_evidence_cannot_qualify_the_matrix(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    for paths in ([], [tmp_path / 'missing.json']):
        with pytest.raises(ValueError, match='calibration'):
            module.validate_evidence(ROOT, paths)


def test_reducing_the_required_matrix_cannot_activate_a_partial_record(tmp_path, cpp_snapshot_template):
    module = importlib.import_module('cpp_calibrate')
    controls = importlib.import_module('cpp_controls')
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    policy_path = root / 'tests/data/cpp_reference/tolerance-policy.json'
    policy = reference._document(policy_path)
    policy['activation_requirements'] = {'compilers': ['MSVC'], 'platforms': ['windows-x86_64']}
    reference._json(policy_path, policy)
    record = dict(calibration_complete=True, upstream_sha=reference.PIN,
                  artifacts=controls._artifacts(root), fields=controls.calibration_policy(root),
                  overlay_sha256=reference._prepared_build(root)['prepared_overlay_sha256'],
                  source_commit='0' * 40, ci_run_url='https://github.com/gagreene/behave_fire/actions/runs/1',
                  reports={
                      'cpp-MSVC': dict(sha256='1' * 64, summary=dict(cases=773, available=722, fields=3610,
                          unavailable=51)),
                      'ports-windows-x86_64': dict(sha256='2' * 64, summary=dict(cases_per_backend=710,
                          fields_per_backend=3550, paired_cases_per_backend=139))})
    with pytest.raises(ValueError, match='calibration'):
        module.validate_qualification(root, record)


@pytest.mark.parametrize('mutation', ['change', 'add', 'delete', 'missing_api', 'changed_api'])
def test_source_changes_invalidate_otherwise_complete_qualification(tmp_path, mutation, cpp_snapshot_template):
    module = importlib.import_module('cpp_calibrate')
    controls = importlib.import_module('cpp_controls')
    root = write_snapshot(cpp_snapshot_template, tmp_path)
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
                  source_files=module._sources(root),
                  public_api=dict(public_functions=1399,
                                  public_identity_sha256=reference._document(
                                      root / 'tests/data/cpp_reference/inventory.json')['public_identity_sha256']),
                  reports={key: dict(sha256='1' * 64, summary=(dict(cases=773, available=722,
                      fields=3610, unavailable=51) if key.startswith('cpp-') else dict(
                      cases_per_backend=710, fields_per_backend=3550, paired_cases_per_backend=139))) for key in keys})
    module.validate_qualification(root, record)
    policy_path = root / 'tests/data/cpp_reference/tolerance-policy.json'
    policy = reference._document(policy_path)
    policy['approval_gate'] = 'Metadata-only activation does not change tested implementation'
    reference._json(policy_path, policy)
    module.validate_qualification(root, record)
    if mutation in {'missing_api', 'changed_api'}:
        if mutation == 'missing_api':
            record.pop('public_api')
        else:
            record['public_api']['public_functions'] = 1399.0
        with pytest.raises(ValueError, match='live public API'):
            module.validate_qualification(root, record)
        return
    if mutation == 'change':
        source.write_text('VALUE = 2\n')
    elif mutation == 'add':
        source.with_name('extra.py').write_text('VALUE = 1\n')
    else:
        source.unlink()
    with pytest.raises(ValueError, match='source'):
        module.validate_qualification(root, record)


# ---- signed zero revalidation tests ----

@pytest.mark.rust
def test_revalidation_preserves_paired_bits_and_signed_zero(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    controls = importlib.import_module('cpp_controls')
    report = controls.measure_ports(ROOT, tmp_path / 'ports.json')
    module._validate_observations(ROOT, report)
    row = report['documented_differences'][0]
    field = reference.FIELDS[0]
    observed = row['numbered_port'][field]
    row['numbered_port'][field] = math.nextafter(observed, math.inf)
    with pytest.raises(ValueError, match='Numbered/custom'):
        module._validate_observations(ROOT, report)
    row['numbered_port'][field] = observed

    row, field = next((row, field) for row in report['observations']
                      for field, value in row['actual'].items() if value == 0.)
    row['actual'][field] = math.copysign(0., -row['actual'][field])
    with pytest.raises(AssertionError, match='signed-zero mismatch'):
        module._validate_observations(ROOT, report)


# ---- version-neutral source hash tests ----

def test_crlf_manifests_hash_like_lf(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    root = _copy_manifests(tmp_path)
    before = module._sources(root)
    for relative in MANIFESTS:
        path = root / relative
        path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    assert module._sources(root) == before


@pytest.mark.parametrize('relative,old,new', [
    ('rust/crates/behave-fire/Cargo.toml', 'version = "<core>"\n', 'version = "<core>"\nversion = "<core>"\n'),
    ('rust/crates/behave-fire/Cargo.toml', 'version = "<core>"\n', 'version = "abc"\n'),
    ('rust/crates/behave-py/Cargo.toml', 'path = "../behave-fire", version = "<core>" }', 'path = "../behave-fire" }'),
    ('rust/crates/behave-py/Cargo.toml', 'path = "../behave-fire"', 'path = "../behave-fire-copy"'),
    ('rust/Cargo.lock', 'name = "behave-py"\nversion = "<py>"\n', 'name = "behave-py"\n'),
    ('rust/Cargo.lock', 'name = "behave-fire"\nversion = "<core>"\n',
     'name = "behave-fire"\nversion = "<core>"\n\n[[package]]\nname = "behave-fire"\nversion = "<core>"\n'),
])
def test_malformed_version_fields_raise(tmp_path, relative, old, new):
    module = importlib.import_module('cpp_calibrate')
    root = _copy_manifests(tmp_path)
    _edit(root, relative, old, new)
    with pytest.raises(ValueError, match=Path(relative).name):
        module._sources(root)


@pytest.mark.parametrize('relative,old,new', [
    ('rust/crates/behave-py/Cargo.toml', 'numpy = "=0.27.0"', 'numpy = "=0.27.1"'),
    ('rust/crates/behave-py/Cargo.toml', 'default = []', 'default = []\nextra = []'),
    ('rust/crates/behave-fire/Cargo.toml', 'edition.workspace = true', 'edition = "2021"'),
    ('rust/Cargo.lock', 'name = "numpy"\nversion = "0.27.0"', 'name = "numpy"\nversion = "0.27.1"'),
    ('rust/Cargo.toml', '[workspace.package]', '[profile.release]\nlto = true\n\n[workspace.package]'),
])
def test_non_version_manifest_edits_change_source_hashes(tmp_path, relative, old, new):
    module = importlib.import_module('cpp_calibrate')
    root = _copy_manifests(tmp_path)
    before = module._sources(root)
    _edit(root, relative, old, new)
    assert module._sources(root)[relative] != before[relative]


def test_unrelated_sources_keep_text_digest(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    root = _copy_manifests(tmp_path)
    tool = root / 'tools/example.py'
    tool.parent.mkdir(parents=True)
    tool.write_bytes(b'VALUE = 1\r\n')
    sources = module._sources(root)
    assert sources['tools/example.py'] == reference._text_digest(tool)
    assert sources['rust/Cargo.toml'] == reference._text_digest(root / 'rust/Cargo.toml')


def test_version_only_edits_keep_source_hashes(tmp_path):
    module = importlib.import_module('cpp_calibrate')
    root = _copy_manifests(tmp_path)
    before = module._sources(root)
    _edit(root, 'rust/Cargo.lock', 'name = "behave-fire"\nversion = "<core>"',
          'name = "behave-fire"\nversion = "9.8.7"')
    _edit(root, 'rust/Cargo.lock', 'name = "behave-py"\nversion = "<py>"', 'name = "behave-py"\nversion = "9.8.7-rc.6"')
    _edit(root, 'rust/crates/behave-py/Cargo.toml', 'path = "../behave-fire", version = "<core>"',
          'path = "../behave-fire", version = "9.8.7"')
    _edit(root, 'rust/crates/behave-py/Cargo.toml', 'version = "<py>"', 'version = "9.8.7-rc.6"')
    _edit(root, 'rust/crates/behave-fire/Cargo.toml', 'version = "<core>"\n', 'version = "9.8.7"\n')
    assert module._sources(root) == before
