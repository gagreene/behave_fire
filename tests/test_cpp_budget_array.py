"""Owner-approved Surface fixture ceilings cannot silently expand."""

import importlib
import json
import random
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
reference = importlib.import_module('cpp_reference')


def _bundle(tmp_path):
    root = tmp_path
    canonical = root / 'tests/data/cpp_reference'
    canonical.mkdir(parents=True)
    reference._json(canonical / 'fixture-budget.json', dict(approved=True, owner_approval='2026-11-15', limits={
        'case_limit': 800, 'canonical': {'raw_bytes': 6291456, 'gzip_bytes': 524288},
        'standalone': {'raw_bytes': 524288, 'gzip_bytes': 65536}}))
    for name in ('surface', 'surface_custom'):
        directory = canonical / name
        directory.mkdir()
        (directory / 'cases.jsonl').write_bytes(b'{}\n')
    (root / 'rust/crates/behave-fire/tests/data/cpp_reference').mkdir(parents=True)
    return root


def test_fixture_budget_allows_the_exact_case_ceiling(tmp_path):
    root = _bundle(tmp_path)
    path = root / 'tests/data/cpp_reference/surface/cases.jsonl'
    path.write_bytes(b'{}\n' * 799)
    assert reference.check_fixture_budget(root)['cases'] == 800
    path.write_bytes(b'{}\n' * 800)
    with pytest.raises(ValueError, match='case'):
        reference.check_fixture_budget(root)


@pytest.mark.parametrize('bundle,kind,size', [('canonical', 'raw', 6291457), ('canonical', 'gzip', 524289),
                                           ('standalone', 'raw', 524289), ('standalone', 'gzip', 65537)])
def test_fixture_budget_rejects_excess_payload(tmp_path, bundle, kind, size):
    root = _bundle(tmp_path)
    reference.check_fixture_budget(root)
    directory = root / ('tests/data/cpp_reference' if bundle == 'canonical'
                        else 'rust/crates/behave-fire/tests/data/cpp_reference')
    payload = b'0' * size if kind == 'raw' else random.Random(0).randbytes(size)
    (directory / 'padding.bin').write_bytes(payload)
    with pytest.raises(ValueError, match=bundle):
        reference.check_fixture_budget(root)


def test_fixture_budget_rejects_unapproved_limit_changes(tmp_path):
    root = _bundle(tmp_path)
    path = root / 'tests/data/cpp_reference/fixture-budget.json'
    policy = json.loads(path.read_text())
    policy['limits']['canonical']['raw_bytes'] += 1
    reference._json(path, policy)
    with pytest.raises(ValueError, match='approved'):
        reference.check_fixture_budget(root)
