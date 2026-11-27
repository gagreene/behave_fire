"""Verify live C++ declarations against a hand-written inventory for a small header."""

import hashlib
import json

import pytest
from tools.cpp_live_api import verify_public_api

HEADER = 'src/behave/gauge.h'
SOURCE = '''\
#pragma once
namespace Fire {
enum Unit { Feet, Meters };
struct Station {
class Gauge {
    void reset();
public:
    enum Unit { Feet, Meters };
    double read(Unit unit) const;
    double read(
        int samples
    ) const { return samples; }
};
};
}
'''


@pytest.fixture
def _registered_api(tmp_path):
    header = tmp_path / HEADER
    header.parent.mkdir(parents=True)
    header.write_text(SOURCE, encoding='utf-8')
    signatures = (
        ('double Fire::Station::Gauge::read(Unit unit) const',
         'double Fire::Station::Gauge::read(Fire::Station::Gauge::Unit unit) const'),
        ('double Fire::Station::Gauge::read( int samples ) const',
         'double Fire::Station::Gauge::read( int samples ) const'),
    )
    identities = sorted((hashlib.sha256(f'{HEADER}|{raw}'.encode()).hexdigest()[:20], raw, qualified)
                        for raw, qualified in signatures)
    inventory = {
        'source_inputs': [{'file': HEADER, 'sha256': hashlib.sha256(header.read_bytes()).hexdigest()}],
        'public_identity_sha256': hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest(),
        'public_functions': [
            {'file': HEADER, 'id': identity, 'access': 'public', 'qualified_name': 'Fire::Station::Gauge::read',
             'signature': raw, 'lexically_qualified_signature': qualified}
            for identity, raw, qualified in identities
        ],
    }
    return header, inventory


# ---- live public declaration tests ----

def test_accepts_registered_public_overloads_and_nested_type(_registered_api, tmp_path):
    _, inventory = _registered_api
    assert verify_public_api(tmp_path, inventory) == {
        'public_functions': 2,
        'public_identity_sha256': inventory['public_identity_sha256'],
    }


def test_ignores_private_and_function_body_changes(_registered_api, tmp_path):
    header, inventory = _registered_api
    source = SOURCE.replace('void reset();', 'void clearCounters();')
    source = source.replace('return samples;', 'return 2 * samples;')
    source = '#define UNUSED_DECLARATION \\\n    double macro_only();\n' + source
    header.write_text(source, encoding='utf-8')
    assert verify_public_api(tmp_path, inventory)['public_identity_sha256'] == inventory['public_identity_sha256']


@pytest.mark.parametrize(
    ('original', 'replacement', 'mismatch'),
    [
        ('int samples', 'long samples', 'missing.*unexpected'),
        ('    double read(Unit unit) const;\n', '', 'missing'),
        ('    enum Unit', '    double read(double sample) const;\n    enum Unit', 'unexpected'),
        ('    double read(Unit unit)', 'private:\n    double read(Unit unit)', 'missing'),
        ('struct Station {', 'class Station {', 'missing'),
        ('    void reset();\npublic:', 'public:\n    void reset();', 'unexpected'),
        ('    enum Unit { Feet, Meters };\n', '', 'missing.*unexpected'),
    ],
    ids=['changed-signature', 'removed', 'added-overload', 'made-private', 'owner-made-private',
         'made-public', 'lost-nested-type'],
)
def test_rejects_live_public_declaration_drift(_registered_api, tmp_path, original, replacement, mismatch):
    header, inventory = _registered_api
    header.write_text(SOURCE.replace(original, replacement), encoding='utf-8')
    with pytest.raises(ValueError, match=mismatch):
        verify_public_api(tmp_path, inventory)


def test_rejects_unregistered_header(_registered_api, tmp_path):
    header, inventory = _registered_api
    header.with_name('extra.h').write_text('void added();\n', encoding='utf-8')
    with pytest.raises(ValueError, match='extra.h'):
        verify_public_api(tmp_path, inventory)
