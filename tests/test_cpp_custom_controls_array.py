"""Explicit percent-unit controls must not silently repair numbered C++ records."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import cpp_controls as controls
import cpp_reference as reference

ROOT = Path(__file__).resolve().parents[1]


def _case():
    rows = map(json.loads, (ROOT / 'tests/data/cpp_reference/surface/cases.jsonl').read_text().splitlines())
    numbered = next(row for row in rows if row['id'] == 'fm110-angle0')
    return dict(numbered, id='custom-fm110-angle0', fuel_mode='percent_unit_custom', moe_percent=24.)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 1e-3, True])
def test_calibration_cannot_accept_invalid_or_wider_caps_even_for_exact_values(value):
    fields = controls.calibration_policy(ROOT)
    fields[reference.FIELDS[0]]['rtol'] = value
    actual = dict.fromkeys(reference.FIELDS, 1.)
    with pytest.raises(ValueError):
        controls.check_values(actual, actual, fields)


def test_calibration_report_rejects_a_wrong_platform_identity(tmp_path):
    report = tmp_path / 'spoofed-platform.json'
    with pytest.raises(ValueError, match='platform'):
        controls.measure_ports(ROOT, report, expected_platform='not-a-real-platform')
    assert not report.exists()


def test_control_schema_accepts_an_explicit_percent_quantity():
    """A separate unit-labelled custom definition must pass the guarded schema."""
    case = _case()
    before = deepcopy(case)
    reference._validate_case(case)
    assert case == before


@pytest.mark.parametrize('field,value', [
    ('fuel_mode', 'automatic_correction'), ('fuel_mode', None),
    ('fuel_model', -1), ('fuel_model', 0), ('fuel_model', 1), ('fuel_model', 256),
    ('moe_percent', 0.), ('moe_percent', -1.), ('moe_percent', 0.24),
    ('moe_percent', 101.), ('moe_percent', float('nan')), ('moe_percent', float('inf')),
    ('moe_percent', True), ('moe_percent', '24'),
])
def test_control_schema_rejects_unreviewed_or_unsafe_definitions(field, value):
    case = _case()
    case[field] = value
    with pytest.raises(ValueError):
        reference._validate_case(case)


def test_control_schema_requires_both_mode_and_quantity():
    for missing in ('fuel_mode', 'moe_percent'):
        case = _case()
        del case[missing]
        with pytest.raises(ValueError):
            reference._validate_case(case)


def test_controls_preserve_source_inputs_and_cover_every_affected_model():
    """Every affected scenario gets one distinct control with original weather intact."""
    cases = [json.loads(line) for line in (ROOT / 'tests/data/cpp_reference/surface/cases.jsonl').read_text(
        ).splitlines()]
    before = deepcopy(cases)
    controls = reference._custom_cases(cases)
    assert cases == before
    assert len(controls) == len({case['id'] for case in controls}) == 139
    assert {case['fuel_model'] for case in controls} == {
        110, 111, *range(150, 160), *range(166, 173), *range(190, 194)}
    originals = {case['id']: case for case in cases}
    for control in controls:
        expected = originals[control['id'].removeprefix('custom-')]
        assert {key: value for key, value in control.items()
                if key not in ('id', 'fuel_mode', 'moe_percent')} == {
                    key: value for key, value in expected.items() if key != 'id'}
        reference._validate_case(control)
    assert next(case for case in controls if case['fuel_model'] == 110)['moe_percent'] == 24.
    assert next(case for case in controls if case['fuel_model'] == 151)['moe_percent'] == 13.


def test_cross_compiler_trust_normalizes_text_but_preserves_binary_bytes(tmp_path):
    text, binary = tmp_path / 'source.cpp', tmp_path / 'data.bin'
    text.write_bytes(b'valid\r\nsource\r\n')
    first = reference._source_digest(text)
    text.write_bytes(b'valid\nsource\n')
    assert reference._source_digest(text) == first
    text.write_bytes(b'changed\nsource\n')
    assert reference._source_digest(text) != first
    binary.write_bytes(b'\x00\r\n')
    first = reference._source_digest(binary)
    binary.write_bytes(b'\x00\n')
    assert reference._source_digest(binary) != first
    text.write_bytes(b'// temperature \xb0C\r\n')
    first = reference._source_digest(text)
    text.write_bytes(b'// temperature \xb0C\n')
    assert reference._source_digest(text) == first
