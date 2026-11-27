"""Check the bounded standalone SlopeTool measurement contract."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from behave_fire.components.behave_units import LengthUnits, SlopeUnits

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
slope = importlib.import_module('cpp_slope_tool')


# ---- guarded measurements tests ----

def test_slope_tool_loads_the_two_pinned_cases():
    projection, map_slope = slope.load_observations(ROOT)

    assert projection['id'] == 'slope-projection'
    assert len(projection['output']) == 7
    assert map_slope['id'] == 'slope-map'
    assert map_slope['output']['elevation_ft'] == 60


@pytest.mark.parametrize('replacement', ['1980.0', '1980.5', 'true', '-0', '24000'])
def test_slope_tool_rejects_unapproved_map_scale(tmp_path, replacement):
    target = tmp_path / slope.DATASET
    target.mkdir(parents=True)
    path = ROOT / slope.DATASET / 'cases.jsonl'
    (target / path.name).write_text(path.read_text(encoding='utf-8').replace('"scale":1980', f'"scale":{replacement}'),
                                    encoding='utf-8')

    with pytest.raises(ValueError, match='approved cases'):
        slope.load_cases(tmp_path)


# ---- measurement receipt tests ----

def test_slope_tool_requires_measurement_only_policy():
    maxima = dict.fromkeys(slope.FIELDS, 0.0)
    receipt = dict(artifacts=slope.artifacts(ROOT), comparisons=176, nonmatching=0, max_abs=maxima)
    policy = slope.MEASUREMENT_POLICY

    slope.validate_measurement(ROOT, policy, None)
    slope.validate_measurement(ROOT, policy, {'slope_tool': receipt})
    with pytest.raises(ValueError, match='measurement-only'):
        slope.validate_measurement(ROOT, dict(policy, approved=True), {'slope_tool': receipt})
    with pytest.raises(ValueError, match='measurement receipt'):
        slope.validate_measurement(ROOT, policy, {})
    with pytest.raises(ValueError, match='comparison statistics'):
        slope.validate_measurement(ROOT, policy, {'slope_tool': dict(receipt, nonmatching=1)})


def test_slope_tool_summarizes_both_output_shapes(tmp_path):
    cpp_rows = slope.load_observations(ROOT)
    port_rows = [dict(id=row['id'], output=row['output']) for row in cpp_rows]
    artifacts = slope.artifacts(ROOT)
    reports = {}
    paths = []
    for compiler in ('GNU', 'MSVC'):
        name = f'cpp-{compiler}'
        report = dict(kind='cpp', compiler={'id': compiler}, protocol={'slope_tool': slope.GUARDS},
                      slope_tool=dict(artifacts=artifacts, observations=cpp_rows))
        raw = json.dumps(report).encode()
        path = tmp_path / f'{name}.json'
        path.write_bytes(raw)
        paths.append(path)
        reports[name] = {'sha256': hashlib.sha256(raw).hexdigest()}
    for platform in ('linux-x86_64', 'linux-aarch64', 'linux-musl-x86_64', 'macos-x86_64', 'macos-arm64',
                     'windows-x86_64', 'windows-arm64'):
        name = f'ports-{platform}'
        report = dict(kind='ports', platform_key=platform,
                      slope_tool=dict(artifacts=artifacts, observations={'python': port_rows, 'rust': port_rows}))
        raw = json.dumps(report).encode()
        path = tmp_path / f'{name}.json'
        path.write_bytes(raw)
        paths.append(path)
        reports[name] = {'sha256': hashlib.sha256(raw).hexdigest()}

    receipt = slope.measurement_evidence(ROOT, paths, {'reports': reports})
    assert receipt['comparisons'] == 176
    assert receipt['nonmatching'] == 0
    assert receipt['max_abs'] == dict.fromkeys(slope.FIELDS, 0.0)
    paths[-1].write_bytes(paths[-1].read_bytes() + b' ')
    with pytest.raises(ValueError, match='source-qualified evidence'):
        slope.measurement_evidence(ROOT, paths, {'reports': reports})


# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_slope_tool_measures_public_port_calls(backend):
    tool_type = importlib.import_module(f'behave_fire.{backend}').SlopeTool
    feet = LengthUnits.LengthUnitsEnum.Feet
    degrees = SlopeUnits.SlopeUnitsEnum.Degrees
    tool = tool_type()
    projection, map_slope = slope.measure_ports(ROOT, (backend,))[backend]

    tool.calculate_horizontal_distance(8.0, feet, 30.0, degrees)
    expected = {f'horizontal_{index * 15}_ft': tool.get_horizontal_distance_at_index(index, feet)
                for index in range(7)}
    assert projection == dict(id='slope-projection', output={k: np.asarray(v).item() for k, v in expected.items()})
    tool.calculate_slope_from_map_measurements(1980, 0.125, feet, 20.0, 3, feet)
    expected = dict(
        elevation_ft=tool.get_slope_elevation_change_from_map_measurements(feet),
        horizontal_ft=tool.get_slope_horizontal_distance_from_map_measurements(feet),
        slope_degrees=tool.get_slope_from_map_measurements_in_degrees(),
        slope_percent=tool.get_slope_from_map_measurements_in_percent(),
    )
    assert map_slope == dict(id='slope-map', output={k: np.asarray(v).item() for k, v in expected.items()})
