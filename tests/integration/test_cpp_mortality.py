"""Guard the standalone Mortality measurement scope and numeric decoding."""

import importlib
import math
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
mortality = importlib.import_module('cpp_mortality')


# ---- guarded measurements tests ----

def test_mortality_preserves_signed_zero_in_observations(tmp_path):
    directory = tmp_path / mortality.DATASET
    directory.mkdir(parents=True)
    (directory / 'cases.jsonl').write_bytes((ROOT / mortality.DATASET / 'cases.jsonl').read_bytes())
    rows = (ROOT / mortality.DATASET / 'results.jsonl').read_bytes()
    path = directory / 'results.jsonl'
    path.write_bytes(rows.replace(b'"scorch_height_ft":0', b'"scorch_height_ft":-0', 1))

    height = mortality.read_cpp_results(tmp_path, path)[0]['output']['scorch_height_ft']
    assert math.copysign(1, height) == -1


def test_mortality_rejects_signed_zero_in_inputs(tmp_path):
    directory = tmp_path / mortality.DATASET
    directory.mkdir(parents=True)
    rows = (ROOT / mortality.DATASET / 'cases.jsonl').read_bytes()
    (directory / 'cases.jsonl').write_bytes(rows.replace(b'"intensity_btu_ft_s":0', b'"intensity_btu_ft_s":-0', 1))

    with pytest.raises(ValueError, match='approved cases'):
        mortality.load_cases(tmp_path)
