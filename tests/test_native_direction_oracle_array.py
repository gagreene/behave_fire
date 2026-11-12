"""Require current-host Python parity for every frozen integer direction input."""

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())
sys.path.insert(0, str(ROOT / "tools"))
from probe_direction_fixtures import _decode, _encode

pytestmark = pytest.mark.rust


def _cases():
    path = ROOT / "rust/crates/behave-fire/tests/data/integer_direction.json"
    return [pytest.param(case, id=f"direction-{index}") for index, case in enumerate(json.loads(path.read_text(encoding="utf-8")))]


@pytest.mark.parametrize("case", _cases())
def test_direction_fixtures_match_current_host_python(case):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native

    inputs = {key: _decode(value) for key, value in case["inputs"].items()}
    normalized = dict(inputs)
    for key, value in inputs.items():
        if np.asanyarray(value).dtype.kind in 'Oc':
            with pytest.raises(TypeError):
                native.calculate_direction_of_max_spread(**inputs, **case['options'])
            return
        normalized[key] = np.asarray(np.ma.asarray(value, dtype=np.float64).filled(np.nan))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            expected = reference.calculate_direction_of_max_spread(**normalized, **case["options"])
            expected = np.where(np.isnan(normalized['rate_vector']), np.nan, expected)
        except Exception as error:
            with pytest.raises(type(error)):
                native.calculate_direction_of_max_spread(**inputs, **case["options"])
            return
        actual = native.calculate_direction_of_max_spread(**inputs, **case["options"])
    expected_metadata, actual_metadata = _encode(expected), _encode(actual)
    expected_metadata.pop("values")
    actual_metadata.pop("values")
    assert actual_metadata == expected_metadata
    np.testing.assert_array_equal(np.isnan(np.ma.getdata(actual)), np.isnan(np.ma.getdata(expected)))
    np.testing.assert_array_equal(np.isposinf(np.ma.getdata(actual)), np.isposinf(np.ma.getdata(expected)))
    np.testing.assert_array_equal(np.isneginf(np.ma.getdata(actual)), np.isneginf(np.ma.getdata(expected)))
    np.testing.assert_allclose(np.ma.getdata(actual), np.ma.getdata(expected), rtol=1e-6 if expected.dtype == np.float32 else 1e-9, atol=1e-7 if expected.dtype == np.float32 else 1e-12)
