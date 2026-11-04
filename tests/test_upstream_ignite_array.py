"""Legacy lightning compatibility and caller-visible deprecation checks."""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from behave_fire.components import calculate_lightning_ignition_probability as exported_lightning
from behave_fire.components.ignite import (
    calculate_firebrand_ignition_probability,
    calculate_lightning_ignition_probability,
)

LEGACY = json.loads((Path(__file__).parent / "data/legacy_lightning.json").read_text(encoding="utf-8"))


def test_firebrand_remains_supported():
    assert float(calculate_firebrand_ignition_probability(80.0, 0.0, 0.06)) >= 0.0


@pytest.mark.parametrize("charge", range(3))
def test_lightning_broadcast_results_and_caller_warning(charge):
    with pytest.warns(DeprecationWarning, match="scientific basis") as warnings:
        call_line = inspect.currentframe().f_lineno + 1
        actual = calculate_lightning_ignition_probability(
            np.arange(8)[:, None], np.array(LEGACY["moisture"])[None, :],
            np.array(LEGACY["duff_depth_inches"])[None, :], charge,
        )
    assert len(warnings) == 1
    assert Path(warnings[0].filename).resolve() == Path(__file__).resolve()
    assert warnings[0].lineno == call_line
    assert actual.shape == (8, 3)
    np.testing.assert_allclose(actual, LEGACY["probabilities"][charge], rtol=1e-14, atol=1e-14)


def test_lightning_reexport_preserves_identity_and_warning():
    assert exported_lightning is calculate_lightning_ignition_probability
    with pytest.warns(DeprecationWarning, match="compatibility only"):
        value = exported_lightning(0, LEGACY["moisture"][0], LEGACY["duff_depth_inches"][0], 0)
    assert float(value) == pytest.approx(LEGACY["probabilities"][0][0][0], rel=1e-14)


@pytest.mark.parametrize("charge", range(3))
@pytest.mark.parametrize("bed", range(8))
@pytest.mark.parametrize("case", range(3))
def test_lightning_scalar_results_preserved(bed, charge, case):
    with pytest.warns(DeprecationWarning):
        actual = calculate_lightning_ignition_probability(
            bed, LEGACY["moisture"][case], LEGACY["duff_depth_inches"][case], charge,
        )
    assert np.shape(actual) == ()
    assert float(actual) == pytest.approx(LEGACY["probabilities"][charge][bed][case], rel=1e-14, abs=1e-14)
