"""Frozen surface-stage parity."""
import pytest

from parity_support import cases, run_case

pytestmark = pytest.mark.rust


# ---- surface parity tests ----

@pytest.mark.parametrize("case", cases("surface", "behave"))
def test_surface_parity(case):
    run_case(case)
