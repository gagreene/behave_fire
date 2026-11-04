"""Frozen surface-stage parity."""
import pytest

pytestmark = pytest.mark.rust
from parity_support import cases, run_case


@pytest.mark.parametrize("case", cases("surface", "behave"))
def test_surface_parity(case):
    run_case(case)
