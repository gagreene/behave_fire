"""Frozen crown-stage parity."""
import pytest

pytestmark = pytest.mark.rust
from parity_support import cases, run_case


@pytest.mark.parametrize("case", cases("crown"))
def test_crown_parity(case):
    run_case(case)
