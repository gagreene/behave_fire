"""Frozen crown-stage parity."""
import pytest

from parity_support import cases, run_case

pytestmark = pytest.mark.rust


@pytest.mark.parametrize("case", cases("crown"))
def test_crown_parity(case):
    run_case(case)
