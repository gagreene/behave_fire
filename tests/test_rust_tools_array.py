"""Frozen tool and state-helper parity."""
import pytest

pytestmark = pytest.mark.rust
from parity_support import cases, run_case


@pytest.mark.parametrize("case", cases("fine_dead_fuel_moisture_tool", "vapor_pressure_deficit_calculator", "fire_size", "slope_tool", "behave_units", "unit_classes"))
def test_tool_parity(case):
    run_case(case)
