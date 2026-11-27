"""Frozen tool and state-helper parity."""
import pytest

from parity_batch_support import grouped_cases, run_cases
from parity_support import cases, run_case

pytestmark = pytest.mark.rust
UNIT_CLASS_GROUPS, INDIVIDUAL_UNIT_CLASSES = grouped_cases()


# ---- tool parity tests ----

@pytest.mark.parametrize("case", cases("fine_dead_fuel_moisture_tool", "vapor_pressure_deficit_calculator", "fire_size",
    "slope_tool", "behave_units") + INDIVIDUAL_UNIT_CLASSES)
def test_tool_parity(case):
    run_case(case)


# ---- unit class parity tests ----

@pytest.mark.parametrize("records", UNIT_CLASS_GROUPS)
def test_unit_class_parity_batch(records):
    run_cases(records)
