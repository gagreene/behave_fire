"""Frozen fixed-step containment parity."""
import pytest

pytestmark = pytest.mark.rust
from parity_support import cases, run_case


@pytest.mark.parametrize("case", cases("contain"))
def test_containment_parity(case):
    run_case(case)


def test_containment_rejects_fractional_step_count():
    from behave_fire.rust import contain

    with pytest.raises(TypeError):
        contain.run_contain_sim_array(10.,2.,1.,10.,0.,max_steps=1.5)
