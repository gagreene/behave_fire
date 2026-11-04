"""Frozen effect-function parity."""
import pytest

pytestmark = pytest.mark.rust
from parity_support import cases, run_case


@pytest.mark.parametrize("case", cases("safety", "spot", "ignite", "mortality", "boundaries"))
def test_effect_parity(case):
    run_case(case)


def test_missing_species_equation_warning():
    from behave_fire.rust import SpeciesMasterTable, SpeciesMasterTableRecord, mortality

    table = SpeciesMasterTable()
    table.add_species_record(SpeciesMasterTableRecord("CUSTOM", mortality_equation_number=77))
    with pytest.warns(UserWarning, match="CUSTOM.*77"):
        coefficients = mortality.build_mortality_lookup(table)
    assert coefficients.shape == (121,3)
    assert not coefficients[77].any()


def test_noncanonical_lightning_charge_uses_unknown_branch():
    import numpy as np
    from behave_fire.components import ignite as reference
    from behave_fire.rust import ignite

    with pytest.warns(DeprecationWarning):
        expected=reference.calculate_lightning_ignition_probability(6,.2,1.,.5)
    with pytest.warns(DeprecationWarning):
        actual=ignite.calculate_lightning_ignition_probability(6,.2,1.,.5)
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)
