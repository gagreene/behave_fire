"""Negative controls for grouped unit-conversion acceptance comparisons."""

import numpy as np
import pytest

from parity_batch_support import assert_comparisons, grouped_cases, run_cases
from parity_support import cases


@pytest.mark.parametrize("corruption", ["dtype", "shape", "mask", "value", "nan", "infinity", "zero-sign"])
def test_batch_comparison_names_the_first_corrupt_source_case(corruption):
    expected = np.array([0., 1., np.nan, np.inf])
    actual = expected.copy()
    if corruption == "dtype":
        actual = actual.astype(np.float32)
    elif corruption == "shape":
        actual = actual.reshape(2, 2)
    elif corruption == "mask":
        actual = np.ma.array(actual, mask=False)
    elif corruption == "value":
        actual[1] = 2.
    elif corruption == "nan":
        actual[2] = 0.
    elif corruption == "infinity":
        actual[3] = -np.inf
    else:
        actual[0] = -0.
    rows = [("good", expected.copy(), expected), ("bad-source", actual, expected),
            ("later", actual.copy(), expected)]
    with pytest.raises(AssertionError, match="bad-source"):
        assert_comparisons(rows)


def test_batch_failure_order_includes_values_before_later_structure_errors():
    rows = [("first-value", np.array([2.]), np.array([1.])),
            ("later-dtype", np.array([1.], np.float32), np.array([1.]))]
    with pytest.raises(AssertionError, match="first-value"):
        assert_comparisons(rows)


@pytest.mark.rust
def test_grouped_calls_preserve_inputs_and_independent_warning_attribution(monkeypatch):
    from behave_fire.rust import behave_units

    records = grouped_cases()[0][0].values[0][:2]
    method = records[0]["entry_point"].split(".")[-2:]
    owner = getattr(behave_units, method[0])
    operation = getattr(owner, method[1])
    seen = []

    def observe(**arguments):
        """Observe real native calls and inject a warning only into the second case."""
        import warnings

        seen.append((arguments["value"].dtype.name, arguments["value"].shape, arguments["units"]))
        if len(seen) == 2:
            warnings.warn("second case only", UserWarning, stacklevel=2)
        return operation(**arguments)

    monkeypatch.setattr(owner, method[1], observe)
    with pytest.raises(AssertionError, match=records[1]["id"] + ".*warnings"):
        run_cases(records)
    assert seen == [(case["inputs"]["value"]["dtype"], tuple(case["inputs"]["value"]["shape"]),
                     case["inputs"]["units"]) for case in records]


@pytest.mark.rust
@pytest.mark.parametrize("fault", ["warning", "exception", "oracle"])
def test_grouped_first_value_failure_precedes_later_case_failure(fault, monkeypatch):
    from copy import deepcopy

    from behave_fire.rust import behave_units

    records = deepcopy(grouped_cases()[0][0].values[0][:2])
    method = records[0]["entry_point"].split(".")[-2:]
    owner = getattr(behave_units, method[0])
    operation = getattr(owner, method[1])
    calls = []
    if fault == "oracle":
        records[1]["output"]["values"][0] = 100.

    def corrupt(**arguments):
        """Corrupt an early result before a distinct later failure."""
        import warnings

        calls.append(arguments)
        result = operation(**arguments).copy()
        if len(calls) == 1:
            result[0] = 100.
        elif fault == "warning":
            warnings.warn("later case", UserWarning, stacklevel=2)
        else:
            raise ValueError("later case")
        return result

    monkeypatch.setattr(owner, method[1], corrupt)
    with pytest.raises(AssertionError, match=records[0]["id"]):
        run_cases(records)
    assert len(calls) == (1 if fault == "oracle" else 2)


@pytest.mark.rust
def test_grouped_results_snapshot_reused_native_buffers(monkeypatch):
    from behave_fire.rust import behave_units

    group = grouped_cases()[0][0].values[0]
    records = (group[0], next(case for case in group if case["inputs"]["units"] != group[0]["inputs"]["units"]))
    method = records[0]["entry_point"].split(".")[-2:]
    owner = getattr(behave_units, method[0])
    operation = getattr(owner, method[1])
    shared = np.empty(3, dtype=np.float64)
    results = []

    def reuse(**arguments):
        """Return a shared buffer overwritten by the next real conversion."""
        np.copyto(shared, operation(**arguments))
        results.append(shared.copy())
        return shared

    monkeypatch.setattr(owner, method[1], reuse)
    run_cases(records)
    assert len(results) == 2
    assert np.any(results[0] != results[1]), "The second case must overwrite different numerical values"


def test_grouping_preserves_each_original_case_and_keeps_object_rejections_separate():
    original = cases("unit_classes")
    grouped, individual = grouped_cases()
    gathered = [case for parameter in grouped for case in parameter.values[0]]
    gathered.extend(parameter.values[0] for parameter in individual)
    assert sorted(case["id"] for case in gathered) == sorted(parameter.values[0]["id"] for parameter in original)
    assert len(gathered) == len({case["id"] for case in gathered}) == 994
    assert len(individual) == 142
    for parameter in grouped:
        records = parameter.values[0]
        assert len(records) > 1
        assert len({case["entry_point"] for case in records}) == 1
        assert len({(case["inputs"]["value"]["dtype"], tuple(case["inputs"]["value"]["shape"]))
                    for case in records}) == 1
        assert all(case["inputs"]["value"]["dtype"] != "object" for case in records)
