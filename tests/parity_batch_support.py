"""Group compatible unit comparisons while retaining every original operation call."""

import numpy as np
import pytest

from parity_support import cases, run_case
from performance_cases import assert_outputs


def assert_comparisons(rows):
    """Compare compatible outputs together, naming the first failing original case.

    :param rows: Ordered triples of source case ID, actual snapshot, and expected snapshot.
    :return: None; raises AssertionError with the source ID on any contract difference.
    """
    try:
        for _, actual, expected in rows:
            assert actual.shape == expected.shape, "shapes differ"
            assert actual.dtype == expected.dtype, "dtypes differ"
            assert not np.ma.isMaskedArray(actual) and not np.ma.isMaskedArray(expected), "masked output"
        assert_outputs(np.stack([row[1] for row in rows]), np.stack([row[2] for row in rows]))
    except AssertionError:
        for identity, actual, expected in rows:
            try:
                assert_outputs(actual, expected)
                assert not np.ma.isMaskedArray(actual) and not np.ma.isMaskedArray(expected), "masked output"
            except AssertionError as error:
                raise AssertionError(f"{identity}: {error}") from error
        raise


def grouped_cases():
    """Partition supported unit-class records by operation and original array contract.

    :return: Group parameters and unchanged individual parameters for unsupported contracts.
    """
    grouped, individual = {}, []
    for parameter in cases("unit_classes"):
        case = parameter.values[0]
        value, output = case["inputs"].get("value"), case.get("output")
        if ("exception" in case or "state_before" in case or "receiver_before" in case
                or set(case["inputs"]) != {"value", "units"}
                or not isinstance(value, dict)
                or value.get("dtype") not in ("float32", "float64", "int64", "uint32", "bool")
                or value.get("shape") != [3] or "mask" in value or value.get("read_only")
                or not isinstance(output, dict) or output.get("dtype") != "float64"
                or output.get("shape") != [3] or "mask" in output):
            individual.append(parameter)
            continue
        key = case["entry_point"], value["dtype"], tuple(case["warnings"])
        grouped.setdefault(key, []).append(case)
    return [pytest.param(tuple(records), id=f"{entry.rsplit('.', 2)[-2]}.{entry.rsplit('.', 1)[-1]}/{dtype}")
            for (entry, dtype, _), records in grouped.items()], individual


def run_cases(records):
    """Run every original case independently and batch only final numerical comparisons.

    :param records: Compatible unit-class records with unique source case IDs.
    :return: None; checks every frozen oracle, native result, and per-case warning contract.
    """
    rows = []
    for case in records:
        before = len(rows)

        def compare(actual, expected, identity=case["id"]):
            """Snapshot this call before the next original case can execute.

            :param actual: Native result for the current source case.
            :param expected: Independently calculated float64 reference.
            :param identity: Original source case ID, bound before the next iteration.
            :return: None; appends independent buffers for the final comparison.
            """
            rows.append((identity, np.asanyarray(actual).copy(), np.asanyarray(expected).copy()))

        try:
            run_case(case, compare=compare)
            assert len(rows) == before + 1, "case did not produce exactly one numerical comparison"
        except (Exception, pytest.fail.Exception) as error:
            if before:
                assert_comparisons(rows[:before])
            raise AssertionError(f"{case['id']}: {error}") from error
    assert_comparisons(rows)
