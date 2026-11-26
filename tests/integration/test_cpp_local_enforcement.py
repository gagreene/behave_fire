"""Prove that the full local check fails on a numerical regression, even with qualification withdrawn.

Each test copies a pinned C++ results fixture to a temporary directory, perturbs one expected
value, and runs the same comparison path the local check uses. The repository fixtures are
never modified.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import cpp_controls
import cpp_fire_size
import cpp_ignite
import cpp_mortality
import cpp_reference as reference

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ["python", pytest.param("rust", marks=pytest.mark.rust)]
EXACT = {"fire_size": cpp_fire_size, "ignite": cpp_ignite, "mortality": cpp_mortality}


def _perturbed_copy(tmp_path, dataset, change):
    source = ROOT / "tests/data/cpp_reference" / dataset / "results.jsonl"
    copy = tmp_path / dataset / "results.jsonl"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(source, copy)
    rows = [json.loads(line) for line in copy.read_text(encoding="utf-8").splitlines()]
    field = change(rows)
    copy.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert copy.read_bytes() != source.read_bytes()
    return [json.loads(line) for line in copy.read_text(encoding="utf-8").splitlines()], field


def _toward_zero(value):
    return math.nextafter(value, 0.0) if value != 0.0 else math.nextafter(0.0, 1.0)


# ---- deliberate failure tests ----

@pytest.mark.parametrize("dataset", sorted(EXACT))
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.cpp_comparison
def test_exact_mismatch_fails_while_qualification_is_withdrawn(tmp_path, dataset, backend):
    module = EXACT[dataset]
    policy = reference.load_reference()["policy"][dataset]

    def change(rows):
        name = module.FIELDS[0]
        rows[0]["output"][name] = _toward_zero(rows[0]["output"][name])
        return name

    expected, field = _perturbed_copy(tmp_path, dataset, change)
    measured = module.measure_ports(ROOT, (backend,))[backend]
    assert measured[0]["id"] == expected[0]["id"]
    with pytest.raises(AssertionError, match=field):
        module.check_exact(measured[0]["output"], expected[0]["output"], policy)
    # The unperturbed remainder still passes, so the failure is the one-ulp change alone.
    for actual, row in zip(measured[1:], expected[1:], strict=True):
        module.check_exact(actual["output"], row["output"], policy)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.cpp_comparison
def test_surface_cap_violation_fails_while_qualification_is_withdrawn(tmp_path, monkeypatch, backend):
    field = reference.FIELDS[0]

    # Load-control models are documented exceptions that the check skips, so perturb a compared case.
    compared = {case["id"] for case in reference.load_reference()["cases"]
                if case["fuel_model"] not in reference.SURFACE_LOAD_CONTROLS}

    def change(rows):
        target = next(row for row in rows if row["status"] == "ok" and row["id"] in compared)
        value = target["output"][field]
        target["output"][field] = value + max(1e-9, abs(value) * 1e-9)
        return target["id"]

    perturbed, case_id = _perturbed_copy(tmp_path, "surface", change)
    original = reference.load_reference

    def load_with_perturbed_results(root=reference.ROOT, **options):
        data = original(root, **options)
        return dict(data, results=perturbed)

    monkeypatch.setattr(cpp_controls.reference, "load_reference", load_with_perturbed_results)
    with pytest.raises(AssertionError, match=field):
        cpp_controls.measure_ports(ROOT, tmp_path / "report.json", backends=(backend,))
    assert case_id in compared
