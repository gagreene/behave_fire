"""Check CI tier selection and the master reuse proof without contacting GitHub."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("ci_plan", ROOT / "tools/ci_plan.py")
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)
PUSH = {"EVENT": "push", "GITHUB_REF": "refs/heads/master", "GITHUB_SHA": "landed", "GITHUB_REPOSITORY": "o/r"}


def _pull(number=76, base="master", merged="2026-11-25T00:00:00Z", head="head"):
    return {"number": number, "base": {"ref": base}, "merged_at": merged, "head": {"sha": head}}


def _reader(pulls, runs):
    def api(path):
        if path.endswith("/pulls"):
            return pulls
        return {"workflow_runs": runs}
    return api


def _record(tree="tree-a", head="head", number=76, tier="full"):
    return {"tier": tier, "tree": tree, "head_sha": head, "pull_request": number}


def _run(identifier=1, created="2026-11-25T01:00:00Z", conclusion="success"):
    return {"id": identifier, "created_at": created, "conclusion": conclusion, "html_url": f"run/{identifier}"}


# ---- reuse proof tests ----

def test_reuse_accepts_only_the_latest_matching_full_record():
    runs = [_run(1, "2026-11-25T01:00:00Z"), _run(2, "2026-11-25T02:00:00Z")]
    records = {1: _record(), 2: _record(tree="tree-b")}
    found = plan.find_reusable_result("landed", "o/r", _reader([_pull()], runs), lambda run, _: records[run],
                                      lambda _: "tree-a")
    assert found["run_url"] == "run/1"


@pytest.mark.parametrize("pulls,runs,record", [
    ([], [_run()], _record()),
    ([_pull(base="pre_release")], [_run()], _record()),
    ([_pull(merged=None)], [_run()], _record()),
    ([_pull()], [], _record()),
    ([_pull()], [_run(conclusion="failure")], _record()),
    ([_pull()], [_run()], None),
    ([_pull()], [_run()], _record(tree="other")),
    ([_pull()], [_run()], _record(head="older-head")),
    ([_pull()], [_run()], _record(number=75)),
    ([_pull()], [_run()], _record(tier="targeted")),
])
def test_reuse_requires_complete_matching_proof(pulls, runs, record):
    found = plan.find_reusable_result("landed", "o/r", _reader(pulls, runs), lambda *_: record, lambda _: "tree-a")
    assert found is None


# ---- tier selection tests ----

@pytest.mark.parametrize("environment,expected", [
    ({"EVENT": "pull_request", "BASE_REF": "pre_release"}, ("quick", "")),
    ({"EVENT": "pull_request", "BASE_REF": "feature/x"}, ("quick", "")),
    ({"EVENT": "pull_request", "BASE_REF": "master"}, ("full", "full")),
    ({"EVENT": "pull_request", "BASE_REF": "master", "ACTION": "edited", "BASE_CHANGED": "false"}, ("none", "")),
    ({"EVENT": "pull_request", "BASE_REF": "master", "ACTION": "edited", "BASE_CHANGED": "true"}, ("full", "full")),
    ({"EVENT": "push", "GITHUB_REF": "refs/heads/pre_release"}, ("quick", "")),
    ({"EVENT": "workflow_dispatch"}, ("full", "full")),
    ({"EVENT": "workflow_dispatch", "SCOPE": "full"}, ("full", "full")),
    ({"EVENT": "workflow_dispatch", "SCOPE": "musl-and-sdist"}, ("targeted", "musl-and-sdist")),
    ({"EVENT": "workflow_dispatch", "SCOPE": "sdist"}, ("targeted", "sdist")),
    ({"EVENT": "push", "GITHUB_REF": "refs/tags/v0.1.0", "CALLED_TIER": "full"}, ("full", "full")),
    ({"EVENT": "workflow_dispatch", "CALLED_TIER": "full"}, ("full", "full")),
])
def test_choose_maps_events_to_tiers(environment, expected):
    result = plan.choose(environment, lambda *_: pytest.fail("reuse lookup is only for master pushes"))
    assert (result["tier"], result["scope"]) == expected


def test_choose_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="unknown scope"):
        plan.choose({"EVENT": "workflow_dispatch", "SCOPE": "everything"})


def test_master_push_falls_back_to_full_without_proof():
    assert plan.choose(PUSH, lambda *_: None) == {"tier": "full", "scope": "full"}


def test_master_push_reuses_a_matching_full_result():
    result = plan.choose(PUSH, lambda *_: {"run_url": "run/7"})
    assert result == {"tier": "reused", "scope": "", "reused_run": "run/7"}
