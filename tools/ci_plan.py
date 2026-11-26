"""Choose the CI tier for one workflow run and reuse a matching full result on master.

The CI workflow runs this script once in its ``plan`` job and writes the printed
``key=value`` lines to ``GITHUB_OUTPUT``. Tiers:

- ``quick``: pull requests into any branch except ``master``, and pushes to ``pre_release``.
- ``full``: pull requests into ``master``, calls from ``release.yml``, a manual dispatch
  with scope ``full``, and a ``master`` push without a reusable full result.
- ``reused``: a ``master`` push whose Git tree equals the merge tree that a successful
  full pull request run tested; that run's ``ci-full-result`` artifact is the proof.
- ``targeted``: a manual dispatch with a narrower scope (``musl``, ``sdist``, or
  ``musl-and-sdist``).
- ``none``: a pull request edit that does not change the base branch.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FULL_RESULT = "ci-full-result"
SCOPES = ("full", "musl", "sdist", "musl-and-sdist")


def _api(path):
    """Read one GitHub REST API resource through the GitHub CLI.

    :param path: API path relative to ``https://api.github.com/``.
    :return: Decoded JSON response.
    """
    output = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout
    return json.loads(output)


def _download_result(run_id, repository):
    """Download and decode a run's full-result record.

    :param run_id: Workflow run identifier.
    :param repository: ``owner/name`` repository.
    :return: Decoded record, or ``None`` when the run has no such artifact.
    """
    with tempfile.TemporaryDirectory() as directory:
        process = subprocess.run(["gh", "run", "download", str(run_id), "--repo", repository, "--name", FULL_RESULT,
                                  "--dir", directory], capture_output=True, text=True)
        path = Path(directory) / "result.json"
        if process.returncode != 0 or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _git_tree(revision):
    """Return the Git tree object of a checked-out revision.

    :param revision: Revision name, such as ``HEAD``.
    :return: Tree object name.
    """
    return subprocess.run(["git", "rev-parse", revision + "^{tree}"], check=True, capture_output=True,
                          text=True).stdout.strip()


def choose(environment, find_reusable=None):
    """Choose the tier and wheel-validation scope for one run.

    :param environment: Mapping with ``EVENT``, ``ACTION``, ``BASE_CHANGED``, ``BASE_REF``,
        ``CALLED_TIER``, ``SCOPE``, ``GITHUB_REF``, ``GITHUB_SHA``, and ``GITHUB_REPOSITORY``.
    :param find_reusable: Callable taking ``(sha, repository)`` and returning a reusable run
        record or ``None``; required only for ``master`` pushes.
    :return: Dictionary with ``tier``, ``scope``, and, when reused, ``reused_run``.
    """
    if environment.get("CALLED_TIER"):
        return {"tier": "full", "scope": "full"}
    event = environment.get("EVENT", "")
    if event == "pull_request":
        if environment.get("ACTION") == "edited" and environment.get("BASE_CHANGED") != "true":
            return {"tier": "none", "scope": ""}
        if environment.get("BASE_REF") == "master":
            return {"tier": "full", "scope": "full"}
        return {"tier": "quick", "scope": ""}
    if event == "push":
        if environment.get("GITHUB_REF") != "refs/heads/master":
            return {"tier": "quick", "scope": ""}
        record = find_reusable(environment["GITHUB_SHA"], environment["GITHUB_REPOSITORY"])
        if record is None:
            return {"tier": "full", "scope": "full"}
        return {"tier": "reused", "scope": "", "reused_run": record["run_url"]}
    if event == "workflow_dispatch":
        scope = environment.get("SCOPE") or "full"
        if scope not in SCOPES:
            raise ValueError(f"unknown scope: {scope}")
        return {"tier": "full" if scope == "full" else "targeted", "scope": scope}
    return {"tier": "full", "scope": "full"}


def find_reusable_result(sha, repository, api=_api, download=_download_result, tree=_git_tree):
    """Find a successful full pull request run that tested the landed tree.

    :param sha: Pushed ``master`` commit.
    :param repository: ``owner/name`` repository.
    :param api: GitHub REST reader, replaceable in tests.
    :param download: Full-result artifact reader, replaceable in tests.
    :param tree: Tree reader for the checked-out push commit, replaceable in tests.
    :return: Matching record with ``run_url`` added, or ``None`` when proof is missing or the
        trees differ.
    """
    landed = tree("HEAD")
    for pull in api(f"repos/{repository}/commits/{sha}/pulls"):
        if pull.get("base", {}).get("ref") != "master" or not pull.get("merged_at"):
            continue
        head = pull["head"]["sha"]
        runs = api(f"repos/{repository}/actions/workflows/ci.yml/runs?event=pull_request&head_sha={head}"
                   "&status=success&per_page=20")["workflow_runs"]
        for run in sorted(runs, key=lambda item: item["created_at"], reverse=True):
            if run.get("conclusion") != "success":
                continue
            record = download(run["id"], repository)
            if (record and record.get("tier") == "full" and record.get("tree") == landed
                    and record.get("head_sha") == head and record.get("pull_request") == pull["number"]):
                return dict(record, run_url=run["html_url"])
    return None


def main():
    """Print the plan as ``key=value`` lines for ``GITHUB_OUTPUT``.

    :return: None; also appends a short explanation to the job summary when available.
    """
    plan = choose(os.environ, find_reusable_result)
    for key, value in plan.items():
        print(f"{key}={value}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            if plan["tier"] == "reused":
                handle.write(f"Reused the full result of {plan['reused_run']} for an identical tree.\n")
            else:
                handle.write(f"CI tier: {plan['tier']}" + (f", scope {plan['scope']}" if plan["scope"] else "")
                             + ".\n")
    print(json.dumps(plan), file=sys.stderr)


if __name__ == "__main__":
    main()
