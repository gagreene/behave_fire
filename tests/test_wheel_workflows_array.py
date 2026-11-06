"""Validate required certification rows through the workflow matrix data."""

from itertools import product
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).parents[1]


def test_maturin_action_uses_only_one_compatibility_setting():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/wheel-validation.yml").read_text())
    steps = workflow["jobs"]["build-native-wheels"]["steps"]
    build = next(step["with"] for step in steps if step.get("uses") == "PyO3/maturin-action@v1")
    assert build["manylinux"] == "${{ matrix.compatibility }}"
    assert "--compatibility" not in build["args"]


def test_native_build_matrix_covers_every_tested_platform():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["build-native-wheels"]
    rows = {row["artifact"]: row for row in job["strategy"]["matrix"]["include"]}
    assert set(rows) == {"linux-x86_64", "linux-aarch64", "linux-musl-x86_64", "macos-x86_64",
                         "macos-arm64", "windows-x86_64", "windows-arm64"}
    assert rows["linux-musl-x86_64"]["target"] == "x86_64-unknown-linux-musl"
    assert rows["linux-musl-x86_64"]["compatibility"] == "musllinux_1_2"
    for platform in ("linux-x86_64", "linux-aarch64"):
        assert rows[platform]["compatibility"] == "2_17"
    assert rows["windows-arm64"]["os"] == "windows-11-arm"
    assert rows["windows-arm64"]["target"] == "aarch64-pc-windows-msvc"
    assert rows["windows-arm64"]["architecture"] == "arm64"
    setup = next(step["with"] for step in job["steps"] if step.get("uses") == "actions/setup-python@v6")
    assert setup["architecture"] == "${{ matrix.architecture }}"


def test_previous_fallback_hosts_require_native_choice():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    for job in ("test-alpine-platform", "test-windows-arm64-platform"):
        steps = workflow["jobs"][job]["steps"]
        install = next(step["run"] for step in steps if "test_wheel_install.py --wheels" in step.get("run", ""))
        assert "--expect-choice rust" in install


class WorkflowTests(unittest.TestCase):
    def test_all_requested_certification_rows_remain_required(self):
        workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
        jobs = workflow["jobs"]
        native = jobs["test-native-platforms"]["strategy"]["matrix"]
        rows = {(target["artifact"], python) for target, python in product(native["platform"], native["python"])}
        expected = set(product(("linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-arm64", "windows-x86_64"), ("3.11", "3.12", "3.13", "3.14")))
        self.assertEqual(rows, expected)
        alpine = jobs["test-alpine-platform"]["strategy"]["matrix"]
        self.assertEqual({row["python"] for row in alpine["include"]}, {"3.11", "3.12", "3.13", "3.14"})
        arm = jobs["test-windows-arm64-platform"]["strategy"]["matrix"]
        self.assertEqual(set(arm["python"]), {"3.11", "3.12", "3.13", "3.14"})
        for job in ("test-native-platforms", "test-alpine-platform", "test-windows-arm64-platform"):
            self.assertNotIn("continue-on-error", jobs[job])
            self.assertNotIn("exclude", jobs[job]["strategy"]["matrix"])
            self.assertIn(job, jobs["check-distributions"]["needs"])

    def test_reusable_branch_workflows_resolve_without_default_branch(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
            for job in workflow["jobs"].values():
                if job.get("uses", "").startswith("./"):
                    target = ROOT / job["uses"]
                    self.assertTrue(target.is_file(), target)
                    called = yaml.load(target.read_text(), Loader=yaml.BaseLoader)
                    self.assertIn("workflow_call", called["on"])


if __name__ == "__main__":
    unittest.main()
