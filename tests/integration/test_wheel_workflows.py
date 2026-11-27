"""Validate required certification rows through the workflow matrix data."""

import unittest
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


# ---- continuous integration workflow tests ----

def test_built_wheels_all_receive_cpp_payload_checks_before_aggregation():
    workflow = yaml.load((ROOT / '.github/workflows/wheel-validation.yml').read_text(), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['check-distributions']['steps']
    checks = [(index, step) for index, step in enumerate(steps)
              if 'tools/check_sdist.py --wheel' in step.get('run', '')]
    assert len(checks) == 1, 'Every produced native and pure wheel needs the no-copy payload check'
    index, step = checks[0]
    assert step['shell'] == 'bash'
    assert 'for wheel in dist/*.whl' in step['run']
    assert '"$wheel"' in step['run']
    metadata_index = next(i for i, entry in enumerate(steps) if 'tools/check_wheel_metadata.py' in entry.get('run', ''))
    assert index < metadata_index


def test_ci_manual_dispatch_never_joins_or_cancels_the_push_group():
    concurrency = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)['concurrency']
    push = "github.event_name == 'push' && github.ref == 'refs/heads/pre_release'"
    assert push + " && github.ref" in concurrency['group']
    assert concurrency['cancel-in-progress'] == "${{ github.event_name == 'pull_request' || (" + push + ") }}"


def test_ci_tiers_come_from_the_plan_job():
    workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    jobs = workflow['jobs']
    assert workflow['on']['push']['branches'] == ['pre_release', 'master']
    plan = jobs['plan']
    assert plan['steps'][-1]['run'] == 'python tools/ci_plan.py >> "$GITHUB_OUTPUT"'
    assert plan['permissions'] == {'contents': 'read', 'actions': 'read', 'pull-requests': 'read'}
    quick = "${{ contains(fromJSON('[\"quick\",\"full\"]'), needs.plan.outputs.tier) }}"
    for name in ('style', 'rust', 'python-only', 'native-pytest'):
        assert jobs[name]['needs'] == 'plan' and jobs[name]['if'] == quick, name
    for name in ('msrv', 'crate-validation'):
        assert jobs[name]['if'] == "${{ needs.plan.outputs.tier == 'full' }}", name
    wheels = jobs['wheel-validation']
    assert 'contains(fromJSON(\'["full","targeted"]\'), needs.plan.outputs.tier)' in wheels['if']
    assert "!contains(needs.*.result, 'failure')" in wheels['if']
    assert wheels['with']['scope'] == '${{ needs.plan.outputs.scope }}'
    record = jobs['full-result']
    assert set(record['needs']) == {'plan', 'style', 'rust', 'msrv', 'python-only', 'native-pytest',
                                    'crate-validation', 'wheel-validation'}
    assert record['if'].startswith("${{ success() && github.event_name == 'pull_request'")
    upload = next(step for step in record['steps'] if step.get('uses') == 'actions/upload-artifact@v4')
    assert upload['with']['name'] == 'ci-full-result'
    text = (ROOT / '.github/workflows/ci.yml').read_text()
    assert text.count('github.base_ref') == 1, 'Only the plan job reads the base branch; tiers come from ci_plan.py'


def test_manual_cpp_calibration_uses_the_opt_in_live_lane():
    workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs']['cpp-calibration']
    assert 'cpp_live' not in workflow['on']['workflow_call']['inputs']
    assert workflow['on']['workflow_call']['inputs']['tier'] == {'type': 'string', 'default': 'full'}
    assert workflow['on']['workflow_dispatch']['inputs']['cpp_qualification']['default'] == 'false'
    manual = ("github.event_name == 'workflow_dispatch' && needs.plan.outputs.tier == 'full' && "
              "inputs.cpp_qualification == true")
    assert job['if'] == "${{ " + manual + " }}"
    release = yaml.load((ROOT / '.github/workflows/release.yml').read_text(), Loader=yaml.BaseLoader)
    assert 'cpp_live' not in release['jobs']['validation'].get('with', {})
    shared = './.github/workflows/cpp-live-validation.yml'
    assert job['uses'] == shared
    crate = yaml.load((ROOT / '.github/workflows/release-crate.yml').read_text(), Loader=yaml.BaseLoader)
    assert 'cpp-live' not in crate['jobs']
    assert set(crate['jobs']['publish-crate']['needs']) == {'verify'}
    assert crate['jobs']['publish-crate']['if'] == 'inputs.publish'
    lane = yaml.load((ROOT / shared).read_text(), Loader=yaml.BaseLoader)
    assert 'workflow_call' in lane['on']
    job = lane['jobs']['live']
    live_steps = [step for step in job['steps'] if 'pytest tests/cpp_live -m cpp_live' in step.get('run', '')]
    live, = live_steps
    assert live['env']['BEHAVE_CPP_LIVE'] == '1'
    assert live['env']['BEHAVE_CPP_LIVE_COMPILER'] == '${{ matrix.compiler }}'
    assert live['env']['BEHAVE_CPP_LIVE_OUT'] == '${{ runner.temp }}/cpp-${{ matrix.compiler }}'
    assert {(row['os'], row['compiler']) for row in job['strategy']['matrix']['include']} == {
        ('ubuntu-24.04', 'GNU'), ('windows-2025', 'MSVC')}
    checkout = next(step for step in job['steps'] if step.get('uses') == 'actions/checkout@v4')
    assert checkout['with']['submodules'] == 'recursive'
    for name, ordinary in workflow['jobs'].items():
        if name != 'cpp-calibration':
            for step in ordinary.get('steps', []):
                if step.get('uses') == 'actions/checkout@v4':
                    assert step.get('with', {}).get('submodules', 'false') == 'false'
    assert workflow['jobs']['wheel-validation']['with']['cpp_calibration'] == "${{ " + manual + " }}"


def test_maturin_action_uses_only_one_compatibility_setting():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/wheel-validation.yml").read_text())
    steps = workflow["jobs"]["build-native-wheels"]["steps"]
    build = next(step["with"] for step in steps if step.get("uses") == "PyO3/maturin-action@v1")
    assert build["manylinux"] == "${{ matrix.compatibility }}"
    assert "--compatibility" not in build["args"]


def test_musl_wheel_links_the_gcc_runtime_statically():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["build-musl-wheel"]
    steps = job["steps"]
    names = [step.get("name", "") for step in steps]
    link = names.index("Link the GCC runtime statically into the musl extension")
    check = names.index("Require a musl wheel without a GCC runtime shared library")
    build = next(index for index, step in enumerate(steps) if step.get("uses") == "PyO3/maturin-action@v1")
    assert link < build < check
    assert steps[build]["with"]["target"] == "x86_64-unknown-linux-musl"
    assert steps[build]["with"]["manylinux"] == "musllinux_1_2"
    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    assert upload["with"]["name"] == "distributions-linux-musl-x86_64"
    assert workflow["jobs"]["test-alpine-platform"]["needs"] == ["build-musl-wheel", "build-pure-wheel"]
    assert "INPUT(-lgcc_eh)" in steps[link]["run"]
    assert "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS=-C link-arg=-static-libgcc" in steps[link]["run"]
    assert "NEEDED.*libgcc_s" in steps[check]["run"]
    assert "libgcc_s[^/]*\\.so" in steps[check]["run"]


def test_native_build_matrix_covers_every_tested_platform():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["build-native-wheels"]
    rows = {row["artifact"]: row for row in job["strategy"]["matrix"]["include"]}
    # The musl wheel has its own job (build-musl-wheel) so targeted runs can build it alone.
    assert set(rows) == {"linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-arm64", "windows-x86_64",
                         "windows-arm64"}
    assert "build-musl-wheel" in workflow["jobs"]
    for platform in ("linux-x86_64", "linux-aarch64"):
        assert rows[platform]["compatibility"] == "2_17"
    assert rows["windows-arm64"]["os"] == "windows-11-arm"
    assert rows["windows-arm64"]["target"] == "aarch64-pc-windows-msvc"
    assert rows["windows-arm64"]["architecture"] == "arm64"
    setup = next(step["with"] for step in job["steps"] if step.get("uses") == "actions/setup-python@v6")
    assert setup["architecture"] == "${{ matrix.architecture }}"


def test_new_platforms_require_full_native_suites_and_keep_isolated_installs():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    for name in ("test-alpine-platform", "test-windows-arm64-platform"):
        job = workflow["jobs"][name]
        checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
        assert "sparse-checkout" not in checkout.get("with", {}), "full suites require source and fixtures"
        commands = [step["run"] for step in job["steps"] if "run" in step]
        assert any("test_native_wheel_suite.py" in command for command in commands)
        assert any("test_wheel_install.py" in command for command in commands)
        full = next(step for step in job["steps"] if "test_native_wheel_suite.py" in step.get("run", ""))
        assert full["if"] == "matrix.python == '3.12'"
        assert full.get("continue-on-error", "false") == "false"
        uploads = [step for step in job["steps"] if step.get("uses") == "actions/upload-artifact@v4"]
        evidence = next(step for step in uploads if "full-suite" in step["with"]["path"])
        assert evidence["if"] == "always()"


def test_previous_fallback_hosts_require_native_choice():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    for job in ("test-alpine-platform", "test-windows-arm64-platform"):
        steps = workflow["jobs"][job]["steps"]
        install = next(step["run"] for step in steps if "test_wheel_install.py --wheels" in step.get("run", ""))
        assert "--expect-choice rust" in install


def test_pypi_publication_requires_a_tag_push():
    release = yaml.load((ROOT / '.github/workflows/release.yml').read_text(), Loader=yaml.BaseLoader)
    jobs = release['jobs']
    # A manual dispatch on a tag ref also has refs/tags/v*; only the tag push may publish to PyPI.
    assert jobs['publish-pypi']['if'] == ("github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v') "
                                          "&& vars.PYPI_PUBLISHING_ENABLED == 'true'")
    assert jobs['publish-testpypi']['if'] == ("github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi' "
                                              "&& vars.TESTPYPI_PUBLISHING_ENABLED == 'true'")
    assert jobs['publish-pypi']['environment']['name'] == 'pypi'


def test_reusable_ci_callers_grant_the_permissions_its_jobs_request():
    called = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    requested = {}
    for job in called['jobs'].values():
        for scope, level in job.get('permissions', {}).items():
            requested[scope] = 'write' if 'write' in (level, requested.get(scope)) else level
    assert requested == {'contents': 'read', 'actions': 'read', 'pull-requests': 'read'}
    callers = 0
    for path in (ROOT / '.github/workflows').glob('*.yml'):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for name, job in workflow['jobs'].items():
            if job.get('uses') != './.github/workflows/ci.yml':
                continue
            callers += 1
            granted = job.get('permissions', workflow.get('permissions', {}))
            for scope, level in requested.items():
                assert granted.get(scope) in ({level, 'write'} if level == 'read' else {'write'}), (path.name, name)
    assert callers == 1


def test_publication_does_not_require_cpp_evidence():
    # C++ evidence is produced when the pinned upstream C++ changes, so no publisher waits on it.
    expected = {'release.yml': {'publish-testpypi': {'validation', 'release-policy'},
                                'publish-pypi': {'validation', 'release-policy'}},
                'release-crate.yml': {'publish-crate': {'verify'}}}
    for name, uploads in expected.items():
        text = (ROOT / '.github/workflows' / name).read_text()
        assert 'receipt' not in text and 'qualification' not in text and 'cpp-live' not in text, name
        jobs = yaml.load(text, Loader=yaml.BaseLoader)['jobs']
        for upload, needs in uploads.items():
            assert set(jobs[upload]['needs']) == needs, (name, upload)
    assert not (ROOT / '.github/workflows/cpp-receipt-validation.yml').exists()

def test_rust_only_consumer_fails_when_python_is_installed():
    jobs = yaml.load((ROOT / '.github/workflows/crate-validation.yml').read_text(), Loader=yaml.BaseLoader)['jobs']
    runs = [step.get('run', '') for job in jobs.values() for step in job.get('steps', [])
            if 'command -v' in step.get('run', '')]
    script, = runs
    # A negated command is exempt from `set -e`, so `! command -v python` cannot fail the step.
    assert '! command -v' not in script
    assert 'if command -v python || command -v python3; then' in script
    assert script.index('exit 1') < script.index('mkdir standalone')


def test_wheel_validation_scopes_select_targeted_jobs():
    workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    assert workflow["on"]["workflow_call"]["inputs"]["scope"]["default"] == "full"
    assert workflow["on"]["workflow_dispatch"]["inputs"]["scope"]["options"] == ["full", "musl", "sdist",
                                                                                  "musl-and-sdist"]
    musl = "${{ contains(fromJSON('[\"full\",\"musl\",\"musl-and-sdist\"]'), inputs.scope) }}"
    sdist = "${{ contains(fromJSON('[\"full\",\"sdist\",\"musl-and-sdist\"]'), inputs.scope) }}"
    full = "${{ inputs.scope == 'full' }}"
    expected = {"build-pure-wheel": musl, "build-musl-wheel": musl, "test-alpine-platform": musl,
                "build-sdist": sdist, "build-native-wheels": full, "test-native-platforms": full,
                "test-windows-arm64-platform": full, "check-distributions": full}
    assert {name: job.get("if") for name, job in jobs.items()} == expected


def test_wheels_require_the_quick_style_gate():
    workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    jobs = workflow['jobs']
    assert 'style' in jobs, 'Dirty Python/order/Rust style must fail quick and release CI'
    style = jobs['style']
    assert style['if'] == jobs['python-only']['if'], 'Style must run in the quick tier and reusable CI'
    assert 'style' in jobs['wheel-validation']['needs'], 'Wheels must wait for successful style checks'
    assert style.get('continue-on-error', 'false') == 'false'
    commands = [step['run'] for step in style['steps'] if 'run' in step]
    assert any('python -m ruff check src tests tools' in command for command in commands)
    assert 'uv run --no-sync python tools/check_function_order.py --require-labels' in commands
    assert any('cargo fmt --manifest-path rust/Cargo.toml --all -- --check' in command for command in commands)
    assert any('cargo clippy --manifest-path rust/Cargo.toml' in command and '-D warnings' in command
               for command in commands)
    assert all(step.get('continue-on-error', 'false') == 'false' for step in style['steps'])
    assert not any(token in command for command in commands for token in ('--ignore', '--extend-exclude', '|| true'))


class WorkflowTests(unittest.TestCase):
    # ---- reusable workflow certification tests ----

    def test_all_requested_certification_rows_remain_required(self):
        workflow = yaml.load((ROOT / ".github/workflows/wheel-validation.yml").read_text(), Loader=yaml.BaseLoader)
        jobs = workflow["jobs"]
        native = jobs["test-native-platforms"]["strategy"]["matrix"]
        rows = {(target["artifact"], python) for target, python in product(native["platform"], native["python"])}
        expected = set(product(("linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-arm64", "windows-x86_64"), (
            "3.11", "3.12", "3.13", "3.14")))
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
