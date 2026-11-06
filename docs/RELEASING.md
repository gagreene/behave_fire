# Python package release procedure

Python package versions come from `rust/crates/behave-py/Cargo.toml`. Cargo `1.0.0-beta.2` maps to Python `1.0.0b2` and Python tag `v1.0.0b2`. The standalone core version (`rust/crates/behave-fire/Cargo.toml`, currently `0.1.0`) is independent. Never bump the core solely to match Python.

Build-only validation is safe on an unmerged branch. PR CI calls `.github/workflows/wheel-validation.yml` through a local reusable-workflow reference, so the branch's own scripts and workflow run before merge. The registered CI workflow also accepts manual dispatch on a selected ref; it builds and tests artifacts without publishing. Manual `release.yml` dispatch defaults to `build-only`. No package upload is authorized by implementing or testing these files.

## Build wheels on GitHub

Push the implementation branch and open its PR to run CI, or dispatch the registered CI workflow on a ref that contains `workflow_dispatch`:

```sh
gh workflow run CI --ref pre_release
gh run list --workflow CI --branch pre_release --event workflow_dispatch --limit 5
gh run watch RUN_ID --exit-status
gh run download RUN_ID --name distributions --dir dist
python tools/check_wheel_metadata.py dist
```

Replace `RUN_ID` with the run reported by GitHub. Use a feature branch instead of `pre_release` when validating changes before merge. CI also runs on pushes to `pre_release` and `master`; its workflow can be rerun through GitHub Actions for an existing commit. The manual command runs the full CI pipeline, including all seven native wheel targets, pure wheel, sdist, isolated installation tests and Rust/package checks. It has read-only repository permissions and no publication job.

Use the `distributions` artifact from a successful run; individual `distributions-*` artifacts are intermediate build outputs. The combined artifact contains seven `cp311-abi3` native wheels, one universal `py3-none-any` wheel and one source archive. Test evidence is uploaded separately per platform/interpreter. All installation tests run outside the source checkout on CPython 3.11–3.14, with both the exact NumPy floor and a current allowed version. On every platform the complete artifact set selects Rust; a separate pure-only install selects Python silently.

The GitHub UI's Run workflow button depends on the workflow being available on the default branch. These build commands use the existing registered CI workflow; they do not require changing the default branch, promoting unreleased code into `master`, creating a release tag or enabling either publisher. The first validation of a newly added workflow remains its PR CI run.

## Local dependency-only source tests

Install uv, then run from the checkout:

```sh
uv sync --locked --no-install-project --extra test --extra dev
uv run --no-sync python -c 'import importlib.util,sys; import pytest,build,twine; sys.path.insert(0,"src"); assert importlib.util.find_spec("behave_fire._rust") is None'
uv run --no-sync python -m pytest
```

The test extra includes pytest and PyYAML for the collected regression and workflow tests, without raster libraries. The examples extra is optional and its dependency-wheel availability is outside the base-package compiler-free claim. `uv run --no-sync` prevents an implicit editable project install. Native development uses `uv sync --locked --extra test --extra dev` with Rust installed, followed by `uv run --no-sync python -m pytest --require-rust`.

## Build and inspect artifacts

Use a clean authorized release commit. All Rust manifests, sources, embedded tables, and notices must be tracked by Git before a git-mode sdist build. `git ls-files` includes staged new files; the tracking gate does not itself certify a clean release commit. Do not commit to bypass a packaging check without permission.

```sh
uv run --no-sync python tools/check_sdist.py --check-tracked .
uv build --sdist --out-dir dist
uv build --wheel --out-dir dist
uv run --no-sync python tools/build_pure_wheel.py --out-dir dist
uv run --no-sync python tools/check_sdist.py dist/behave_fire-1.0.0b2.tar.gz
uv run --no-sync python -m twine check dist/*
```

These local commands build the current host's native wheel, the universal pure wheel and the source distribution. They do not build or certify the complete platform set. Run `tools/check_wheel_metadata.py` on the assembled CI artifact: it requires exactly one native wheel per tested platform family, one universal pure wheel and one matching source distribution. It rejects missing/duplicate families, incorrect interpreter/ABI/platform tags, mismatched project versions, metadata or license payloads. Equivalent manylinux alias tags count as one wheel.

Maturin builds the native wheels using `rust/crates/behave-py/Cargo.toml`, PyO3 `abi3-py311`, and `src/behave_fire`. Source/git/editable/`--no-binary` project builds require Rust and a native linker. The pure builder uses a temporary setuptools project and never changes root metadata or invokes Rust. The pure wheel contains the same Python sources, project metadata, license expression, and notices, without a native extension or Rust sources.

The universal project wheel itself requires no Rust compiler. A complete compiler-free install also requires binary NumPy availability for the particular interpreter/platform. Use `pip install --only-binary=:all: <wheel>` to enforce that condition. Project-wheel selection and dependency resolution are separate checks: the validation runner preinstalls binary NumPy, selects the exact project version using `--no-deps --no-index --find-links`, and then repeats a complete binary-only install in another fresh venv. Both native and pure artifacts run with `python -I` outside the checkout. Never infer support from a wheel filename alone.

Extract the sdist into an external directory without `.git`, the original repository, or sibling checkouts, then build and install its wheel there. CI performs this step and checks both crates' source inventory. A release also runs from a clean authorized commit, so ignored or untracked files cannot fill archive gaps.

## Certification matrix

Every row runs CPython 3.11, 3.12, 3.13, and 3.14. All rows remain **unverified until their real CI jobs pass**. No PyPy, free-threaded CPython (`3.13t`/`3.14t`), or optional-extra support is implied. ABI3 wheels do not cover free-threaded interpreters.

| Target | Project selected | Exact NumPy floor | Required evidence |
| --- | --- | --- | --- |
| manylinux 2_17 x86-64 | native abi3 | 2.1.3 (3.11-3.13); 2.3.3 (3.14) | native plus pure checks |
| manylinux 2_17 aarch64 | native abi3 | same | native plus pure checks |
| macOS x86-64 | native abi3 | same | native plus pure checks |
| macOS arm64 | native abi3 | same | native plus pure checks |
| Windows x86-64 | native abi3 | same | native plus pure checks |
| Alpine/musl x86-64 | native musllinux 1_2 abi3 | same | native plus pure checks, measured musl identity, container Python |
| Windows ARM64 | native win_arm64 abi3 | 2.3.3 (all four Pythons) | native plus pure checks, `platform_machine == 'ARM64'`, `win_arm64` pip tags |

Each row tests the exact applicable floor and a current resolver-selected binary release separately, logs interpreter/platform/package provenance and dependency versions, and uploads pip installation reports. Floor metadata uses mutually exclusive markers, including Python 3.14 on Windows ARM64. Alpine jobs pin Docker Hub manifest digests resolved during implementation and force `linux/amd64`; JavaScript actions run on the Ubuntu host. An unavailable runner or missing binary dependency leaves its row unverified. Failing rows block artifact aggregation and publishing. Never remove a row, tolerate its failure, or substitute a source dependency build. Omitting a requested target requires an explicit release-scope revision.

Runner labels and setup-python ARM64 support were checked against [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) and [actions/setup-python](https://github.com/actions/setup-python). Recheck platform availability when certifying a release.

## Owner-only publisher setup

1. Confirm ownership of `behave-fire` on PyPI and TestPyPI, or register the pending project with its correct owner. Project ownership and namespace registration require owner approval.
2. Create protected GitHub environments `testpypi` and `pypi`, require owner reviewers, disable self-review as appropriate, and restrict `pypi` deployment to approved `v*` tags.
3. Register trusted publishers on each index: GitHub owner `gagreene`, repository `behave_fire`, workflow `release.yml`, and the matching environment (`testpypi` or `pypi`). See [PyPI trusted publisher documentation](https://docs.pypi.org/trusted-publishers/).
4. After explicit owner authorization, enable only the intended repository variable: `TESTPYPI_PUBLISHING_ENABLED=true` or `PYPI_PUBLISHING_ENABLED=true`. These switches default to disabled when absent and prevent an unconfigured tag from publishing. The protected environment remains the final approval gate.
5. Use an explicit manual `target=testpypi` dispatch only after its own authorization. A default manual dispatch builds and validates without publishing. Verify uploaded TestPyPI artifacts separately when upload is authorized.
6. Follow `docs/CONTRIBUTION_PROCEDURES.md`: version-bump PR into `pre_release`, promotion PR into `master`, full Linux/macOS/Windows checks, then an explicitly authorized annotated Python tag on an approved commit reachable from `master`. The workflow checks annotation, ancestry, and exact Python-version spelling. Tag creation and pushing each require permission.
7. Production publication requires the approved tag, every CI/package/wheel/MSRV gate, the enabled production switch, and protected-environment approval. No release can omit a requested certification row without explicit scope approval.

Crates.io publication is separate (`release-crate.yml`, independent `behave-fire-vX.Y.Z` tags and core version). Standalone archive validation is a required CI gate and therefore also gates Python release validation. Publishing the crate is not a prerequisite for Python publication: Python source archives bundle both crates and build through their relative path dependency. See [standalone crate verification](CRATES_IO.md) for package, Rust-only consumer, and separately authorized registry steps.
## Build backend minimum

The supported backend range is `maturin>=1.10,<2`. Maturin's [primary changelog](https://www.maturin.rs/changelog) records full PEP 639 support in 1.9.0, the `License-Expression` correction in 1.9.2, and sdist license inclusion fixes in 1.9.3. A direct Windows CPython 3.12 metadata build with 1.9.3 still failed while creating `licenses/LICENSES/NumPy-BSD-3-Clause.txt`; the identical operation with 1.10.0 succeeded and preserved all notice bytes. Native CI pins 1.10.0 to exercise this tested minimum, while isolated source builds resolve the supported 1.x range.
