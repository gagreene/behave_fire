# Contribution procedures

These procedures apply to changes proposed for behave_fire. The root [`README.md`](../README.md) describes the
package and how to run its tests.

## Branches, commits, and pushes

1. Create a focused work branch from its intended PR base under the merge
   routing below. Keep unrelated edits, generated results, temporary files,
   and local reference data out of the change. Check the staged diff before
   committing.
2. Use commits that describe the actual change. Do not add AI authorship or
   attribution statements to commit messages, PR titles, or PR descriptions.
   All commits must be verified and authored by a human. Use the GitHub noreply
   address for commits and tags, never a personal e-mail address.
3. Coding agents must obtain the user's **direct permission** before creating
   or amending any commit. They must obtain **separate direct permission** before
   pushing a commit, branch, or tag. Permission to edit, test, stage, prepare a
   PR, or continue work does not grant permission to commit or push. Do not
   automate commits or pushes without explicit user authorization for each action.

## Merge routing and releases

Version numbers use `MAJOR.MINOR.PATCH` (for example, `1.2.3`); pre-releases keep the
`pyproject.toml` form already in use (for example, `1.0.0b2`). Two branches live
permanently, and neither may ever be deleted:

- `master` holds released code only. Every release tag is on `master`.
- `pre_release` is the integration branch for all work toward the next release,
  whatever its size.

Route every change as follows:

| Change | Work branch starts from | PR target |
| --- | --- | --- |
| Standalone change or bottom PR of a next-release stack | `pre_release` | `pre_release` |
| Upper PR in a dependent next-release stack | Preceding stack branch | Preceding stack branch until it lands, then retarget as the stack advances |
| Hotfix for a serious defect in released code | `master` (`hotfix/<name>`) | `master` |

Do not classify a change as a patch, minor, or major change when its PR is opened.
The owner chooses the version when the release is promoted, from the combined
changes: anything that alters inputs, results, or documented behavior is at least a
minor release, and only fixes with no such change make a patch release.

### Stacked pull requests

Dependent PRs may target preceding branches within the same repository. The
bottom PR targets `pre_release`; all PRs in the stack collectively must land on
`pre_release`. An upper PR's temporary base is not a separate integration destination.

Use `gh stack` to manage the stack when available. Land the bottom PR first, then
let the stack workflow rebase and retarget remaining PRs as predecessors merge.
Verify the resulting bases and commit ranges after each transition. Resolve
conflicts before continuing, and preserve the reviewed scope of each layer.

Every PR keeps its own review and test record. A rebase changes its head commit:
rerun the full default suite and update the recorded SHA and results before that
PR merges. Stack management does not grant permission to commit, push, merge,
delete branches, or update either permanent branch directly.

Landing a stack is a separate, explicitly authorized step for each layer. Merge
commits are the project's preference; the installed `gh stack` supports merge,
squash, and rebase methods, so this is a policy choice, not a tool limitation.
To land one layer, use the single-PR command with a head guard, then synchronize:

```shell
gh pr merge <bottom-pr-number> --repo gagreene/behave_fire --merge --match-head-commit <tested-head-sha>
gh stack sync
```

Never use an unqualified `gh stack merge` or `--yes` as a one-layer shortcut: they
can merge several PRs atomically. Do not pass `--prune` to `gh stack sync` (it
deletes branches). After each layer lands, verify that the next PR targets
`pre_release` and that its commit range is unchanged, then rerun its tests.

### Release notes

A next-release PR, including an upper stacked PR, that changes behavior adds its bullet under an `Unreleased`
heading at the top of `CHANGELOG.md` (create the heading if it is absent) and does
not change the package version. Breaking changes, and the action users need to take,
go in a `Breaking changes and action needed` block of the same section.

### Releasing

1. Confirm that `pre_release` holds everything intended for the release and that
   every PR in it has its test record.
2. Open a promotion PR from `pre_release` into `master`. Along with a hotfix PR, it is
   the only PR that changes the version: it bumps `pyproject.toml` and renames the
   `Unreleased` heading of `CHANGELOG.md` to the new version and date. The description
   of the promotion PR carries the test record for the head of `pre_release`.
3. After the owner merges it, run the full test suite on Linux, macOS, and Windows
   before tagging (use the CI matrix when one is configured), then create an
   annotated tag on `master`. Release tags use the `vMAJOR.MINOR.PATCH` form (for
   example, `v1.2.3`).
4. `master` is now one merge commit ahead of `pre_release`: merge `master` back into
   `pre_release` through a PR, as a merge commit, so that the two branches stay equal
   after every release.

### Hotfixes

When a serious defect is found in released code and cannot wait for the next release:

1. Branch `hotfix/<name>` from `master`, fix the defect with a regression test, and
   open a PR into `master`. PRs that are open on `pre_release` are not blocked or
   changed by this.
2. The hotfix PR is its own release: it bumps the patch version and adds a section for
   the new version to `CHANGELOG.md`. After the owner merges it, run the full test
   suite and tag, as for any release.
3. Merge `master` back into `pre_release` through a PR. Resolve any conflict with
   unreleased work in that PR, where the author of the fix is best placed to do it;
   the open feature PRs then only need "Update branch". If `pre_release` already
   holds a different fix for the same defect, keep the correct one and say so in the PR.

## Local tests before every PR merge

Run the full default test suite locally against the **latest PR commit** before
merging. The project builds with Maturin, so `pip install -e .` needs a Rust
toolchain. From the repository root, the Python-only route installs dependencies
without building or installing the project:

```shell
uv sync --locked --no-install-project --extra test --extra dev
uv run --no-sync python -m pytest -vv
```

Changes that touch the native extension or its tests also run the native route,
which builds the extension (Rust required) and fails, rather than skips, if it is
missing or incompatible:

```shell
uv sync --locked --extra test --extra dev
uv run --no-sync python -m pytest --require-rust -vv
```

When a PR changes anything under `rust/`, also run the Rust checks. There is no
`Cargo.toml` at the repository root, so every command passes `--manifest-path`:

```shell
cargo fmt    --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets --locked -- -D warnings
cargo test   --manifest-path rust/Cargo.toml -p behave-fire --locked --no-fail-fast
cargo doc    --manifest-path rust/Cargo.toml --workspace --no-deps --locked
cargo +<MSRV> check --manifest-path rust/Cargo.toml --workspace --all-targets --locked
```

`<MSRV>` is the `rust-version` declared in `rust/Cargo.toml` (currently 1.78). After
any change under `rust/`, rebuild the extension before the native route (a plain
`uv sync` does not notice Rust source changes; use `uv sync --reinstall-package behave-fire`).

Document any skips, failures, or unavailable platform checks; a failed full suite
does not satisfy the passing-test checkbox below. If pytest cannot use its cache
folder, add `-p no:cacheprovider`.

Rerun the suite after changes to the PR branch. Record the command, Python
version, operating system, latest commit SHA, and pass/skip counts in the PR
description. Do not treat a run against an earlier commit as validation of the
current PR head.

## Dependency changes

- `pyproject.toml` is the declaration of the package's dependencies, and
  `requirements.txt` must stay consistent with it.
- Raise a minimum version (a `>=` floor) only after confirming the exact version
  installs on every supported Python version and operating system. Do not rely on a
  dry-run resolution, which can silently pick a newer version.
- A numerical change after an upgrade is a behavior change, not a tooling update.
  Review it as such before merging.

## Pull requests and merge review

- Every branch merge requires a PR, including promotions, hotfixes, and the
  merge of `master` back into `pre_release`. Follow the target branch in the
  merge routing above. Do not merge or fast-forward work branches directly into
  either permanent branch.
- Every PR description must include this checkbox and the test record:

  ```markdown
  - [ ] Full default test suite passed locally against this PR's latest commit.

  Commit SHA:
  Python / OS:
  Command:
  Result (passed / skipped):
  Additional opt-in tests or limitations:
  ```

- Check the box only after the full default suite passes on the latest commit.
  If tests were not run or did not pass, leave it unchecked and explain why.
  **Do not merge a PR while the box is unchecked.** Reviewers must verify the
  recorded SHA and results before merging.
- Describe changes to equations, constants, units, and regression risks in the PR.
  Because the package targets parity with the Behave C++ reference implementation,
  cite the reference source for any changed equation or constant. Add or update
  tests for behave_fire-owned behavior; avoid tests that merely repeat behavior
  owned by external dependencies.

The checkbox is a reviewer-enforced rule, not a GitHub-required status check. A
checked box records the author's local test result; it does not replace review of
the test output or prove that other operating systems passed.

Coding agents must obtain direct permission before pushing release tags.

## Python packaging and source tests

Use the procedures in [RELEASING.md](RELEASING.md) for build-only GitHub runs, downloading the complete seven-platform native wheel set, the universal fallback wheel, and source distribution validation. Every tested OS/architecture has a native wheel, including Alpine/musl x86-64 and Windows ARM64; pure-only installation remains independently tested. Python-only source tests must use `uv sync --locked --no-install-project --extra test --extra dev` and `uv run --no-sync python -m pytest`; never install the project editable in that route. The native test route must run `python -m pytest --require-rust` against a built extension. The test extra contains pytest and PyYAML, without optional raster dependencies. All Python package release tags target approved released commits on `master`; the standalone Rust core has its own version and tag namespace.
