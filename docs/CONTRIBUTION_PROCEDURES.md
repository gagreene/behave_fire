# Contribution procedures

These procedures apply to changes proposed for behave_fire. The root [`README.md`](../README.md) describes the
package and how to run its tests.

## Function organization

- Organize functions alphabetically in every code file, in every language (Python, Rust, and any other), to make them easier to locate in an IDE. This covers top-level functions and the methods within a class, `impl` block, or equivalent.
- Python: group dunder methods first, then other underscore names, then public names. Alphabetize each group case-insensitively by the full name.
- Rust: group non-`pub` functions first, then `pub` functions, alphabetically by name within each group. In other languages, group private functions before public functions and alphabetize each group by the bare name.
- Apply the same organization to production modules and test modules.
- Preserve a non-alphabetical order only when a framework, runtime dependency, or the idiomatic conventions of the language require it, and document the reason in the file. In Rust, that includes `use` and `mod` declarations, a trailing `#[cfg(test)] mod tests` block, trait-implementation method order set by the trait, and items whose declaration order matters, such as macros defined before use.

## Function signatures and docstrings

- Keep short function signatures on one line.
- When a signature must wrap, put the opening parenthesis after the function name, place one parameter per line with an eight-space continuation indent, and align the closing parenthesis with `def`.
- Give every production function and method a docstring, including private, nested, and `None`-returning helpers.
- Use reStructuredText fields: a concise summary, a blank line, one `:param name:` entry per input in signature order, and a final `:return:` entry.
- Indent wrapped `:param` and `:return:` descriptions by eight spaces, and describe side effects in `:return:` when a function writes files instead of returning a value.
- Apply these rules to first-party production code. Do not reformat vendored or generated code solely to enforce this preference.

## Rust and native backend

- The Cargo workspace is in `rust/` (core crate `behave-fire`, unpublished PyO3 bindings `behave-py`); there is no `Cargo.toml` at the repository root, so pass `--manifest-path rust/Cargo.toml`. Build, test, and lint commands appear below under "Local tests before every PR merge".
- The Python implementation under `src/behave_fire/components/` is the behavioral reference. Rust must match it within the documented tolerances; do not change expected values to make a change pass, and treat regenerating fixtures (`tools/export_rust_reference.py`) as an explicit, reviewed action. A suspected Python defect needs an explicit fix decision; Rust results must not silently differ from Python.
- Keep the `behave-fire` core free of normal and build dependencies and of any Python, NumPy, or C++ requirement. Record intentional Rust/Python differences in `docs/BACKEND_DIFFERENCES.md`.
- Never publish to PyPI or crates.io, create release tags, or enable publishing switches without the user's direct permission. See `docs/RELEASING.md` and `docs/CRATES_IO.md`.

## Local-only files and staging

- Plans, ledgers, scratch files, PR-description drafts, agent instruction files (`AGENTS.md`, `.claude/`, skill files), session logs, `development/`, `.superpowers/`, `*.pdb`, and `rust/target/` are never committed or pushed.
- `.claude/` is listed in `.gitignore`. `AGENTS.md` is not tracked and is excluded locally through `.git/info/exclude`, which lives in the common git directory. Every linked worktree of a clone shares it, but a fresh clone does not inherit it, so add `AGENTS.md` and `.claude/` to `.git/info/exclude` once per clone. A worktree created from a branch that does not track `AGENTS.md` has no copy of the guidance file, so bootstrap the local guidance files per worktree.
- Stage explicit paths. Never use `git add -A`, `git add .`, or `git commit -a`.
- Check `git diff --cached --name-only` before every commit and `git diff --name-only <base> HEAD` after it. Stop and report if a local-only file appears.

### Fresh clones and linked worktrees

Set up the exclusions once per clone, using `git rev-parse --git-path info/exclude`
to locate the common exclusion file. Add `AGENTS.md` and `.claude/` on separate
lines if they are absent. Linked worktrees share that exclusion file.

Bootstrap local guidance separately in each worktree: create an untracked
`AGENTS.md` and, when used, `.claude/CLAUDE.md` with a short pointer to these
procedures. Preserve personal instructions locally. Do not copy personal paths,
session-context settings, or personal e-mail addresses into tracked documents.

## Branches, commits, and pushes

1. Create a focused work branch from its intended PR base under the merge
   routing below. Keep unrelated edits, generated results, temporary files,
   and local reference data out of the change. Check the staged diff before
   committing.
2. Use commits that describe the actual change. Do not add AI authorship or
   attribution statements to commit messages, PR titles, or PR descriptions.
   All commits must be verified and authored by a human. Use the GitHub noreply
   address `127547449+gagreene@users.noreply.github.com` for commits, tags,
   and package metadata, never a personal e-mail address. Before the first commit,
   check `git config user.email`; stop and tell the user if it differs.
3. Coding agents must obtain the user's **direct permission** before creating
   or amending any commit. They must obtain **separate direct permission** before
   pushing a commit, branch, or tag. Permission to edit, test, stage, prepare a
   PR, or continue work does not grant permission to commit or push. Do not
   automate commits or pushes without explicit user authorization for each action.
   After preparing and validating changes, stop and report what is ready, then
   ask separately before committing and before pushing. Never otherwise modify
   an existing commit without the user's direct permission.

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

### CI tiers

The `CI` workflow (`.github/workflows/ci.yml`) runs in two tiers:

- **Quick tier**, for a pull request that does not target `master` (including each
  upper layer of a stack): `cargo fmt --check`, Clippy, the Rust core tests and the
  documentation build on one operating system, `python-only` on one Python
  version, and `native-pytest` on one Linux job. The MSRV, wheel, and crate
  validation jobs are skipped.
- **Full tier**, for a pull request into `master`, every push to `pre_release` and
  `master`, a call from `release.yml`, and a manual `workflow_dispatch` on any
  branch: every job and every matrix row, including the wheel and crate validation.
  The wheel validation starts only after the other jobs have passed.

A newer run for the same pull request, or for `pre_release`, cancels the older one,
so a burst of merges ends with one full run on the last commit. Every other run
(`master`, tags, manual dispatch) has its own concurrency group, so it is never
cancelled and never replaces a queued run, because each `master` run gates a
release. Changing a pull request's base branch re-runs the workflow, so retargeting
a pull request to `master` runs the full tier; an edit to its title or description
starts a run in which every job is skipped. A wheel or platform failure on `pre_release`
is fixed promptly, because it can block the next pull request. The quick tier does
not replace the local full suite and the test record above, which remain required
for every PR.

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


## C++ parity

The coverage unit is hybrid (owner decision 11, 2026-11-13). Every public C++ method belongs to a route bundle, and every scientific candidate with Python or Rust targets also has a per-function status. State and lifecycle methods are exercised through bundle contracts. Route assignment and source-derived candidate mappings are not execution evidence. During pilot preparation, `pending_pilot` and `not_in_pilot` explicitly identify work that has no qualified numerical dataset; neither status is a waiver or a claim of tested coverage. The offline inventory check validates the committed snapshot without the C++ checkout. Its `--require-ready` gate rejects a snapshot with no executable scientific route. A tested status requires hashed cases, results, a manifest, and the owner's approved field policy and finding decision.

- Python remains the behavioral reference of this repository. The official upstream C++ in `reference/behave_cpp`, pinned to `f3b2943e69abca942e795c5b1b45276f6338abb5`, is the validation oracle for the operations it can execute safely. Every Python and Rust implementation of Behave scientific logic is compared with it, and every difference is either fixed with the owner's approval or recorded as an owner-approved finding.
- Every public C++ function has an entry in the coverage inventory (`tests/data/cpp_reference/inventory.json`) with its fully qualified signature and a status of tested, waived (owner-approved, with a reason), no counterpart, unreachable oracle, or deferred (owner-approved and time-limited, used for the C++ containment code while the module is out of the repository). A waiver or a deferral needs the owner's approval and a reason. A new or changed ported function needs an inventory entry in the same PR.
- Every output field of every compared operation has a classification in the tolerance policy: exact, relative, absolute, angular, or documented difference.
- Do not loosen a tolerance, widen a comparison, skip a test, or regenerate a golden merely to make a test pass. A golden refresh requires a written reason that names the C++ revision, the driver, the inputs, the build, and the defect in the old baseline.
- Record every intended difference from C++ as a numbered finding in the findings register, and pin the exact difference in a test (both the C++ value and the port value), so that a change on either side fails.
- The driver never sends an ID outside 0 to 255 to C++, because the upstream storage and getters are unchecked for other values. Every operation also declares and enforces its own index, enum, derived-index, and conversion bounds before any C++ call (for example Western Aspen model numbers 1 to 5, and the month, humidity, shading, slope, and time-of-day indices of the fine dead fuel moisture tool). Such cases are recorded as unavailable and are never filled with invented C++ results.
- Change the pinned C++ revision only in a dedicated PR whose purpose is to update behave_fire to a newer upstream version. Do not change the pin as part of unrelated work.
- No scientific test and no automatic job discovers or compares the pinned revision with the upstream default branch. Detecting and adopting newer upstream versions is a manual decision by the owner. Explicit retrieval of pinned dependencies (`uv sync`, wheel installs) and of the reference checkout at its pinned SHA is allowed.
- A PR that changes ported science states the pinned C++ SHA it was checked against in its description, and lists new or changed findings.
- A PR that changes a golden dataset includes the written refresh reason described in `docs/TESTING.md`.
- A PR that changes the pinned C++ revision does only that work: the submodule update, regenerated goldens, ports, and documentation updates.
- Python-facing numerical behavior changes found by a comparison are described plainly in the PR and need the owner's explicit approval.
- Do not edit files inside the pinned C++ checkout. The driver and build files live in the overlay folder.
- No C++ source, notice text, or built driver is copied into any published artifact. The submodule `reference/behave_cpp` and the overlay are excluded from the sdist, wheels, and the crate archive, and a packaging check fails if they appear. The C++ is built as upstream builds it, for local use only, and the Phase 1 audit records the file-level license notices (the upstream repository says public domain, but some files carry author copyright notices and one group declares GPL 2).

## Code style and linting

- Function order: in every code file the checker supports (Python and Rust), private functions come first and functions are alphabetical within each group, including methods in a class or `impl` block. For other languages the rule is a manual review rule. A different order is allowed only where language idiom or an import-time dependency requires it, marked with `# order-exempt: <reason>` (Rust: `// order-exempt: <reason>`); a marker without a reason exempts nothing. Test files have their own intentional order (below), which is an exception to the alphabetical rule for tests.
- Test files: helpers come first in the same order, then the tests in groups that start with a `# ---- <label> ----` comment (Rust: `// ---- <label> ----`). Every test sits in a group, the labels are alphabetical with "other tests" last, and tests keep their order inside a group. Helpers must come before the first test. The check for missing labels is switched on after the `tests/` directory is refactored into the unit, integration, and regression tiers; until then the checker skips only that diagnostic and still enforces helper placement.
- Underscore and dunder ordering: Python: dunder methods first, then other underscore names, then public names, each group alphabetical (case-insensitive, full names). Rust: functions without `pub` first, then `pub` functions. The checker implements the same rule (decided by the owner, 2026-11-12).
- Python lint: `python -m ruff check src tests tools`, configured in `pyproject.toml` with line length 120, source roots `src`, `tests`, and `tools`, and the rule families `E`, `F`, `I`, `W`, `UP`, and `B`, with no ignore list. The ruff version is pinned. The blocking `style` CI job runs in the quick tier and reusable release CI, checks Python lint and function order plus Rust format and strict Clippy, and must pass before wheel validation starts. No ignore list or additional exclusion is enabled; exceptions require owner approval and a reason.
- Rust lint and format: `cargo fmt --manifest-path rust/Cargo.toml --all -- --check` and `cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets --locked -- -D warnings`.
- A reorder is its own commit, separate from lint fixes and from behavior changes. Mechanical commits are verified: the checker compares fully qualified function identities, signatures, decorators and attributes, defaults, and bodies, proves that all other code is unchanged except approved provenance changes, and normalizes only declaration order, and the full suite gives the same results. Under the squash policy the reorder categories (Python production, Rust production, Rust tests) are separate sequential PRs.
- A change to a hashed source (`behave.py` and every `components/*.py`, plus the four files that `tools/check_rust_tables.py` hashes) refreshes both sets of reference provenance, and a change to a function signature or annotation is classified as signature metadata, not as a hash-only change. Run `python tools/export_rust_reference.py --check` and `python tools/check_rust_tables.py --check` before opening the PR.
- Run the style checks against the PR's latest commit and record them in the PR description with the test record.

The checker tests cover controlled snippets and assert that the whole repository passes its own scanner. Run the scanner with `python tools/check_function_order.py`; add `--require-labels` after test labelling. Missing labels remain deferred until then; helper placement and existing label order stay enforced. `--prove-reorder BEFORE AFTER` verifies one Python or Rust file pair. Python ASTs retain signatures, decorators, defaults, bodies, and scope; Rust comparison retains tokens, attributes, string contents, and enclosing impl/module identities. Duplicate same-named definitions retain their relative order. Runtime/import tests and the full suite remain required because structural proof alone cannot establish import-time safety.
