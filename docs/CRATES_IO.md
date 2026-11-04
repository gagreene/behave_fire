# Standalone Rust packaging and publication

`rust/crates/behave-fire` is the independently versioned Rust library. The
`behave-py` crate is private and is never published to crates.io. The core has
no normal or build dependencies, Python callbacks, build-time data generation,
or external runtime tables. Its license is `MIT AND BSD-3-Clause`; the archive
retains the project license, attribution notice, and NumPy kernel license.

The initial core version is `0.1.0`. Publication requires separate approval.
A read-only crates.io query on 2026-11-04 returned no existing `behave-fire`
package. This does not reserve the name or establish ownership. Recheck both
before the first upload. The Python version and its `v*` tags are independent.

## Local artifact verification

Use the Rust minimum declared in `rust/Cargo.toml` to generate the archive's
normalized manifest and lockfile. CI reads that declaration rather than
duplicating the version. From the repository root, run:

```text
cargo package --manifest-path rust/Cargo.toml -p behave-fire --locked
cargo package --list --manifest-path rust/Cargo.toml -p behave-fire --locked
python tools/check_crate_package.py rust/target/package/behave-fire-0.1.0.crate
cargo publish --dry-run --manifest-path rust/Cargo.toml -p behave-fire --locked
```

Run release verification from the exact clean commit being approved. The
checker enforces a 5 MiB compressed budget, embedded tables, notices, compact
fixtures, archive-local includes, and a self-contained normalized manifest.
Cargo's documented registry limit is currently 10 MB; recheck it when releasing.
The repository budget stays smaller. [Cargo publishing documentation](https://doc.rust-lang.org/cargo/reference/publishing.html)

`crate-validation.yml` extracts the archive in a separate Rust container with
no checkout and no Python interpreter. It runs packaged tests and doctests,
the standalone example, and a separate downstream consumer. The consumer uses
default/custom fuels, embedded species, surface fire, mortality, and worker
counts 0/1/2/4. Its dependency points only at the extracted package. Linux
link inspection rejects Python and C++ runtime dependencies. These checks
establish archive independence; they do not prove registry installation.

## First publication

Approve the package name, independent version, release commit, and upload
separately. Complete crates.io account verification and confirm ownership
before obtaining a scoped publication credential. Keep credentials out of
source files, command output, and Git metadata.

The registry's first publication must be bootstrapped before a trusted
publisher can be registered. Use the approved commit and the checks above;
upload only after authorization. The implementation does not perform that
bootstrap. [Rust's trusted publishing announcement](https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07/)

After the first upload, configure the crates.io trusted publisher for
`gagreene/behave_fire`, workflow `release-crate.yml`, and GitHub environment
`crates-io`. Protect that environment with the repository's release approval
policy. No persistent registry token is needed by subsequent workflow runs.
The official authentication action exchanges the job's OIDC identity for a
short-lived credential. [Official authentication action](https://github.com/rust-lang/crates-io-auth-action)

## Subsequent releases

The manual workflow defaults to `publish=false`, which runs checks only. An
authorized publication uses `publish=true` on the exact
annotated `behave-fire-vX.Y.Z` tag matching the core manifest. The workflow requires that
tag's commit to be in released `master` history, then waits for the protected
`crates-io` environment. It reuses package validation before authentication and
upload. Creating the tag, configuring the environment, and publishing remain
separate owner actions.

After an authorized upload, verify registry installation in a new consumer
outside the checkout. Use a version-only dependency such as
`behave-fire = "=0.1.0"`, with no path dependency or `[patch]` override. Run
the same downstream smoke program and inspect `cargo tree` to confirm crates.io
resolution. Record the actual published version and results before describing
crates.io installation as verified.
