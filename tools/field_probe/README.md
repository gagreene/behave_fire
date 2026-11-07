# Prepared binding field probe

This unpublished Rust-only test crate compiles the production argument,
selector, converter, and allocation-counter modules from `behave-py` directly.
It tests their behavior without requiring a Python linker or reimplementing the
production accessors.

Run from the repository root:

```text
cargo test --manifest-path tools/field_probe/Cargo.toml --locked -- --test-threads=1
```

The single-test-thread setting isolates the process-wide diagnostic allocator.
The lockfile is generated with Rust 1.78. The probe is not part of normal wheel
builds, and the allocation feature remains disabled in distributed extensions.