# Binding field tests

This unpublished Rust-only test crate compiles the production primitive
argument, static schema, typed converter, direct column writer, and model
observer modules from `behave-py` directly. The PyO3 extension module cannot run
`cargo test` itself, so these tests check its field roles, input lengths, empty
inputs, output schemas, Crown scalar records, unit conversion, and staged and
configured model outputs without a Python linker. A raster parity test compares
float64 model raster columns with independently computed scalar results.

Run from the repository root:

```text
cargo test --manifest-path tools/binding_field_tests/Cargo.toml --locked -- --test-threads=1
```

The lockfile is generated with Rust 1.78. The crate is not part of normal wheel
builds.
