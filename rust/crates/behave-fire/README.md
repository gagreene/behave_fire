# behave-fire

Standalone Rust foundation for the behave_fire fire-behavior port. This initial
version provides checked grid inputs and owned typed outputs. Scientific models
are added in later implementation layers; this foundation does not perform fire
calculations yet.

The core has no runtime dependencies and does not require Python, NumPy, C++,
external data, or a sibling repository. Rust 1.78 and edition 2021 are the tested
minimum toolchain and edition for this foundation.

```rust
use behave_fire::{GridOutput, GridShape, PerCell};

let shape = GridShape::new(vec![2, 3])?;
let wind = PerCell::Scalar(4.0_f64);
wind.validate(&shape, "wind")?;
assert_eq!(wind.get(5), Some(&4.0));

let output = GridOutput::new(shape, vec![false; 6])?;
assert_eq!(output.shape().dimensions(), &[2, 3]);
assert_eq!(output.cells(), &[false; 6]);
# Ok::<(), behave_fire::BehaveError>(())
```

Empty dimensions represent one scalar cell. A zero dimension represents an empty
grid. Nonzero dimension products and per-field lengths are checked before work.
`PerCell` borrows row-major cells or broadcasts an owned scalar. `GridOutput<R>`
is intended for small generic results; future large surface and crown batches
use named per-field vectors.

`effective_worker_count` caps positive requests by the logical work count. Zero
requests use available parallelism, and empty work needs no workers. The core
accepts unsigned counts; bindings validate signed and dynamically typed inputs.

The Rust core version is independent of the Python distribution version. The
initial `0.1.0` is a proposed package version, not publication authorization.

## License and provenance

MIT licensed. The crate includes the repository's existing MIT license and
copyright notice in `LICENSE`.

This layer contains newly authored shape/error code and embedded fuel, species,
and fine dead fuel moisture tables generated from the explicit Python reference
at commit `7933789f00470c6bbfbe942dc22bdd7f74e192f9`. The tables contain 96 fuel
records, 24 species records, and 774 moisture entries. Their provenance is
recorded in the packaged fixtures; `tools/check_rust_tables.py --check` checks
drift during repository development and CI. Building or using the crate never
requires that Python tool. Later scientific ports retain applicable source and
data notices and record their provenance.
