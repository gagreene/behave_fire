# behave-fire

Standalone Rust core for the behave_fire fire-behavior port. The library provides
checked grid inputs, owned fuel/species data, unit conversions, and complete
scalar surface and crown pipelines with their component stages. It also includes
mortality, spotting, ignition, safety zones, vapor pressure deficit, fine dead
fuel moisture, stateful slope/ellipse tools, and the reference containment model.
The optional Python extension uses this same core through a separate, unpublished
binding crate. Rust applications do not need that extension.

The core has no runtime dependencies and does not require Python, NumPy, C++,
external data, or a sibling repository. Rust 1.78 and edition 2021 are the tested
minimum toolchain and edition.

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
is intended for small generic results; `SurfaceBatch` and `CrownBatch` use named
per-field vectors for large surface and crown batches.

`effective_worker_count` caps positive requests by the logical work count. Zero
requests use available parallelism, and empty work needs no workers. The core
accepts unsigned counts; bindings validate signed and dynamically typed inputs.

The Rust core version is independent of the Python distribution version. The
initial `0.1.0` is a proposed package version, not publication authorization.

## License and provenance

Project-owned code is MIT licensed. The NumPy-derived float32 exponential
kernel and bundled Intel SVML atan2 assembly are BSD-3-Clause licensed; the
combined crate declares `MIT AND BSD-3-Clause`. Preserve `LICENSE`, `NOTICE`,
`LICENSES/NumPy-BSD-3-Clause.txt`, and `LICENSES/Intel-SVML-BSD-3-Clause.txt`
when redistributing the corresponding source or compiled code.

The crate contains newly authored shape/error code and embedded fuel, species,
and fine dead fuel moisture tables generated from the explicit Python reference
at commit `7933789f00470c6bbfbe942dc22bdd7f74e192f9`. The tables contain 96 fuel
records, 24 species records, and 774 moisture entries. Their provenance is
recorded in the packaged fixtures; `tools/check_rust_tables.py --check` checks
drift during repository development and CI. Building or using the crate never
requires that Python tool. Scientific modules retain applicable source and
data notices and record their provenance.

## Surface and crown calculations

The scalar surface and crown pipelines and their public stages are available
without Python. Inputs and outputs use the documented reference base units;
typed unit selectors are available where the reference accepts units.

```rust
use behave_fire::{BehaveRun, FuelModels, surface::SurfaceInput, units::SpeedUnit};

let model = BehaveRun::new(FuelModels::default());
let mut input = SurfaceInput::new(124);
input.wind_speed = 5.0;
input.wind_speed_units = SpeedUnit::MilesPerHour;
let result = model.do_surface_run(&input)?;
assert!(result.spread_rate.is_finite());
# Ok::<(), behave_fire::BehaveError>(())
```

The `surface::single` and `crown::single` modules preserve the reference's
float32 intermediate arithmetic. Mask-aware companion APIs reproduce the
documented NumPy masked-array stage behavior. Float32 NumPy exponential dispatch
can differ by a few ULPs from scalar libc; select the matching `MathProfile` when
matching a particular NumPy build. No Python runtime is used by either profile.
The Python bindings remain incomplete at this layer.

## Effects and utility calculations

`BehaveRun` provides all eleven calculation routes of the Python facade. Rust
scalar methods use documented base units, with conversion functions in `units`
for callers using other units. Stateful helpers preserve their checked geometry
errors and numeric-kind metadata; `GeometryError` distinguishes domain,
overflow, and division-by-zero errors.

```rust
use behave_fire::{BehaveRun, FuelModels};

let model = BehaveRun::new(FuelModels::default());
let scorch_ft = BehaveRun::calculate_scorch_height(100.0, 5.0, 77.0);
let mortality = model.calculate_crown_scorch_mortality(
    scorch_ft, 80.0, 0.5, 12.0, 1,
);
assert!((0.0..=1.0).contains(&mortality.probability_mortality));
```

Containment intentionally reproduces the Python fixed-step approximation and
its status codes; it does not implement RK4 or the complete upstream ContainSim.
Lightning ignition remains available as a deprecated compatibility calculation.

Run `cargo run --example standalone` from this crate directory for a complete
default/custom-fuel surface and mortality example with registry snapshots.
