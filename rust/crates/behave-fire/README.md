# behave-fire

Standalone Rust core for the behave_fire fire-behavior port. The library provides
checked grid inputs, owned fuel/species data, unit conversions, and complete
scalar surface and crown pipelines with their component stages. It also includes
mortality, spotting, ignition, safety zones, vapor pressure deficit, fine dead
fuel moisture and stateful slope/ellipse tools.
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

Project-owned code is MIT licensed. The float64 engine contains no unsafe code,
float32 exponential profiles, or Intel SVML assembly. Historical third-party
notices remain packaged; the crate retains `MIT AND BSD-3-Clause` while that
provenance is retained. Preserve `LICENSE`, `NOTICE`,
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
use behave_fire::{
    engine::{PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks},
    FuelModels, units::SpeedUnit,
};

let fuels = PreparedFuels::from_registry(&FuelModels::default());
let plan = SurfacePlan::new(SurfaceOptions::default());
let mut input = SurfaceCell::new(124);
input.wind_speed = SpeedUnit::MilesPerHour.to_base(5.0);
let result = plan.run_cell(&input, fuels.for_grid(124), SurfaceTweaks::default());
assert!(result.spread_rate.is_finite());
# Ok::<(), behave_fire::BehaveError>(())
```

All physical arithmetic uses float64. Codes and flags retain integral and
boolean types. Missing physical values use NaN with output-specific dependency
rules. The separate Python binding widens real inputs and converts masked
physical values to NaN; no dtype, mask-payload, or object arithmetic emulation
runs in the core. Rust-only callers need no Python runtime.

## Raster calculations and stage injection

`raster::SurfaceRaster` and `raster::CrownRaster` run a plan over a whole grid.
Inputs are borrowed structure-of-arrays columns (`SurfaceColumns`,
`CrownColumns`): each field is a uniform `PerCell::Scalar` or a slice with exactly
one value per cell in row-major order. Lengths are validated once, before any
calculation or worker starts, and results are owned typed columns
(`SurfaceRasterOutput`, `CrownRasterOutput`). The library builds only the current
cell on the stack and never a full-grid vector of cell records.

Overrides are per-cell columns (`SurfaceTweakColumns`, `CrownTweakColumns`)
applied at the stage that would produce the value, so downstream outputs use
them; `run_from_fuelbeds` resumes from edited `FuelbedColumns`. An optional native
`validity` column skips cells that are false and returns NaN physical values and
classification -1 for them.
That low-level validity policy is separate from the Python-facing NaN policy.
`run_tile` processes a caller-chosen range and returns a
one-dimensional tile in original logical order. Raster file input and output
belong to the caller.

`cargo run --example raster_tweaks` runs a small raster with per-cell reaction
intensity overrides, a validity mask, and a tile. The raster engine is a Rust
interface alongside the configured model batches and owned staged sessions used
by the Python extension. Final end-to-end performance certification is pending.

## Effects and utility calculations

`BehaveRun` provides all eleven calculation routes of the Python facade. Rust
scalar methods use documented base units, with conversion functions in `units`
for callers using other units. Stateful helpers store float64 physical values
and use IEEE NaN and infinity for numeric domain results. Structural shape,
control, and allocation errors remain checked before scientific work.

```rust
use behave_fire::{BehaveRun, FuelModels};

let model = BehaveRun::new(FuelModels::default());
let scorch_ft = BehaveRun::calculate_scorch_height(100.0, 5.0, 77.0);
let mortality = model.calculate_crown_scorch_mortality(
    scorch_ft, 80.0, 0.5, 12.0, 1,
);
assert!((0.0..=1.0).contains(&mortality.probability_mortality));
```

Lightning ignition remains available as a deprecated compatibility calculation.

Run `cargo run --example standalone` from this crate directory for a complete
default/custom-fuel surface and mortality example with registry snapshots.

## Prepared physics and Crown stages

`PreparedFuels` snapshots the registry into immutable heap-backed numerical rows.
SAVR terms are prepared for every row; static fuels also retain fuelbed and
reaction coefficients. Moisture-dependent quantities are still calculated per
cell. Changes to a standalone `PreparedFuel` invalidate cached terms through
bitwise physical-field checks, including signed zero. Dynamic fuels and edited
fuelbeds retain the appropriate uncached calculations.

`CrownPlan::prepare_surface` returns the complete public Surface intermediates.
`CrownPlan::prepare_spread` returns `CrownSpreadResult`, the particle, fuelbed,
reaction, spread, and length-to-width stages consumed by the complete Crown
pipeline. Both retain supplied terrain and overrides; the crowning backsolve
uses its independent flat/default controls.
