# Native Rust raster engine

All 88 native operation entries now use float64 physical arithmetic with separate
integer controls and boolean flags. Masked physical inputs become NaN; physical
outputs are ordinary float64 arrays. Python remains the unchanged reference and
handles only explicitly selected or verified automatic complex fallback routes.
See [backend differences](BACKEND_DIFFERENCES.md) for the numeric and NoData rules.

Binding API 4 owns primitive input snapshots before detached computation. Full
Surface/Crown models use their existing core batches; component helpers write
initialized final columns directly. Unit selectors and immutable coefficients
are decoded before cell loops. No per-cell numeric tags, object arithmetic,
float32 profiles, mask-payload emulation, or SVML assembly remain. The standalone
core forbids unsafe code and has no normal or build dependencies.

`behave_fire.rust.SurfaceGrid` and `CrownGrid` expose owned staged sessions,
per-field injections, explicit output retention, and recomputation. Rust callers
can use `engine::GridSession` without Python. Model batches and sessions share
the scientific equations; component fuelbed calculations also use the session's
field evaluator. Classification codes use int32, with -1 when unavailable;
missing classifications clear their boolean flags.

Fuel registries have 257 lookup rows, indexed 0 through 256. Supported fuel IDs
are 1 through 256; other IDs select undefined row zero and retain the existing
zero-spread behavior. Custom model 256 is supported. These boundary behaviors
are project decisions where the pinned C++ has unsafe lookups; they are not
claims of computed C++ parity. See [the fuel-ID contract](BACKEND_DIFFERENCES.md)
for the distinction between the C++ definition bounds and its storage bounds.

## Typed scalar stages

`engine::SurfaceCell` and `engine::CrownCell` contain numerical values in
fractions, feet, ft/min, and degrees (foliar moisture remains percent).
`SurfaceOptions` uses `WindOrientation`, `WindHeight`, and `WafMethod` enums.
The native core no longer interprets Python mode strings or constructs string
controls, synthetic full Surface inputs for Crown, or per-cell fuel registries.
Python's permissive mode matching is confined to the binding adapter.

`PreparedFuels::from_registry` takes an immutable numerical snapshot with dense
lookup rows 0 through 256. Later edits to the original registry do not change that snapshot.
`PreparedFuel` can also be copied and edited explicitly for an experimental
scalar call. Names and other registry metadata are not copied into cell loops.
Dynamic herbaceous transfer remains moisture-dependent; preparation does not
cache it incorrectly by fuel ID.

`SurfacePlan::run_cell` evaluates a complete cell. `prepare_fuelbed` returns
publicly editable `FuelbedIntermediates`; `run_from_fuelbed` resumes from that
value without rebuilding it. `SurfaceTweaks` can replace reaction intensity,
no-wind/no-slope rate, midflame wind, wind/slope amplification, or forward rate.
Overrides apply before dependent outputs. Reaction intensity is BTU/ft²/min,
including the named output. A forward-rate override recomputes shape, intensity,
and flame length. Undefined-fuel output masking remains the final step.

`CrownPlan::run_cell` accepts three independent `SurfaceFireInputs<f64>` values
and a borrowed prepared model-10 fuel. `CrownTweaks` provides internal Surface
overrides, critical intensity/rate overrides, and a fraction-burned override.
The model-10 particles, fuelbed, and reaction are calculated once and reused.
The supplied slope is retained, matching Python's frozen behavior even though
the inspected C++ Rothermel path resets it.

The former unreleased `SurfaceInput`/`CrownInput` interfaces and their adapters
are replaced. In-repository consumers, tests, and examples migrate together.
Python entry points and output dictionaries retain their existing signatures
and scientific contract. Native raster columns are implemented in the core
(see "Borrowed raster API" below).

## Borrowed raster API

`raster::{SurfaceRaster, CrownRaster}` borrows structure-of-arrays inputs through
`SurfaceColumns` and `CrownColumns`. Each `PerCell` field is either a uniform
scalar or a slice with exactly `GridShape::len()` elements. Constructing a raster
validates every supplied input length before calculations or worker launch;
calling a pipeline validates override and stage columns before allocating its
outputs. Shapes are logical row-major dimensions, including scalar `[]` and
empty shapes such as `[0, 3]`. Classification is an owned `Vec<i32>`.

`SurfaceTweakColumns` and `CrownTweakColumns` supply uniform or per-cell overrides
at the same stages as their scalar counterparts. `FuelbedColumns` lets callers
edit and resume fuelbed intermediates without rebuilding particles. These are
native Rust interfaces; existing Python stage helpers remain the Python
injection interface. Reusing a plan with changed overrides keeps no old results.

Native `validity: Some(PerCell::Cells(...))` explicitly skips false cells, keeps
their positions, returns zero numerical payloads and a false output validity.
Undefined-fuel scientific masking remains separate, including its length/width
ratio of one. Python masked arrays never imply this native skip policy: their
underlying data can participate in later calculations.

`run_tile(range, ...)` returns a one-dimensional tile in original logical order.
Input and override slices still describe the full raster; callers retain its
original dimensions and the range offset. A caller reading external windows
can instead construct one raster per window, using that window's own shape and
slices. The library performs no file I/O. See
`rust/crates/behave-fire/examples/raster_tweaks.rs` for a complete example.

The implementation uses the bounded worker executor with disjoint output
slices. It builds only the current scalar cell on the stack, never a full-grid
vector of cell records. Output allocations depend on fields and workers. This
layer makes no out-of-core scheduler claim.

## Timing

`python tools/time_standard_runs.py` times a standard Surface and Crown run on
each available backend and prints a report for that machine. The report has no
pass or fail threshold and is not regression evidence. The development-era
benchmark campaigns, allocation probes, and their evidence were removed from the
repository; Git history at commit `647b6f6` retains them.
