# Rust backend contract

All 89 native operation entries and scalar state tools use float64 physical
arithmetic. Real float16, float32, integer, and boolean physical inputs widen
once. Masked physical values become NaN; physical results are plain float64
arrays. Codes and flags retain separate integer/boolean storage. Python keeps
its original dtype, mask, and arithmetic behavior. Automatic selection can
therefore return different types on native and Python-only installations.
See [backend differences](BACKEND_DIFFERENCES.md) for the complete contract.

The [final Windows acceptance](performance/2026-11-09-reviewed-float64-acceptance.md)
passes every required speed, confidence, scalar, memory, and allocation gate.
Warmed 100k/1M speedups are 3.18-5.52x with one worker and 6.06-13.44x with four,
on the declared Windows host using CPython 3.12.8 and NumPy 2.1.3. Other hosts
and workloads require their own speed measurements.

## Backend selection

```python
from behave_fire import BehaveRun, FuelModels, available_backends, get_backend

model = BehaveRun(FuelModels(), backend="auto", threads=4)
python = get_backend("python")
rust = get_backend("rust")
```

`auto` chooses usable Rust when available. Otherwise it selects Python and
emits one `BehaveBackendWarning` per process at the first actual fallback.
Discovery and imports stay silent and do not consume that warning. Explicit
Python neither probes nor warns; explicit Rust raises an actionable ImportError
when its extension is absent or incompatible.

A separate once-per-process warning accompanies a verified complex-input route
to Python. Selection happens before Rust coercion or computation. The complete
Python operation runs once, preserving its complex values and masks; its errors
propagate. Unsupported/lossy complex operations, invalid complex controls, and
general object/custom arithmetic raise instead of silently discarding data.
Real masked inputs stay on Rust. Native calculation errors do not replay Python.

`requested_backend` retains the caller policy; `preferred_backend` records the
initial available choice; `backend_in_use` reports the last completed call.
Automatic namespaces use proxies for this reporting. Verified complex FireSize
operations move all physical state to Python before mutation and pin that
instance there. Other mutable helpers migrate only where support is proven.
New Rust-only grid sessions reject complex/object data and have no fallback.

Warn-as-error filters must be installed before the first warning. A warning
promoted to an exception does not consume the warning or mutate a migrating
state object. Changing filters later cannot retroactively make an emitted
once-per-process warning strict.

```python
import warnings
from behave_fire import BehaveBackendWarning

warnings.simplefilter("error", BehaveBackendWarning)
```

The root model uses composition; its class identity and repr can differ from
`behave_fire.behave.BehaveRun`. Use `behave_fire.python.BehaveRun` for historical
subclassing/type checks. Historical class-level static calls retain their Python
route. Built-in copies and pickles preserve values, selected backend, cached fuel
snapshot, and thread settings. Native-selected serialized objects require their
compatible extension; loading does not silently change their backend. Grid
sessions reject pickling explicitly.

## Numeric and ownership rules

Float64 normalization intentionally changes float32 rounding, integer overflow,
weak-scalar promotion, MaskedArray identity/fill values, and boxed scalar behavior.
Physical integers above 2^53 can round. IDs, selectors, iteration limits, codes,
and flags remain typed, with entry-point-specific validation before workers.
Missing classifications/statuses use int32 -1; missing classification flags are
false. Missing physical denominators do not become valid zero. Output availability
follows actual dependencies and selected branches rather than masking whole cells.
Structural errors raise before computation; physical numeric domains use IEEE
NaN/infinity. Finite scientific equations and existing unit constants remain.

Inputs, registry values, and overrides are snapshotted before detached native
work. Scalars stay compact; real broadcast columns materialize once. Logical
array order is preserved for strided, readonly, endian-swapped, and empty inputs.
Results and getters are independently owned; modifying them cannot alter input
buffers or a session. Buffers are initialized before work and split safely across
workers. Ordinary scientific/encoder loops request zero heap allocations.
Full models use the existing core batch owners; output chunks are not concatenated.

`threads=1` is serial, `threads=0` uses available parallelism, and positive counts
are capped by work. Native routes reject negative/nonintegral counts. Small jobs
can remain serial. The automatic Python route ignores `threads`; the explicit
Python constructor keeps its original signature without that argument.

## Raster stages and injection

`behave_fire.rust.SurfaceGrid` and `CrownGrid` own input columns, options, fuels or
supplied fuelbeds, persistent overrides, and intermediate state. Constructors
accept a common logical shape and base-unit scalar/per-cell inputs. Call
`run_stage`, edit a field with `set`, then call `run_remaining`. Requested/kept
fields control retention; freed dependencies can be recomputed. Explicit producer
runs replace their transient injections. `run_fused` restarts from edited inputs
and persistent overrides, discarding transient injections.

Reaction intensity uses BTU/ft^2/min; lengths use feet and spread uses ft/min.
Dead/live fuel moistures are fractions; foliar moisture is percent. Python getters
copy arrays. See `src/examples/staged_raster.py` and the Rust `staged_raster`,
`configured_tiles`, and `raster_tweaks` examples. Raster file I/O belongs to callers.

## Standalone core and bindings

The root `rust/` workspace has two crates:

- `rust/crates/behave-fire` is the independently versioned publishable core.
  It uses only the Rust standard library for normal/build dependencies, forbids
  unsafe code, and needs no Python, NumPy, C++, network, external tables, or
  sibling repository at build/runtime. Rust-only development dependencies test it.
- `rust/crates/behave-py` is unpublished and builds `behave_fire._rust` through
  PyO3/numpy 0.27. It owns Python conversion and transport around the same science.

Binding API 4 and the complete operation manifest identify compatible extensions.
Rebuild the extension when switching source revisions. No float32 math calibration,
SVML dispatch, masked payload arithmetic, or object-number emulation remains.
Historical third-party notices are retained as provenance. The initial Rust
package version 0.1.0 is independent of the Python distribution version.

## Artifacts and release routing

Native ABI3 wheels cover glibc Linux x86-64/aarch64, musl x86-64, macOS
x86-64/arm64, and Windows x86-64/ARM64. The matching universal Python wheel
contains no extension; the matching source archive contains both crates. Tests
cover CPython 3.11-3.14, applicable NumPy floors/current versions, actual-binary
Alpine/Windows ARM64 full suites, and strict artifact inventory/version/tag checks.
A pure project wheel requires no compiler. A completely compiler-free install
also depends on binary NumPy availability for that platform/interpreter.

See [release verification](RELEASING.md), [standalone crate delivery](CRATES_IO.md),
and [performance evidence](performance/2026-11-09-reviewed-float64-acceptance.md). Builds
and validation do not authorize PyPI/crates.io publishing, tags, or merges.
Dependent PRs ultimately target `pre_release`; `master` receives released code
through the documented promotion route.
