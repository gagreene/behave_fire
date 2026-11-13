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

The [recorded Windows acceptance](performance/2026-11-09-reviewed-float64-acceptance.md)
passes all 40 cases and 80 first/warmed confidence bounds, scalar regression
budgets, peak-memory comparisons, full-output validation, and zero-allocation
loop gates. Warmed 100k/1M ratios are 3.18-5.52x serial and 6.06-13.44x with four
workers on the declared Windows/NumPy2.1.3 host. These are Python/Rust ratios.
Other-host speed claims require their own timing. Earlier records below remain
historical evidence. Separate helper/staged diagnostics do not replace final
model acceptance.
The recorded scientific source is `9be8a50`. The
[later review follow-up](performance/2026-11-09-claude-pr-review-followup.md)
documents subsequent boundary corrections separately from these measurements.

## Historical engine and performance records

The following design notes and dated campaigns describe earlier revisions.
References to API 3, compact or masked arithmetic, float32 profiles, old Rust
entry points, the former 0–260 fuel lookup, and prior feasibility failures are
historical. The contract above and `BACKEND_DIFFERENCES.md` describe the current
implementation.

## Typed scalar stages

`engine::SurfaceCell` and `engine::CrownCell` contain numerical values in
fractions, feet, ft/min, and degrees (foliar moisture remains percent).
`SurfaceOptions` uses `WindOrientation`, `WindHeight`, and `WafMethod` enums.
The native core no longer interprets Python mode strings or constructs string
controls, synthetic full Surface inputs for Crown, or per-cell fuel registries.
Python's permissive mode matching is confined to the binding adapter.

`PreparedFuels::from_registry` takes an immutable numerical snapshot with dense
0–260 lookup. Later edits to the original registry do not change that snapshot.
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
(see "Borrowed raster API" below). Direct typed output in the Python binding is
not: removing scalar core allocations alone does not remove the large
typed-result encoder cost.

The scalar core allocation probe now measures zero allocation/reallocation
requests per Surface/Crown cell after preparation, including rate overrides
and the checked heterogeneous typed-Crown sample. Encoder allocation remains
explicitly unmeasured in that core-only probe. The scalar core and its three
doctests pass 561 tests on stable Rust and Rust 1.78; the new frozen-engine and
override regressions cover custom/undefined FM10, slope passthrough, dynamic
curing, snapshot isolation, non-finite values, and zero signs.

## Accepted targets

Compare each dataset and route independently on a declared host with at least
four available physical cores. A passing mean across routes does not qualify.

| Cells | Route | One worker | Four workers |
| ---: | --- | ---: | ---: |
| 10,000 | Surface and Crown f64 | 1.25× Python | Diagnostic |
| 10,000 | Typed Crown A and B | At least Python parity (1.0×) | Diagnostic |
| 100,000 and 1,000,000 | All four routes | 2.0× Python | 4.0× Python |

Ratios divide Python median public-call time by Rust median public-call time.
Both first-call and warmed results must pass, including the lower endpoint of a
paired-bootstrap 95% interval. Use seven fresh-process blocks, with five warmed
calls after the initial call. A Python observation may pair with multiple worker
settings inside its block; that does not create extra independent samples.

The one-cell native regression budget is the greater of 10% or 0.25 ms against
the pre-optimization native baseline. Report scalar and small-array latency
separately from raster claims. Native median peak process memory must not exceed
Python's on each large case. Required native kernels and output writers must
perform no per-cell allocations after bounded batch/tile setup.

The historical investigation and its limitations remain in
[RUST_BACKEND_BENCHMARKS.md](RUST_BACKEND_BENCHMARKS.md). These targets do not
assert that the current implementation already meets them.

## Reproducible evidence

`tools/benchmark_backends.py` retains the original route/worker selectors and
adds explicit datasets, repeated blocks, warmed calls, and separate validation
workers. The output is a schema-1 report with `records`, `validations`,
`allocations`, `baseline`, and `errors`. Standalone workers retain legacy
first-call fields such as `total_seconds`, `numpy`, and `output_bytes`.

Default runs use an unprofiled three-block development screen and one warmed
call. Add `--preset acceptance` for seven blocks and five warmed calls; select
`--sizes 1 10000 100000 1000000` and `--worker-threads 1 4` for full acceptance,
including fresh one-cell candidates for the scalar regression gate.
Automatic-worker diagnostics run separately. Full acceptance is a multi-hour
campaign; estimate its duration from a pilot and retain partial observations.

Supply `--baseline preoptimization-scalar.json` to load a separately retained
pre-optimization report containing Rust one-cell observations for every route
and dataset, using one worker and the same Python/NumPy build, platform and seed.
That baseline needs at least three fresh-process observations and five warmed
calls per worker. Its old timings do not replace fresh one-cell candidate
measurements in the acceptance campaign. Record its source and native-binary
SHA-256 hashes independently when that build is captured.

Uniform inputs reproduce the historical model-10 case. Heterogeneous inputs
cycle models 1, 4, 10, 101, 124, 145, 201, and 0, with deterministic weather and
moisture variation. Crown also varies canopy inputs. Typed A supplies f32 spread,
uint16 intensity, and masked f64 heat (every seventh position). Typed B supplies
masked f64 spread (every fifth position), int32 intensity, and f32 heat.

Every observation records Python, NumPy version/build, platform, source and
extension hashes, input seed, layout, and worker request. Changed NumPy builds
need new reference timings. Source/binary hashing runs after measurements.
Model construction, imports, input generation, and untimed validation are
excluded from call timing and reported separately where applicable.

Separate workers check every output against explicit Python, including field
inventory, dtype, shape, masks, masked raw data, fill values, non-finite values,
zero signs, and integer/status values. Oracle arrays are never retained in timed
workers, preventing false memory regressions caused by the validation itself.

Profiling is opt-in with `--profile`. Native copy, detached, concatenation, and
transfer attribution exists only in diagnostic runs. Unprofiled attribution is
reported as unmeasured, not incorrectly classified as Python execution time.
Profiled runs and development screens cannot certify the performance targets.

Run the checker with the independently recorded pre-optimization hashes
(replace the two uppercase placeholders with those recorded SHA-256 values):

```text
python tools/check_performance.py --results results.json --requirements tools/performance_requirements.json --baseline-source-hash PREOPT_SOURCE_SHA256 --baseline-extension-hash PREOPT_EXTENSION_SHA256
```

Programmatic callers can instead declare these expected hashes in
`requirements["baseline_identity"]`, with `source_hash` and `extension_hash` keys.
The expected baseline identity is separate from the candidate identity and must
come from the trusted build record, rather than being inferred from the report
being checked. Missing/empty identities and any baseline row with different
source/binary hashes fail the scalar gate, including mixtures of builds.

The checker fails on missing cases, failed validation,
inconsistent provenance/NumPy builds, inadequate samples, missed intervals,
allocation/memory failures, or scalar regression. The checked-in requirements
record the owner-approved targets. Changing a target needs an explicit decision.

## Native attribution probe

`tools/perf_probe/` is an unpublished standalone Rust consumer. Build/run its
release timing mode without `count-alloc`; use a separate build with that
feature for allocation mode. The tool-only allocator delegates to the system
allocator and counts requested allocations/reallocations separately from timing.
Production numerical code gains no instrumentation or unsafe allocator wrapper.

The initial probe measures prepared scalar core inputs over a bounded repeating
period, with complete results retained through `black_box`. Mask periods are
included in that period. Setup, Python transport, and output-column storage are
excluded. Such results are attribution, not end-to-end speed claims. The probe
reports `encoder_per_cell: null` until actual encoder instrumentation is added;
an excluded or unmeasured encoder is never reported as allocating zero times.
The probe also measures native raster output storage (`--scope raster`) and a
disposable compact typed experiment (`--scope compact`); see the feasibility
checkpoint below.

## Execution checkpoints

After the typed scalar and raster engine exists, check core headroom before
production binding work. The provisional core screen is 1.25 times the public
target (2.5× core speed for a 2× public-call target). Measure actual typed layouts;
do not substitute float64 results. At most one disposable prototype per failing
route is allowed before returning evidence for a go/no-go decision. Later kernel
tuning allows three isolated experiments per failing route before another
decision. A missed target is not an instruction to tune indefinitely.

No native optimization has been applied in the measurement/contracts layer.

## Initial development screen

The new harness captured 156 unprofiled timing observations and 32 separate
full-output validations with no validation errors. The host runs CPython 3.12.8,
NumPy 2.1.3, and Windows x86-64, with eight physical and sixteen logical cores.
Each raster case has three fresh-process blocks and one warmed call per worker;
the scalar baseline has five warmed calls per worker. These are development
screens, not seven-block performance certification.

First-call medians in seconds:

| Dataset | Route | Cells | Python | Rust, one worker | Rust, four workers |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | Surface | 10,000 | 0.038195 | 0.105696 | 0.047122 |
| Uniform | Crown | 10,000 | 0.051741 | 0.155424 | 0.069789 |
| Uniform | Surface | 100,000 | 0.397675 | 1.102967 | 0.396879 |
| Uniform | Crown | 100,000 | 0.530616 | 1.464129 | 0.537084 |
| Heterogeneous | Surface | 10,000 | 0.041041 | 0.108855 | 0.048226 |
| Heterogeneous | Crown | 10,000 | 0.052217 | 0.185304 | 0.067162 |
| Heterogeneous | Surface | 100,000 | 0.409366 | 1.012044 | 0.375249 |
| Heterogeneous | Crown | 100,000 | 0.573916 | 1.675330 | 0.621312 |
| Uniform | Typed Crown A | 10,000 | 0.052722 | 4.268690 | 1.410682 |
| Uniform | Typed Crown B | 10,000 | 0.055730 | 4.957631 | 1.521601 |
| Heterogeneous | Typed Crown A | 10,000 | 0.055655 | 4.482313 | 1.587227 |
| Heterogeneous | Typed Crown B | 10,000 | 0.055267 | 4.440838 | 1.438473 |

Both mixed layouts reproduce the slowdown. Four-worker common raster execution
at 100,000 cells remains around Python speed on this host, below the accepted
4× target. No old typed-million run was attempted. The allocation probe measures
nine core allocations and 115 requested allocation bytes per Crown cell in
common and typed routes, excluding the separately investigated output encoder.

The measured extension SHA-256 is
`18894ede5a7539df87c3e426612e15d97714d774a47d8e82e34146fa33aaea3a`.
The observations contain source-content hash
`e54891d4f99cc70e497f2c535686edf838d1601c940697928cd749baa84ac68e`.
That fingerprint covers scientific sources, manifests, input generation, and
the measurement driver; it is not a claim that an earlier binary was rebuilt
from a later commit. The binary was built in release mode from the original
stack's final rebased source, whose tree equals integration commit `500221b`.

Raw reports are retained locally under this plan's ignored execution workspace:
`s1-scalar-baseline.json`, `s1-common-screen.json`, and `s1-typed-screen.json`.
They retain warmed observations, NumPy build fingerprints, per-worker memory,
and the independent validation records. Initial core probe JSON and compiler
output are retained alongside them. These local artifacts are not packaged;
the tables and method above are the tracked reviewer record.

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

The initial implementation reuses the existing bounded worker executor and
owned output vectors. It builds only the current scalar cell on the stack,
never a full-grid vector of cell records. Output allocations depend on fields
and workers. Disjoint worker output slices and tile scheduling remain later
performance work; this layer makes no out-of-core scheduler claim.

## Recovery core confirmation (2026-11-06)

**Decision: GO for the planned binding work.** Seven fresh paired processes per case, each with five warmed calls, passed all 32 first/warm lower-95% thresholds. Exploratory samples were excluded. All stored numeric values, typed masks, classification values, and constant schemas matched independently built unchanged reference outputs outside timing.

The measurements use Windows x86-64, CPython 3.12.8, NumPy 2.1.3, Rust 1.99, ordinary release settings, and the same Xeon E5-2640 v3 host as the profile sweep. Ratios divide Python public-call time by native complete-column time. They exclude the Python/native boundary and do not certify end-to-end wrapper performance.

| Route at 100,000 cells | First-call median speedup range | Warm median speedup range | Lowest first/warm 95% bound |
| --- | ---: | ---: | ---: |
| surface | 3.70–3.82x | 3.52–4.06x | 3.43x |
| crown | 3.13–3.34x | 3.16–3.24x | 2.83x |
| typed-crown | 3.02–3.20x | 3.04–3.29x | 2.71x |
| typed-crown-b | 3.04–3.12x | 3.04–3.18x | 2.78x |

Both required datasets passed at 10,000 and 100,000 cells. The core thresholds remain 1.5625x for small Surface/Crown, 1.25x for small typed Crown, and 2.5x for every large route. Public-call targets remain unchanged; million-cell and four-worker acceptance still require the later binding and scheduling work.

Retained changes prepare SAVR exponentials and size bins, separate minimal Crown spread stages from complete Surface output, and cache static fuelbed/reaction coefficients. Bitwise raw-field keys invalidate caches after edits, dynamic fuels retain moisture-dependent preparation, and explicitly supplied fuelbeds bypass registry caches. Full public Crown intermediates remain complete.

The [confirmation record](performance/2026-11-06-core-confirmation.json) contains all 224 fresh-worker observations, confidence results, source hashes, compiler identity, and the frozen probe hash. Full independent output snapshots remain in the local ignored execution workspace. `tools/perf_probe/` exports complete columns outside timing when `BEHAVE_PROBE_OUTPUT` names a destination directory.

Allocation evidence now uses `tools/measure_native_allocations.py` and `tools/assemble_allocation_evidence.py`. The counting feature is absent from normal builds and wheels. Raw kernel counts were zero in 48 verified samples across all four routes; the encoder is explicitly unmeasured until the direct writer is implemented. The acceptance checker rejects missing scopes, nonzero counts, incomplete worker samples, and mismatched build provenance.

## Historical Task 3 feasibility checkpoint

The following earlier decision is superseded by the recovery GO confirmation above.

**Decision: pause before production binding and specialization work.** The
large-raster core headroom screen is unmet after the one permitted disposable
prototype for each typed route. Owner-approved public-call targets remain
unchanged. This is a feasibility result, not final performance certification
or a speed claim for the current Python Rust wrapper.

The release probe's `--scope raster` includes complete output-column storage,
using the new native raster API for f64 and the allocation-free checked typed
core for mixed layouts. `--scope compact` runs a disposable typed experiment
under `tools/perf_probe/src/compact.rs`; it is not a supported core API, a
production specialization, or a fallback for other dtype combinations. It
reuses the FM10 science stages and implements compact postprocessing only for
the two defined layouts, preserving f32 input quantization, the layout-B f32
heat predicate, integer intensity inputs, masked left-operand restoration,
mask-dropping selection and all 21 outputs. Every stored result and integer
classification is compared outside timing with the checked scalar reference.
Metadata, masks and raw masked values are also checked over the complete
weather/fuel/mask period. The probe rejects missing or duplicate output fields.

Each route/dataset/size uses three paired fresh-process blocks in seeded shuffled
order, with one first and one warmed call per worker. Python is the public
explicit backend on CPython 3.12.8, NumPy 2.1.3, Windows x86-64. Native timing
includes complete numerical output storage but excludes Python transport;
NumPy public-call timing includes that transport. Setup, validation and allocation
instrumentation are outside timed regions. Existing Python-oracle differential
suites remain separate from the probe's scalar replay checks. The measurements
cover 10,000 and 100,000 cells as a reduced development screen; they do not
certify million-cell, four-worker or cross-platform targets.
Seconds below are medians across three fresh processes. Native typed rows use
the compact experiment; the checked typed path remains slower. Ratios compare
the public Python call with native core plus complete output storage only.

| Cells | Dataset | Route | Python first / warm | Native first / warm | Ratio first / warm |
| ---: | --- | --- | ---: | ---: | ---: |
| 10,000 | uniform | Surface | 0.034897 / 0.035757 | 0.014171 / 0.013960 | 2.46x / 2.56x |
| 10,000 | uniform | Crown | 0.046033 / 0.046140 | 0.024075 / 0.024437 | 1.91x / 1.89x |
| 10,000 | uniform | Typed A | 0.049553 / 0.048343 | 0.025890 / 0.026176 | 1.91x / 1.85x |
| 10,000 | uniform | Typed B | 0.050608 / 0.050803 | 0.027652 / 0.026299 | 1.83x / 1.93x |
| 10,000 | heterogeneous | Surface | 0.033892 / 0.033579 | 0.015647 / 0.014862 | 2.17x / 2.26x |
| 10,000 | heterogeneous | Crown | 0.048483 / 0.048894 | 0.023749 / 0.025003 | 2.04x / 1.96x |
| 10,000 | heterogeneous | Typed A | 0.048877 / 0.048334 | 0.025614 / 0.026064 | 1.91x / 1.85x |
| 10,000 | heterogeneous | Typed B | 0.050168 / 0.048107 | 0.025227 / 0.025419 | 1.99x / 1.89x |
| 100,000 | uniform | Surface | 0.359708 / 0.377137 | 0.159002 / 0.154541 | 2.26x / 2.44x |
| 100,000 | uniform | Crown | 0.481924 / 0.474781 | 0.248961 / 0.252465 | 1.94x / 1.88x |
| 100,000 | uniform | Typed A | 0.485923 / 0.493117 | 0.262803 / 0.262991 | 1.85x / 1.88x |
| 100,000 | uniform | Typed B | 0.518501 / 0.507598 | 0.277326 / 0.291026 | 1.87x / 1.74x |
| 100,000 | heterogeneous | Surface | 0.358690 / 0.359472 | 0.146491 / 0.145789 | 2.45x / 2.47x |
| 100,000 | heterogeneous | Crown | 0.502872 / 0.502467 | 0.293021 / 0.253576 | 1.72x / 1.98x |
| 100,000 | heterogeneous | Typed A | 0.485383 / 0.499888 | 0.258745 / 0.263036 | 1.88x / 1.90x |
| 100,000 | heterogeneous | Typed B | 0.483849 / 0.503197 | 0.264876 / 0.259771 | 1.83x / 1.94x |

At 10,000 cells, the native stored-output ratios clear the provisional screens
of 1.5625x for Surface/Crown and 1.25x for typed Crown. At 100,000 cells none
consistently clears 2.5x on both datasets and call categories. Crown and the
compact typed prototype are still below the 2x public target in several cases,
even before Python-boundary costs. This is sufficient for a no-go at the reduced
checkpoint. Million-cell and four-worker acceptance are unverified.

The ordinary checked typed path, with full-column storage but no compact
arithmetic, takes about 0.06–0.08 seconds at 10,000 cells and 0.6–0.8 seconds at
100,000 cells. The single compact candidate substantially improves that core
cost while preserving the checked outputs, but does not establish the required
large-raster headroom. No additional candidate was tried for either typed
layout. Further f64 prototype work was not spent after the typed checkpoint
already required an owner decision.

Equations-only attribution below excludes full output storage. These are three
iterations in one native process per case, separate from the paired public-call
screen, so differences between these and the stored-output table must not be
interpreted as additive, precisely isolated costs.

| Route, 100,000 cells | Uniform equations only | Heterogeneous equations only |
| --- | ---: | ---: |
| Surface | 0.150952 s | 0.123615 s |
| Crown | 0.240064 s | 0.224170 s |
| Checked typed A | 0.586304 s | 0.592001 s |
| Checked typed B | 0.576295 s | 0.596573 s |

A separate instrumented build reports zero allocation requests/bytes per scalar
core cell on all four routes. Whole stored-output calls make 17 requests for
Surface, 25 for Crown, 45 for checked typed layouts and 43 for the compact
experiment, unchanged between 1,000 and 10,000 cells. Requested output bytes
still scale with cell count: flat request counts do not mean constant memory.
These figures exclude input preparation and Python output encoding.

The final reduced screen contains 120 observations and zero worker/validation
errors. Its probe SHA-256 is
`9c391dbe25d5ea242e91941a2d1186d99f7cca8b5576de7d2f2f8dd64ba44531`;
the core/probe source fingerprint is
`831abbcb84983bfdbc52a8cf6583c123ff55052fcfde939f5b92d4be046594a4`.
Each Python observation retains the exact NumPy version/build and source/binary
provenance. Raw `s3-feasibility-screen.json`, `s3-equations-only.jsonl`, and
`s3-allocations.jsonl` remain in the ignored execution workspace. The source and
probe fingerprints are recorded independently; they do not imply the old
Python extension already uses the new raster/prototype transport.

The historical next decision was whether to authorize a bounded block-kernel feasibility
investigation before binding work, continue the planned binding work while
explicitly deferring this unmet screen, or revise the targets. The recommended
option is the bounded kernel investigation with the targets unchanged. No
production binding/API-3, generalized typed writer, parallel-tile optimization,
wheel publication or performance-stack merge is included in this checkpoint.
## Prepared input boundary (recovery layer 2)

Native evaluators now use compile-time field slots and borrowed views over owned
Rust snapshots. Metadata prefixes, integer kinds, constant math profiles, string
controls, and uniform numeric unit selectors are decoded before cell loops.
Varying selectors and metadata remain indexed buffers. Missing optional values
no longer construct discarded error strings. Unit converters fill numerical
buffers directly and retain the existing plain-f64 normalization and public
selector conventions, including negative table indices.

A separate allocation build observed these one-worker request slopes between
1,000 and 10,000 cells. They include all native allocations outside the measured
science kernel, so they are not encoder-only certification:

| Route | Before preparation | After preparation |
| --- | ---: | ---: |
| Surface | 8/cell | 0/cell |
| Crown | 7/cell | 0/cell |
| Typed Crown A/B | 842/cell | 679/cell |

Forty-eight real observations passed complete Python output comparisons, including
four-worker checks. The encoder scope remains unmeasured until the general writer
is installed. The native-only `tools/field_probe` tests compile the actual binding
preparation modules without linking Python and check metadata, error order, unit
selectors, and allocation counts.

Reduced development screens used three fresh pairs and one warmed call, not the
seven-pair acceptance procedure. At 10,000 cells, common Surface calls took about
27–29 ms and Crown 34–35 ms, versus Python's 35–36 ms and 50–55 ms respectively.
Typed Crown still took 4.4–5.4 seconds; these measurements do not establish the
final public-call targets. The remaining output transport and pipeline fusion
work are required.
## General output writer (recovery layer 3)

Binding API 3 replaces the per-cell `Cell` maps and expanded f64 metadata columns
with typed output buffers, constant field schemas, and preallocated variable
sidecars. All supported layouts use the same writer. Float32 results and integer
classifications retain their native storage types; integer payloads do not round
through f64. Object payload tags, raw masks, fills, complex components, and
independent output shapes remain covered by the reference suite.

A first-row capture establishes schema and retains that computed row for replay;
the science is evaluated exactly once per cell. Operation identities and output
field order are fixed before the remaining loop. Encoder instrumentation starts
at the first actual output write, after input preparation and kernel execution.

All 96 measured observations passed complete Python output comparisons and had
zero allocation requests in both kernel and encoder loop scopes. They cover both
datasets, all four routes, 1,000/10,000 cells, three fresh workers per size, and
actual four-worker checks. The [raw allocation evidence](performance/2026-11-06-column-allocation-evidence.json)
passes `tools/assemble_allocation_evidence.py` and the acceptance checker's
provenance validation. A rehashed nonzero encoder sample is rejected.

The reduced three-pair, one-warm-call screen puts typed Crown at 82–93 ms for
10,000 cells, down from 4.4–5.4 seconds before the writer. Python remains about
51–56 ms in those cases. Complete model fusion and general compact arithmetic
are still required; these figures do not certify final public-call targets.
