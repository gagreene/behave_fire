# Native Rust raster engine performance contract

The performance work replaces unreleased Rust interfaces with typed calculation
stages and raster columns. Python-facing calculations, shape/dtype/mask behavior,
and scientific tolerances remain the reference contract. The Rust core remains
self-contained and has no normal/build dependencies.

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
Task 3 extends the probe to native raster output storage before the feasibility
checkpoint.

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
