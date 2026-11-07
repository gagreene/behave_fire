# Rust backend execution measurements

## Wind-cache confirmation checkpoint: 2026-11-07

The cache candidate passed all 32 bounds in a fresh core screen, then passed 31 of 32 in the required seven-new-pair confirmation. Every Crown/A/B bound passes. Uniform Surface at 100k misses its first-call lower confidence bound (2.241x versus 2.5x required; median 3.814x). Work stops before API4. The [checkpoint and concrete Surface follow-up](performance/2026-11-07-wind-cache-core-checkpoint.md), [fresh screen](performance/2026-11-07-wind-cache-core-screen.json), and [confirmation](performance/2026-11-07-wind-cache-core-confirmation.json) preserve the complete evidence. No public backend speed certification is implied.

## Bounded Crown investigation: 2026-11-07

An isolated immutable wind-term cache reduced warmed Crown core time by about 24% (1.31-1.32x relative to the existing Rust core) in fixed diagnostic pairs. Forward Surface and backsolve repeat physical wind powers; CPU pinning did not remove variance. See the [investigation and limitations](performance/2026-11-07-crown-profile.md) and [raw diagnostic evidence](performance/2026-11-07-crown-profile.json). The integrated candidate's fresh confirmation result is recorded above; these diagnostic ratios remain separate from acceptance.

## Float64 core checkpoint: 2026-11-07

The configured float64 core passes complete-output correctness checks, but its first headroom screen misses 3 of 32 confidence bounds. Work stops before the Python API4 switch. This is core-only evidence; the public backend still uses API3. See the [checkpoint, timings, and owner choices](performance/2026-11-07-f64-core-checkpoint.md) and [frozen raw screen](performance/2026-11-07-f64-core-screen.json). No confirmation or repeat sampling was run.

## Recovery pipeline checkpoint: 2026-11-07

**Task 4 stops at the serial-performance checkpoint.** Model calls now fuse normalization, numeric fuel preparation, shared native raster execution, and typed output into one native compute call. Ordinary Surface and Crown show faster median times. Both mixed Crown layouts remain slower than Python and miss the accepted targets. Task 5 parallel output work and final acceptance have not started.

The screen used Windows 10 x86-64, CPython 3.12.8, NumPy 2.1.3, 16 logical CPUs, three fresh paired processes per case, and one warmed call per process. It collected 240 timing records and passed 40 complete-output comparisons across both datasets, four routes, and 1/16/1,024/10,000/100,000 cells. Profiling was separate. This is exploratory evidence, not the required seven-pair/five-warm acceptance campaign.

[Raw screen](performance/2026-11-07-pipeline-screen.json), [paired first/warm summaries](performance/2026-11-07-pipeline-summary.json), and [separate diagnostic profiles](performance/2026-11-07-pipeline-profiles.json) retain measurements and identities. Ratios are **Python time divided by Rust time**: above 1 means Rust is faster.

### 10,000 cells: warmed serial medians

| Dataset | Route | Python ms | Rust ms | Speed ratio | Exploratory lower 95% |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform | surface | 35.92 | 15.57 | 2.31x | 2.25x |
| uniform | crown | 54.80 | 27.36 | 2.00x | 1.92x |
| uniform | typed-crown | 54.66 | 78.81 | 0.69x | 0.68x |
| uniform | typed-crown-b | 52.10 | 67.76 | 0.77x | 0.76x |
| heterogeneous | surface | 34.57 | 14.98 | 2.31x | 2.29x |
| heterogeneous | crown | 48.76 | 23.41 | 2.08x | 2.04x |
| heterogeneous | typed-crown | 49.67 | 74.24 | 0.67x | 0.65x |
| heterogeneous | typed-crown-b | 53.02 | 69.65 | 0.76x | 0.69x |

### 100,000 cells: warmed serial medians

| Dataset | Route | Python ms | Rust ms | Speed ratio | Exploratory lower 95% |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform | surface | 392.17 | 140.34 | 2.79x | 2.60x |
| uniform | crown | 504.71 | 207.70 | 2.43x | 2.28x |
| uniform | typed-crown | 517.20 | 749.96 | 0.69x | 0.56x |
| uniform | typed-crown-b | 511.69 | 673.38 | 0.76x | 0.61x |
| heterogeneous | surface | 373.33 | 161.22 | 2.32x | 2.32x |
| heterogeneous | crown | 593.13 | 220.30 | 2.69x | 1.80x |
| heterogeneous | typed-crown | 530.88 | 669.60 | 0.79x | 0.70x |
| heterogeneous | typed-crown-b | 517.70 | 666.97 | 0.78x | 0.72x |

The 10k targets are 1.25x for common routes and parity for typed routes; the large serial target is 2x for every route. Typed layouts fail clearly. The heterogeneous common Crown 100k lower bound also falls below 2x despite its faster median. Three pairs do not certify the common routes. Four-worker gains cannot substitute for the failed serial gate.

### Small-call latency

These uniform-data medians include call normalization, native work, output restoration, and lazy first-call setup. Imports, model construction, and input generation remain outside timing. Heterogeneous observations are retained in the linked summary.

| Route | Cells | Python first / warm ms | Rust first / warm ms |
| --- | ---: | ---: | ---: |
| surface | 1 | 2.480 / 1.814 | 3.028 / 1.863 |
| surface | 16 | 2.251 / 1.816 | 2.895 / 1.765 |
| surface | 1,024 | 6.203 / 5.462 | 4.248 / 3.028 |
| crown | 1 | 2.699 / 2.446 | 4.150 / 3.016 |
| crown | 16 | 2.757 / 2.417 | 4.083 / 3.458 |
| crown | 1,024 | 7.227 / 6.977 | 6.252 / 5.095 |
| typed-crown | 1 | 3.391 / 3.075 | 4.944 / 3.795 |
| typed-crown | 16 | 3.506 / 3.383 | 4.620 / 3.715 |
| typed-crown | 1,024 | 8.211 / 7.951 | 10.992 / 10.060 |
| typed-crown-b | 1 | 3.525 / 3.485 | 4.156 / 3.326 |
| typed-crown-b | 16 | 3.383 / 3.137 | 4.914 / 4.019 |
| typed-crown-b | 1,024 | 8.492 / 7.492 | 10.791 / 9.515 |

### Native diagnostic intervals

Each measured first and warmed model call used one native `batch` call. Optional caller-requested output-unit conversions remain separate. These cases do not exercise float32 direction calibration. The following single-run warmed profiles are diagnostic only; their instrumented durations were not pooled with timing observations.

| Route, 100k uniform | Owned copy ms | Detached ms | NumPy transfer ms |
| --- | ---: | ---: | ---: |
| surface | 11.18 | 121.62 | 0.033 |
| crown | 20.35 | 177.53 | 0.080 |
| typed-crown | 18.03 | 658.86 | 0.105 |
| typed-crown-b | 16.59 | 678.39 | 0.112 |

The typed cost is dominated by the detached Rust interval, which includes arithmetic, prepared work, typed encoding, and worker/merge overhead. This profile does not yet separate arithmetic from encoding. The compact route is reachable for both typed layouts; neither weak/object fallback nor repeated model boundaries explain these observations. A further bounded investigation should separate those native costs before selecting another optimization. The plan requires an owner decision before that additional tuning.

The timed source hash is `841d99b5f2d1a975bbc957f513592adf42ac0f715c4c51cf741ed26761cd05ef`; the release extension hash is `52a050f1755417022a4ba12b673c309843d9398e422dd1ad8cdb8fbb111cac71`. After the screen, review fixed ignored UserInput-WAF selector shapes, and the diagnostic harness was updated to count API-3 nested schema arrays. The diagnostic artifact records that later source identity with the same extension. The screen is retained as evidence for its recorded source, not claimed as final latest-head acceptance. No Linux or other-platform speed guarantee is established.

## Historical measurements before recovery

The later [core recovery confirmation](RUST_ENGINE.md#recovery-core-confirmation-2026-11-06) passes the native headroom gate. The historical Python-boundary measurements below still describe the transport being replaced; end-to-end speed is not yet certified.

Measured on 2026-11-04 against the final ownership-fix release build based on `7d3fb2a0dbdc73d8433a6638607e1f71703f6793`, with binding API 2, the complete backend enabled, and exact integer-alias, object, empty, complex, and weak-scalar transport. The Windows calibration selects the platform-libm atan2 profile; Linux SVML timing is not measured here. These measurements supersede the earlier development tables.

## Method

Runs used Windows 10 x86-64, CPython 3.12.8, NumPy 2.1.3, a release Rust extension, and 16 logical CPUs. Each table entry is one fresh worker process, not a median or a performance guarantee. Model construction and input generation are excluded. First-call Python normalization, lazy setup, native copies, computation, output restoration, and allocation remain included.

Inputs are one-dimensional, normalized f64 arrays for fuel model 10, moisture fractions 0.06/0.07/0.08/0.6/0.9, wind 8 feet/minute, wind direction 35 degrees, and slope 12 degrees. Crown inputs use canopy base 4 feet, height 30 feet, bulk density 0.01 lb/ft^3, and foliar moisture 100 percent. Moisture arrays match the fuel grid shape.

Python and Rust receive identical inputs within each route. The typed Crown route supplies float32 spread rate, uint16 fireline intensity 200, and masked f64 heat per unit area, with every seventh position masked. Its timings are separate from the common f64 route.

Peak memory is the operating system lifetime peak working set of each isolated process, including interpreter, inputs, model state, and outputs. A 5 ms sampler also records current working set; the OS peak is the primary peak measure. Output byte counts exclude separately allocated masks.

`owned_copy_seconds` covers boundary validation and logical row-major copies into owned Rust vectors, including integer-bit metadata. `detached_seconds` includes native scalar record construction, allocation, per-cell metadata, worker setup/join, equations, concatenation, and GIL reacquisition. It is not pure equation time. Surface cells copy scalars and fuel records and construct owned input strings. Crown cells also construct a small native fuel registry. No array-sized input array-of-structs is staged.

`concatenate_seconds` is a subset of detached time. Appended byte counts measure explicit later-chunk vector appends; allocator relocation of the first chunk during final reservation is excluded. `numpy_transfer_seconds` measures field-wise ArrayD/NumPy ownership transfer. f64 vector buffers move without an output data copy, verified by core pointer identity tests. Crown classification conversion to the generic f64 adapter and then int32 Python output requires additional copies. Typed mask, dtype, integer-bit decoding, and shape restoration also allocate and are included in Python marshalling/output time.

These surface and Crown routes do not exercise first-call float32 direction-profile calibration; its startup cost is not separately timed. Imports, model construction, and input generation are excluded.

## Total runtime

| Cells | Route | Python (s) | Rust 1 (s) | Rust 4 (s) | Rust 0 (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | surface | 0.002547 | 0.005377 | 0.005685 | 0.005463 |
| 1 | crown | 0.002610 | 0.006675 | 0.006144 | 0.005998 |
| 10,000 | surface | 0.037260 | 0.098920 | 0.057441 | 0.047504 |
| 10,000 | crown | 0.048238 | 0.153905 | 0.064425 | 0.055618 |
| 10,000 | typed-crown | 0.054994 | 4.303325 | 1.594120 | 0.762723 |
| 1,000,000 | surface | 4.677120 | 10.489552 | 3.320992 | 2.086901 |
| 1,000,000 | crown | 6.898933 | 14.789995 | 5.364704 | 2.653753 |

Python is faster for all measured small and 10,000-cell cases and for serial Rust at one million cells. Parallel Rust improves the measured common million-cell paths on this host. Typed supplied-Crown remains substantially slower than Python. No universal speed target or speedup is claimed. The expensive historical typed-million case was not repeated.

## Million-cell memory and native interval details

| Route | Backend / threads | Peak MiB | Owned copy (s) | Detached (s) | Concatenate (s) | NumPy transfer (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| surface | python / 1 | 1235.8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| surface | rust / 1 | 647.8 | 0.312624 | 10.015841 | 0.000032 | 0.000188 |
| surface | rust / 4 | 671.8 | 0.259630 | 2.891386 | 0.063877 | 0.000153 |
| surface | rust / 0 | 658.4 | 0.261920 | 1.643790 | 0.070855 | 0.000163 |
| crown | python / 1 | 1346.9 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| crown | rust / 1 | 732.0 | 0.172222 | 14.510820 | 0.000035 | 0.000174 |
| crown | rust / 4 | 762.1 | 0.218318 | 5.018626 | 0.093106 | 0.000191 |
| crown | rust / 0 | 738.9 | 0.213561 | 2.321464 | 0.100715 | 0.000199 |

At 10,000 typed Crown cells, native output columns occupy 23,600,000 bytes, including 21,840,000 bytes of exact dtype, mask, scalar-kind, and integer-alias metadata. This allocation and per-cell encoding remain material costs.

## Public-call snapshot costs

This campaign includes the final shared public-call snapshot and strict
masked-division zero restoration. The snapshot copies caller arrays and nested
containers before the first native stage, then reuses owned buffers and views.
It adds Python-side copying and memoization and retains inputs/intermediates
until the call returns. These costs are included in total runtime and process
peak memory. Native "Owned copy" intervals above measure only the Rust boundary,
not the earlier Python snapshot.

The following residual interval includes snapshot work, normalization, and output
restoration; it does not isolate copying or establish a causal speed difference
from the earlier single-run campaign.

| Million-cell route | Threads | Python marshalling and output (s) |
| --- | ---: | ---: |
| surface | 1 | 0.137176 |
| surface | 4 | 0.145238 |
| surface | 0 | 0.154664 |
| crown | 1 | 0.091172 |
| crown | 4 | 0.106489 |
| crown | 0 | 0.099823 |

The provenance file hashes the exact release binary, ownership code, bridge, and relevant Rust implementation. The commit containing these tables records the measured source; no earlier-head artifact is relabeled.

## Crown routing correction

The final integration check found that the generic weak-scalar guard rejected ordinary Crown calls because unit controls, aspect, and foliar moisture carried Python scalar metadata. An intermediate 10,000-cell Crown run therefore took 3.7186 seconds serial and 0.68256 seconds with automatic workers. The reference explicitly normalizes aspect and foliar moisture to float64 arrays, and validates unit codes as controls. Crown fusion now permits only those four exact weak keys. Objects, masks, float32, integer-valued computational fields, and all other weak fields retain the checked typed route. The generic dispatch guard remains unchanged. New routing tests compare reference values, dtypes, masks, shapes, and signed infinities while verifying both fused and protected routes.

## Serial thresholds

The surface/crown pipeline threshold remains 1,024 cells; standalone helper dispatch uses 10,000 cells. These came from earlier local threshold experiments, not a new integrated threshold sweep. Generic `map_chunks` imposes no threshold and uses `effective_worker_count`. No threshold was tuned using these single-run final measurements.

## Performance investigation: 2026-11-05

The slowdown is reproducible. The largest measured problem is native result
packaging, followed by avoidable per-cell setup. Python uses compiled NumPy
array operations; replacing them with a scalar Rust loop does not by itself
improve performance.

The investigation used the same host, interpreter, NumPy, release extension,
and benchmark inputs described above. The checkout was based on
`d37d400cd2d90f307e64b3b47f0d2da235f9a199`, with an uncommitted Python input
validation fix. No production optimization was applied. These fresh-process
results confirm the earlier pattern; they do not replace the historical tables.

| 10,000-cell route | Python (s) | Rust 1 (s) | Rust 4 (s) | Rust 0 (s) |
| --- | ---: | ---: | ---: | ---: |
| surface | 0.033918 | 0.110720 | 0.060278 | 0.043474 |
| crown | 0.045716 | 0.131147 | 0.061859 | 0.046484 |
| typed-crown | 0.051588 | 3.748687 | 1.201821 | 0.616116 |

Serial typed Crown spent 3.725172 seconds inside detached native execution,
0.002611 seconds validating/copying inputs at the native boundary, and
0.020139 seconds in Python marshalling/output restoration. These intervals do
not identify Python execution or boundary copying as the main bottleneck.

### Isolated native measurements

Disposable Rust consumers measured the existing standalone core and exact
copies of production binding helpers. Timing builds used release optimization
without allocation instrumentation. Separate builds counted allocation and
reallocation calls through the system allocator. Entire computed records were
passed through `black_box` before selecting checksum fields.

| Isolated operation | Cells | Median runtime (s) | Allocation/reallocation calls per cell |
| --- | ---: | ---: | ---: |
| Surface core with prepared input/fuel | 100,000 | 0.144506 | 0 |
| Crown core with prepared input/registry | 100,000 | 0.622957 | 9 |
| Typed Crown core with prepared inputs/registry | 10,000 | 0.114412 | 9 |
| Typed Crown output encoder alone | 10,000 | 2.852865 | 679 |
| Surface binding record setup alone | 100,000 | 0.473416 | 7 |
| Crown binding record/registry setup alone | 100,000 | 0.443193 | 5 |

Core/encoder timings are medians of three repetitions; setup timings are medians
of five. Allocation counts used 1,000 core/encoder cells and 10,000 setup cells.
Core inputs and registry are reused. The typed core case uses a representative
masked heat value in every cell, rather than the public benchmark's every-seventh
mask. The encoder reuses one computed typed result, runs the exact production
`masked::outputs` helper, adds classification, and destroys the result map each
iteration. Setup probes use representative model-10 argument maps, without all
public-call transport metadata. Core runs omit output-column storage and Python
transport. These are attribution experiments, not end-to-end speed claims;
their runtimes cannot be subtracted to infer exact production time shares.

### Causes supported by the measurements and source

1. **Typed output serialization creates excessive allocations.**
   [`masked::output` and `masked::outputs`](../rust/crates/behave-py/src/masked.rs)
   construct string-keyed maps for every field of every cell. Twenty-one fields
   each produce one payload and thirteen metadata columns; classification brings
   the total to 295 columns. The measured encoder makes 679 allocation or
   reallocation calls and requests 71,954 bytes per cell. At 10,000 cells that is
   6.79 million calls and approximately 719.5 MB of cumulative requested
   allocation, including temporary maps and strings. This is not peak memory.
   Final native columns occupy 23.6 MB; 21.84 MB (92.54%) is metadata. Encoding
   alone is much more expensive than the isolated typed equations.
2. **Common float64 calls rebuild inputs inside the cell loop.**
   [`structures::input` and `structures::fuel`](../rust/crates/behave-py/src/structures.rs)
   repeatedly read named maps and rebuild records. Surface constructs three
   default mode strings and three replacements per cell. Its absent optional
   `user_waf` also formats an error string that is immediately discarded by
   [`Arguments::optional`](../rust/crates/behave-py/src/arrays.rs).
   [`Crown setup`](../rust/crates/behave-py/src/crown.rs) constructs an input and
   a one-record fuel registry per cell. Measured setup costs are substantial
   even before equations run.
3. **Crown contains duplicated equation stages.**
   [`calculate_crown_surface_fire`](../rust/crates/behave-fire/src/crown.rs)
   computes particles, fuelbed intermediates, and reaction intensity, then calls
   `surface::run`, which requests the same stages again. The
   [Python implementation](../src/behave_fire/components/crown.py) reuses its
   intermediates through `run_surface_fire`. The source duplication is confirmed;
   its retained runtime cost after compiler optimization was not isolated.
4. **Typed scalar machinery carries additional work.**
   `MaskedValue` measures 216 bytes on this target, including inline capacity
   for wide integers. Typed arithmetic and input decoding repeatedly handle
   precision, masks, integer aliases, and scalar kinds. This is another profiling
   target, but no measured time share is assigned to it here.

The first optimization should replace per-cell output maps with a schema
prepared once per batch and direct writes into output columns. Next, bind input
columns and constant controls once, reuse fuel records where possible, and reuse
Crown intermediates. Compact constant metadata and specialize common typed
layouts only while preserving existing dtype, mask, integer, error, broadcasting,
and ownership behavior. SIMD/math-kernel investigation comes after these costs
are measured again. None of these proposed changes has yet established an
end-to-end speedup.

Investigation evidence is preserved locally under
`.superpowers/perf-investigation/`: `wrapper-10k.json`,
`core-timing-final.jsonl`, `core-allocations-final.jsonl`,
`f64-setup-*-timing.jsonl`, and `f64-setup-*-allocations.jsonl`, plus the disposable
`corebench/` and `f64-setup/` consumers. These ignored artifacts are not packaged.

## Reproduction and evidence

Run `tools/benchmark_backends.py --output results.json` from an environment containing a release native build. Run `--sizes 10000 --operations typed-crown` separately. Model construction and input generation are excluded; native boundary calls and metadata are recorded per invocation. Crown uses 11 native calls, including unit conversions.

Final raw evidence is preserved under `.superpowers/sdd/2026-11-03-rust-backend-port/`: `benchmarks-fix-common.json`, `benchmarks-fix-typed.json`, their logs, and `benchmarks-fix-provenance.json`. That ignored local evidence is not distributed with the package. The tracked tables above record the results for reviewers.

Linux, macOS, other CPU families, other workload distributions, and repeated statistical measurements remain unmeasured. Numerical correctness is established separately by the native and reference suites; timing is not a correctness or ownership assertion.
