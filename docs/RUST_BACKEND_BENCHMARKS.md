# Rust backend execution measurements

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

## Reproduction and evidence

Run `tools/benchmark_backends.py --output results.json` from an environment containing a release native build. Run `--sizes 10000 --operations typed-crown` separately. Model construction and input generation are excluded; native boundary calls and metadata are recorded per invocation. Crown uses 11 native calls, including unit conversions.

Final raw evidence is preserved under `.superpowers/sdd/2026-11-03-rust-backend-port/`: `benchmarks-fix-common.json`, `benchmarks-fix-typed.json`, their logs, and `benchmarks-fix-provenance.json`. That ignored local evidence is not distributed with the package. The tracked tables above record the results for reviewers.

Linux, macOS, other CPU families, other workload distributions, and repeated statistical measurements remain unmeasured. Numerical correctness is established separately by the native and reference suites; timing is not a correctness or ownership assertion.
