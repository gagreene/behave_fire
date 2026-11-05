# Rust backend execution measurements

Measured on 2026-11-04 against the integrated release build, with the complete backend enabled and exact integer-alias, object, empty, complex, and weak-scalar transport. These measurements supersede the earlier development tables.

## Method

Runs used Windows 10 x86-64, CPython 3.12.8, NumPy 2.1.3, a release Rust extension, and 16 logical CPUs. Each table entry is one fresh worker process, not a median or a performance guarantee. Model construction and input generation are excluded. First-call Python normalization, lazy setup, native copies, computation, output restoration, and allocation remain included.

Inputs are one-dimensional, normalized f64 arrays for fuel model 10, moisture fractions 0.06/0.07/0.08/0.6/0.9, wind 8 feet/minute, wind direction 35 degrees, and slope 12 degrees. Crown inputs use canopy base 4 feet, height 30 feet, bulk density 0.01 lb/ft^3, and foliar moisture 100 percent. Moisture arrays match the fuel grid shape.

Python and Rust receive identical inputs within each route. The typed Crown route supplies float32 spread rate, uint16 fireline intensity 200, and masked f64 heat per unit area, with every seventh position masked. Its timings are separate from the common f64 route.

Peak memory is the operating system lifetime peak working set of each isolated process, including interpreter, inputs, model state, and outputs. A 5 ms sampler also records current working set; the OS peak is the primary peak measure. Output byte counts exclude separately allocated masks.

`owned_copy_seconds` covers boundary validation and logical row-major copies into owned Rust vectors, including integer-bit metadata. `detached_seconds` includes native scalar record construction, allocation, per-cell metadata, worker setup/join, equations, concatenation, and GIL reacquisition. It is not pure equation time. Surface cells copy scalars and fuel records and construct owned input strings. Crown cells also construct a small native fuel registry. No array-sized input array-of-structs is staged.

`concatenate_seconds` is a subset of detached time. Appended byte counts measure explicit later-chunk vector appends; allocator relocation of the first chunk during final reservation is excluded. `numpy_transfer_seconds` measures field-wise ArrayD/NumPy ownership transfer. f64 vector buffers move without an output data copy, verified by core pointer identity tests. Crown classification conversion to the generic f64 adapter and then int32 Python output requires additional copies. Typed mask, dtype, integer-bit decoding, and shape restoration also allocate and are included in Python marshalling/output time.

## Total runtime

| Cells | Route | Python (s) | Rust 1 (s) | Rust 4 (s) | Rust 0 (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | surface | 0.002948 | 0.034484 | 0.035135 | 0.034264 |
| 1 | crown | 0.003070 | 0.041099 | 0.052584 | 0.049250 |
| 10,000 | surface | 0.041792 | 0.138138 | 0.075845 | 0.066202 |
| 10,000 | crown | 0.049908 | 0.152452 | 0.085650 | 0.071790 |
| 10,000 | typed-crown | 0.049483 | 4.183450 | 1.346176 | 0.655990 |
| 1,000,000 | surface | 4.421471 | 10.559714 | 3.464661 | 2.126346 |
| 1,000,000 | crown | 6.146485 | 13.106461 | 4.697492 | 2.556314 |

Python is faster for all measured small and 10,000-cell cases and for serial Rust at one million cells. Parallel Rust improves the measured common million-cell paths on this host. Typed supplied-Crown remains substantially slower than Python. No universal speed target or speedup is claimed. The expensive historical typed-million case was not repeated.

## Million-cell memory and native interval details

| Route | Backend / threads | Peak MiB | Owned copy (s) | Detached (s) | Concatenate (s) | NumPy transfer (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| surface | python / 1 | 1232.4 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| surface | rust / 1 | 580.4 | 0.283743 | 10.068248 | 0.000023 | 0.000183 |
| surface | rust / 4 | 604.2 | 0.255096 | 3.031267 | 0.067486 | 0.000205 |
| surface | rust / 0 | 589.8 | 0.301390 | 1.634693 | 0.081734 | 0.000194 |
| crown | python / 1 | 1343.4 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| crown | rust / 1 | 556.9 | 0.162319 | 12.859245 | 0.000035 | 0.000197 |
| crown | rust / 4 | 587.3 | 0.181494 | 4.444669 | 0.090455 | 0.000176 |
| crown | rust / 0 | 566.1 | 0.226604 | 2.246071 | 0.126699 | 0.000283 |

At 10,000 typed Crown cells, native output columns occupy 23,600,000 bytes, including 21,840,000 bytes of exact dtype, mask, scalar-kind, and integer-alias metadata. This allocation and per-cell encoding remain material costs.

## Crown routing correction

The final integration check found that the generic weak-scalar guard rejected ordinary Crown calls because unit controls, aspect, and foliar moisture carried Python scalar metadata. An intermediate 10,000-cell Crown run therefore took 3.7186 seconds serial and 0.68256 seconds with automatic workers. The reference explicitly normalizes aspect and foliar moisture to float64 arrays, and validates unit codes as controls. Crown fusion now permits only those four exact weak keys. Objects, masks, float32, integer-valued computational fields, and all other weak fields retain the checked typed route. The generic dispatch guard remains unchanged. New routing tests compare reference values, dtypes, masks, shapes, and signed infinities while verifying both fused and protected routes.

## Serial thresholds

The surface/crown pipeline threshold remains 1,024 cells; standalone helper dispatch uses 10,000 cells. These came from earlier local threshold experiments, not a new integrated threshold sweep. Generic `map_chunks` imposes no threshold and uses `effective_worker_count`. No threshold was tuned using these single-run final measurements.

## Reproduction and evidence

Run `tools/benchmark_backends.py --output results.json` from an environment containing a release native build. Run `--sizes 10000 --operations typed-crown` separately. Model construction and input generation are excluded; native boundary calls and metadata are recorded per invocation. Crown uses 11 native calls, including unit conversions.

Final raw evidence is preserved under `.superpowers/sdd/2026-11-03-rust-backend-port/`: `benchmarks-root-final-common.json`, `benchmarks-root-final-typed.json`, their logs, and `benchmarks-root-final-provenance.json`. That ignored local evidence is not distributed with the package. The tracked tables above record the results for reviewers.

Linux, macOS, other CPU families, other workload distributions, and repeated statistical measurements remain unmeasured. Numerical correctness is established separately by the native and reference suites; timing is not a correctness or ownership assertion.
