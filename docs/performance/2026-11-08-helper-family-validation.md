# Float64 helper family validation (2026-11-08)

Task 5 replaces legacy component metadata and numeric emulation with owned
primitive columns. These measurements compare that change with commit
`182d47842c7267fb6f03b7bf12061fa267e604e2`. They do not compare Rust with Python,
and they do not certify the final model throughput targets in Task 6.

The same Windows host, CPython 3.12.8, and NumPy 2.1.3 ran both sides. For each
size, three fresh uninstrumented processes recorded a first call and three warm
calls per family. Allocation-only processes ran separately. No build or test
workloads ran during the timing campaign. Medians below pool nine warmed calls;
first-call medians use three observations. This small descriptive campaign has
no confidence-interval certification claim.

The 35 family/size cases cover all helper domains at 1/10,000/100,000 cells.
FireSize and SlopeTool expose scalar state APIs, so their cases use one cell;
vector geometry is covered separately. The driver preserves the frozen baseline
and each candidate package. Every candidate is retained, including regressions.
Run `tools/benchmark_component_families.py --site <regular-package-parent>
--probe-site <allocation-package-parent> --output <report.json>` from the root
with the configured Python executable. Build the probe extension separately with
the `allocation-probe` Cargo feature. The script verifies the loaded package path
and expected presence or absence of that feature in every worker.

## Latency and payloads

Times are milliseconds. A ratio above 1 means the new helper call is faster.
Physical outputs now use float64, so a previously float32 output can need more
bytes even when transport and calculation are faster. Output bytes count actual
independently shaped fields, not a fabricated common-grid result.

| Family | Cells | Before warm | After warm | Before/after | Before first | After first | Output bytes before/after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| surface.geometry | 1 | 0.2019 | 0.1434 | 1.41 | 0.9607 | 0.6740 | 4/8 |
| surface.direction | 1 | 0.4161 | 0.1987 | 2.09 | 0.6826 | 0.2873 | 4/8 |
| crown.component | 1 | 0.5512 | 0.3678 | 1.50 | 0.6154 | 0.4099 | 40/40 |
| units | 1 | 0.3015 | 0.1972 | 1.53 | 0.4603 | 0.2784 | 8/8 |
| mortality | 1 | 0.5923 | 0.4164 | 1.42 | 0.6570 | 0.6842 | 24/24 |
| vpd | 1 | 0.7752 | 0.3329 | 2.33 | 0.7959 | 0.4417 | 24/24 |
| fdfm | 1 | 0.9429 | 0.3804 | 2.48 | 0.8695 | 0.4415 | 8/8 |
| containment | 1 | 0.5239 | 0.3059 | 1.71 | 0.6389 | 0.3325 | 28/28 |
| ignite | 1 | 0.2561 | 0.1707 | 1.50 | 0.2981 | 0.1964 | 8/8 |
| safety | 1 | 0.3420 | 0.2507 | 1.36 | 0.4495 | 0.2999 | 24/24 |
| spot | 1 | 0.3840 | 0.1536 | 2.50 | 0.4590 | 0.2337 | 8/8 |
| fire_size.scalar | 1 | 0.1195 | 0.1593 | 0.75 | 0.1764 | 0.2476 | 8/8 |
| slope.scalar | 1 | 0.2288 | 0.3525 | 0.65 | 0.2760 | 0.4492 | 0/0 |
| surface.geometry | 10000 | 8.2495 | 0.5647 | 14.61 | 7.9756 | 1.4170 | 40000/80000 |
| surface.direction | 10000 | 11.5316 | 1.6753 | 6.88 | 10.0055 | 1.6340 | 40000/80000 |
| crown.component | 10000 | 18.6042 | 1.1755 | 15.83 | 20.0865 | 1.4921 | 160024/160024 |
| units | 10000 | 0.6790 | 0.2424 | 2.80 | 1.1738 | 0.4140 | 80000/80000 |
| mortality | 10000 | 4.5053 | 2.6525 | 1.70 | 4.9894 | 1.8903 | 240000/240000 |
| vpd | 10000 | 2.2780 | 1.0408 | 2.19 | 2.4164 | 1.5546 | 240000/240000 |
| fdfm | 10000 | 1.9772 | 0.9194 | 2.15 | 2.9092 | 1.0185 | 80000/80000 |
| containment | 10000 | 5.9316 | 3.7959 | 1.56 | 4.2354 | 3.6813 | 280000/280000 |
| ignite | 10000 | 2.5363 | 2.0122 | 1.26 | 2.8950 | 2.3684 | 80000/80000 |
| safety | 10000 | 1.8908 | 0.5384 | 3.51 | 1.5639 | 0.7931 | 240000/240000 |
| spot | 10000 | 3.5568 | 2.8979 | 1.23 | 3.6005 | 2.8929 | 80000/80000 |
| surface.geometry | 100000 | 72.2741 | 5.8277 | 12.40 | 73.7963 | 7.0815 | 400000/800000 |
| surface.direction | 100000 | 106.0128 | 14.6673 | 7.23 | 101.7585 | 17.4526 | 400000/800000 |
| crown.component | 100000 | 169.6936 | 11.3025 | 15.01 | 166.3348 | 11.0255 | 1600024/1600024 |
| units | 100000 | 2.6962 | 2.5744 | 1.05 | 3.2427 | 3.5325 | 800000/800000 |
| mortality | 100000 | 29.6082 | 17.6243 | 1.68 | 30.8151 | 16.9904 | 2400000/2400000 |
| vpd | 100000 | 18.6131 | 11.7870 | 1.58 | 18.4727 | 11.5771 | 2400000/2400000 |
| fdfm | 100000 | 14.3887 | 11.3689 | 1.27 | 14.6969 | 7.6702 | 800000/800000 |
| containment | 100000 | 40.6452 | 36.0319 | 1.13 | 40.7896 | 32.4744 | 2800000/2800000 |
| ignite | 100000 | 18.4397 | 17.5830 | 1.05 | 18.4458 | 17.2424 | 800000/800000 |
| safety | 100000 | 17.6433 | 6.1323 | 2.88 | 13.5100 | 6.3847 | 2400000/2400000 |
| spot | 100000 | 35.1885 | 32.7800 | 1.07 | 35.3835 | 30.4295 | 800000/800000 |

## Regression investigation

The declared investigation gate is a median increase exceeding both 10% and
0.25 ms. Candidate 1 triggered it for scalar SlopeTool and 100k unit conversion;
candidate 2 still triggered it for units. Separate profiling identified recursive
state snapshot copies and masked-array construction for plain scale-table lists.
State role normalizers now take one owned snapshot before native detachment.
Plain real lists avoid masked-array construction. The unit producer uses its
predecoded selector and a direct safe final slice, and normalized C-contiguous
native inputs use bulk owned copies. Strided inputs keep logical C-order copying.

Neither first-call nor warmed medians in the final candidate trigger the gate.
Scalar FireSize and SlopeTool remain slower than this baseline, with absolute
warm increases below 0.25 ms; the table reports those differences explicitly.
The final 100k unit warm median improves, while its first-call median increases
by less than 10%. No threshold or scientific tolerance was lowered.

## Memory and allocation evidence

Raw files and the summary include each family's current resident memory before
and after calls, process-lifetime peak resident memory, and actual output bytes.
These are process diagnostics, not isolated per-call peak memory: families run
sequentially within each worker, and allocator retention affects RSS. The final
model memory acceptance campaign remains separate.

The allocation records contain outside/kernel/encoder scopes, each as
`[requests, requested bytes, observed calls, distinct workers]`. Every final
component family has observed kernel and encoder calls with zero allocation
requests and zero requested bytes. The separate actual-production-module probe
also verifies scientific/encoder zero allocations on real four-worker paths.
State helpers have no scientific/encoder scope instrumentation; their scoped
zeros are not allocation evidence. The old component producer likewise had zero
observed scoped calls, so only its all-call outside counts are meaningful.

| Family at 100k | Before all-call requests | After all-call requests | After kernel calls | After encoder calls | After kernel/encoder requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| surface.geometry | 33 | 25 | 100000 | 100000 | 0/0 |
| surface.direction | 56 | 22 | 100000 | 100000 | 0/0 |
| crown.component | 106 | 93 | 100002 | 200003 | 0/0 |
| units | 30 | 16 | 100000 | 100000 | 0/0 |
| mortality | 63 | 40 | 100000 | 300000 | 0/0 |
| vpd | 71 | 38 | 100000 | 300000 | 0/0 |
| fdfm | 163 | 42 | 100000 | 100000 | 0/0 |
| containment | 74 | 48 | 100000 | 400000 | 0/0 |
| ignite | 30 | 19 | 100000 | 100000 | 0/0 |
| safety | 47 | 34 | 100000 | 300000 | 0/0 |
| spot | 30 | 19 | 100000 | 100000 | 0/0 |

## Evidence and correctness

- [Frozen before observations](2026-11-08-helper-family-before.json).
- [Candidate 1](2026-11-08-helper-family-candidate1.json) and
  [candidate 2](2026-11-08-helper-family-candidate2.json), retained for the regression investigation.
- [Final observations](2026-11-08-helper-family-after.json).
- [Summary, memory diagnostics, hashes, and per-family gates](2026-11-08-helper-family-summary.json).

Correctness runs are separate from timings. Original Python code and scientific
fixtures remain unchanged. Native tests compare finite science with independently
widened Python inputs, and explicitly test reviewed NoData rules, typed controls,
empty/dependency-local shapes, complete state, byte order, and ownership. The
surviving Rust-only f64 probe validates every final column outside timed regions
for Surface, Crown, and both formerly typed layouts across uniform,
heterogeneous, and continuous inputs. No retired compact or masked arithmetic
is needed to run that probe.
