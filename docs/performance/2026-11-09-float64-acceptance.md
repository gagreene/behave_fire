# Final float64 Windows performance acceptance (2026-11-09)

**GO:** all 40 required cases and all 80 first/warmed confidence bounds pass.
Scalar regression budgets, peak-memory comparisons, complete output checks, and
ordinary scientific/encoder allocation gates also pass. No target or tolerance
was lowered. Ratios divide original-input Python time by Rust time: **larger
ratios mean faster Rust**.

The measured scientific source is commit
`fe43dce9f0389f3f25a7c787b291a5b8c30dcb61`. The host has eight physical cores
and sixteen logical CPUs, Windows 10 build 19044, CPython 3.12.8, and NumPy 2.1.3.
The compiler and complete source/build inventory are in the allocation manifest.
Results certify this Windows host. Native wheel correctness on other platforms
does not establish their performance.

A later metadata-only correction admits three existing diagnostic route labels
and prevents those labels from substituting for the four final acceptance routes.
The accepted request sets, inputs, timed call bodies, scientific code, oracle, and
native binary are unchanged. The [controlled source-diff attestation](2026-11-09-float64-metadata-attestation.json)
records this distinction; original observations retain their measured source identity.

## Protocol and result matrix

Both uniform and heterogeneous datasets cover Surface, Crown, and original mixed
input recipes A/B at 1, 10k, 100k, and 1M cells, using one and four workers. Seven
fresh paired blocks supply a first call and five warmed calls per worker. Each
block shares one explicit Python reference across the two Rust worker counts.
The full campaign contains 672 timing records and 64 independent complete-output
validations. Python timings use original caller dtype/mask data. Only the separate
untimed correctness oracle widens values and applies the accepted NaN rules.

The candidate, binary, harness, and inputs remained frozen. No build, test,
profiling, or competing benchmark workload from this task ran during timing.
Required ratios use paired bootstrap 95% intervals, with 5,000 resamples. Every
lower endpoint clears the unchanged threshold. Four-worker 10k and one-cell
measurements remain diagnostics; scalar budgets use the independently recorded
preoptimization native baseline. Complete arrays accompany this report.

| Dataset | Route | Cells | Workers | Target | First ratio [95% interval] | Warm ratio [95% interval] | Rust/Python peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | surface | 10000 | 1 | 1.25 | 3.20 [2.92, 3.35] | 3.14 [2.85, 3.48] | not gated |
| uniform | surface | 100000 | 1 | 2.00 | 4.18 [4.00, 4.32] | 3.92 [3.60, 4.53] | 0.467 |
| uniform | surface | 100000 | 4 | 4.00 | 7.95 [7.42, 8.23] | 7.85 [7.60, 8.42] | 0.467 |
| uniform | surface | 1000000 | 1 | 2.00 | 5.40 [5.00, 5.77] | 5.65 [5.30, 5.76] | 0.288 |
| uniform | surface | 1000000 | 4 | 4.00 | 13.79 [13.37, 14.40] | 13.47 [13.40, 14.23] | 0.287 |
| uniform | crown | 10000 | 1 | 1.25 | 2.72 [2.54, 3.21] | 2.81 [2.75, 2.95] | not gated |
| uniform | crown | 100000 | 1 | 2.00 | 3.23 [3.06, 3.36] | 3.32 [3.18, 3.58] | 0.600 |
| uniform | crown | 100000 | 4 | 4.00 | 6.61 [6.12, 6.89] | 6.62 [6.55, 7.25] | 0.602 |
| uniform | crown | 1000000 | 1 | 2.00 | 4.75 [4.66, 4.86] | 4.61 [4.54, 4.73] | 0.475 |
| uniform | crown | 1000000 | 4 | 4.00 | 12.23 [11.68, 12.73] | 11.94 [11.75, 12.05] | 0.473 |
| uniform | typed-crown | 10000 | 1 | 1.00 | 3.02 [2.60, 3.17] | 3.05 [2.65, 3.15] | not gated |
| uniform | typed-crown | 100000 | 1 | 2.00 | 3.59 [3.30, 3.82] | 3.52 [3.43, 3.60] | 0.608 |
| uniform | typed-crown | 100000 | 4 | 4.00 | 7.22 [6.55, 7.75] | 6.83 [6.62, 6.98] | 0.609 |
| uniform | typed-crown | 1000000 | 1 | 2.00 | 4.88 [4.62, 5.10] | 4.98 [4.85, 5.08] | 0.482 |
| uniform | typed-crown | 1000000 | 4 | 4.00 | 12.24 [11.77, 12.59] | 12.07 [11.37, 12.26] | 0.481 |
| uniform | typed-crown-b | 10000 | 1 | 1.00 | 2.92 [2.60, 3.32] | 2.98 [2.74, 3.38] | not gated |
| uniform | typed-crown-b | 100000 | 1 | 2.00 | 3.47 [3.01, 3.68] | 3.45 [3.15, 3.59] | 0.608 |
| uniform | typed-crown-b | 100000 | 4 | 4.00 | 6.84 [6.62, 7.03] | 6.80 [6.24, 7.16] | 0.609 |
| uniform | typed-crown-b | 1000000 | 1 | 2.00 | 5.12 [4.72, 5.18] | 4.95 [4.73, 5.12] | 0.484 |
| uniform | typed-crown-b | 1000000 | 4 | 4.00 | 12.03 [10.64, 12.75] | 12.12 [11.55, 12.48] | 0.483 |
| heterogeneous | surface | 10000 | 1 | 1.25 | 2.69 [2.67, 2.89] | 2.74 [2.68, 2.79] | not gated |
| heterogeneous | surface | 100000 | 1 | 2.00 | 3.61 [3.56, 3.71] | 3.53 [3.41, 3.59] | 0.467 |
| heterogeneous | surface | 100000 | 4 | 4.00 | 7.08 [6.74, 7.53] | 7.20 [7.03, 7.45] | 0.468 |
| heterogeneous | surface | 1000000 | 1 | 2.00 | 4.75 [4.48, 4.84] | 4.74 [4.49, 4.80] | 0.289 |
| heterogeneous | surface | 1000000 | 4 | 4.00 | 11.85 [11.40, 12.67] | 11.77 [11.49, 12.54] | 0.288 |
| heterogeneous | crown | 10000 | 1 | 1.25 | 2.60 [2.47, 2.83] | 2.80 [2.59, 3.26] | not gated |
| heterogeneous | crown | 100000 | 1 | 2.00 | 3.26 [3.08, 3.39] | 3.27 [3.16, 3.39] | 0.919 |
| heterogeneous | crown | 100000 | 4 | 4.00 | 5.86 [5.73, 6.32] | 6.06 [5.87, 6.45] | 0.919 |
| heterogeneous | crown | 1000000 | 1 | 2.00 | 4.70 [4.46, 4.82] | 4.71 [4.53, 4.98] | 0.888 |
| heterogeneous | crown | 1000000 | 4 | 4.00 | 11.19 [10.89, 11.82] | 11.28 [10.97, 11.57] | 0.888 |
| heterogeneous | typed-crown | 10000 | 1 | 1.00 | 2.67 [2.45, 2.84] | 2.60 [1.96, 2.82] | not gated |
| heterogeneous | typed-crown | 100000 | 1 | 2.00 | 3.55 [3.06, 3.64] | 3.43 [3.27, 3.58] | 0.922 |
| heterogeneous | typed-crown | 100000 | 4 | 4.00 | 6.59 [6.26, 6.80] | 6.39 [6.12, 6.74] | 0.922 |
| heterogeneous | typed-crown | 1000000 | 1 | 2.00 | 4.93 [4.59, 5.24] | 4.88 [4.77, 4.95] | 0.892 |
| heterogeneous | typed-crown | 1000000 | 4 | 4.00 | 11.62 [11.43, 12.30] | 11.50 [11.32, 11.85] | 0.892 |
| heterogeneous | typed-crown-b | 10000 | 1 | 1.00 | 2.86 [2.73, 3.02] | 2.73 [2.27, 2.87] | not gated |
| heterogeneous | typed-crown-b | 100000 | 1 | 2.00 | 3.56 [3.30, 3.66] | 3.48 [3.24, 3.67] | 0.921 |
| heterogeneous | typed-crown-b | 100000 | 4 | 4.00 | 6.55 [6.23, 6.70] | 6.26 [6.07, 6.48] | 0.921 |
| heterogeneous | typed-crown-b | 1000000 | 1 | 2.00 | 4.81 [4.62, 5.01] | 4.97 [4.60, 5.23] | 0.892 |
| heterogeneous | typed-crown-b | 1000000 | 4 | 4.00 | 11.20 [10.99, 11.97] | 11.25 [10.91, 11.86] | 0.892 |

Peak memory is process-lifetime resident memory, including prepared inputs and
construction. Untimed validation/oracle arrays run in other fresh processes.
Raw reports also retain pre-call RSS and sampled call peaks. Independent shape
groups can retain backing arrays larger than their exposed views; the process
measurements include those allocations. These numbers do not describe isolated
kernel scratch memory or certify explicit staged-session retention policies.

## Scalar regression budget

The baseline is copied byte-for-byte from the preoptimization campaign, with
independently declared source/binary identities. A regression must stay within
the larger of 10% or 0.25 ms. The table uses milliseconds.

| Dataset | Route | Old first | New first | Old warm | New warm |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform | surface | 5.471 | 3.263 | 4.064 | 2.723 |
| uniform | crown | 6.997 | 3.803 | 5.876 | 3.057 |
| uniform | typed-crown | 20.304 | 4.287 | 18.253 | 3.255 |
| uniform | typed-crown-b | 15.629 | 3.946 | 12.576 | 3.230 |
| heterogeneous | surface | 5.359 | 3.559 | 4.253 | 2.839 |
| heterogeneous | crown | 6.003 | 4.317 | 5.486 | 3.022 |
| heterogeneous | typed-crown | 14.198 | 3.874 | 11.085 | 3.339 |
| heterogeneous | typed-crown-b | 13.120 | 4.246 | 11.670 | 3.235 |

## Allocation, scheduling, and other input layouts

Matched release builds use identical sources and build settings, differing only
by the private allocation-probe feature. Ninety-six fresh allocation observations
cover eight route/dataset pairs at 1k/10k cells with one/four workers and three
repeats. Every observed scientific and encoder scope has zero requests and zero
requested bytes. Setup/output storage is counted separately. Actual-production
Rust probes also derive eligible visits from an explicit validity bitmap, verify
initialized skipped outputs, and exercise four workers. Twelve Rust buffer/grid
tests verify pointer preservation, while twelve production-module tests verify
actual scope behavior. The normal distributed extension contains no allocator
instrumentation.

One existing balanced-chunk/10k parallel-threshold setting was retained. The
scheduling pilot compared one, four, and automatic workers across all four routes
at 10k/100k, with 24 full-output validations. A single pilot pair is descriptive,
not certification. No dtype-specific fast path or numerical/ISA tuning was added.

Separate diagnostics add mask-only Crown, a third mix (float64 rate, masked
float32 intensity, uint32 heat), and variable-cost containment with 128 possible
steps. Fifty-four fresh timing workers cover 10k/100k and one/four/automatic
workers, followed by twelve separate allocation workers. Every complete output
comparison passes; every observed kernel/encoder allocation count is zero.
Sixteen additional untimed comparisons cover nonrepeating continuous inputs at
10k/100k across all four model routes and one/four workers. Continuous data is
not substituted for either accepted timing dataset.

## Separate attribution profiles

Eight standalone core probes separate scalar arithmetic from full f64 raster
storage. Sixteen instrumented boundary profiles cover all routes and datasets
with one/four workers at 100k. All use one model compute boundary, zero appended
bytes, and zero concatenation time. Calibration is absent from the float64 path.
Kernel and encoder scope times sum worker elapsed durations and can overlap in
parallel runs; they are not additive wall-time fractions. These profiles are
excluded from public timing evidence.

| Dataset | Route | Workers | Owned copy (ms) | Detached wall (ms) | Kernel scopes (ms) | Encoder scopes (ms) | NumPy transfer (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | surface | 1 | 9.031 | 90.369 | 58.024 | 7.670 | 0.086 |
| uniform | surface | 4 | 8.999 | 36.345 | 61.960 | 11.063 | 0.058 |
| uniform | crown | 1 | 10.650 | 134.644 | 90.371 | 9.970 | 0.079 |
| uniform | crown | 4 | 10.620 | 49.605 | 105.030 | 12.693 | 0.081 |
| uniform | typed-crown | 1 | 10.518 | 134.377 | 87.759 | 10.731 | 0.076 |
| uniform | typed-crown | 4 | 10.626 | 46.095 | 90.232 | 10.869 | 0.101 |
| uniform | typed-crown-b | 1 | 10.912 | 128.466 | 84.675 | 9.974 | 0.093 |
| uniform | typed-crown-b | 4 | 10.674 | 47.537 | 92.226 | 11.174 | 0.086 |
| heterogeneous | surface | 1 | 8.704 | 103.371 | 71.822 | 7.525 | 0.092 |
| heterogeneous | surface | 4 | 9.036 | 42.151 | 84.628 | 9.208 | 0.072 |
| heterogeneous | crown | 1 | 13.916 | 131.965 | 87.005 | 9.762 | 0.134 |
| heterogeneous | crown | 4 | 15.761 | 55.658 | 109.784 | 15.018 | 0.110 |
| heterogeneous | typed-crown | 1 | 14.377 | 133.374 | 87.500 | 10.370 | 0.096 |
| heterogeneous | typed-crown | 4 | 14.325 | 51.960 | 99.931 | 12.375 | 0.080 |
| heterogeneous | typed-crown-b | 1 | 13.728 | 130.703 | 85.945 | 9.911 | 0.080 |
| heterogeneous | typed-crown-b | 4 | 14.127 | 42.886 | 85.129 | 9.997 | 0.068 |

## Portable evidence

- [Acceptance records](2026-11-09-float64-acceptance-results.json) and
  [strict checker result](2026-11-09-float64-acceptance-check.json).
- [Matched builds](2026-11-09-float64-allocation-builds.json),
  [raw allocation observations](2026-11-09-float64-allocation-evidence.json), and
  [assembled allocation evidence](2026-11-09-float64-allocations.json).
- [Preoptimization scalar baseline](2026-11-09-float64-preoptimization-scalar.json),
  [scheduling pilot](2026-11-09-float64-scheduling-pilot.json),
  [continuous checks](2026-11-09-float64-continuous-correctness.json),
  [additional layout/containment cases](2026-11-09-float64-auxiliary.json), and
  [separate profiles](2026-11-09-float64-profiles.json).
- [Bundle and report hashes](2026-11-09-float64-evidence-manifest.json).

Full actual and independent-reference arrays are compressed in the neighboring
`2026-11-09-float64-*-arrays.tar.xz` bundles. Extract them at the repository root
with `tar -xf <bundle>` before running the checker on another checkout. Original
execution-relative paths and every NPZ digest remain unchanged. The reports are
byte-for-byte copies, not reconstructed measurements or checksums in place of
scientific arrays. Never overwrite an existing campaign with a rerun.

```text
python tools/check_performance.py --results docs/performance/2026-11-09-float64-acceptance-results.json --requirements tools/performance_requirements.json --baseline-source-hash 66685bb82a32fa27cece497f559775acbd13ab1c6a4c6e31ce4948f54518e10f --baseline-extension-hash 8cc1a5543e1d36891cdeba4d5d4723311d3aabdab581272e389c3ffad7683c8a
```
