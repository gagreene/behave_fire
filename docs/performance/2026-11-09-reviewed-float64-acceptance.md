# Reviewed-source float64 Windows performance acceptance (2026-11-09)

**GO:** all 40 required cases and all 80 first/warmed confidence bounds pass.
Scalar regression budgets, peak-memory comparisons, complete output checks, and
ordinary scientific/encoder allocation gates also pass. No target or tolerance
was lowered. Ratios divide original-input Python time by Rust time: **larger
ratios mean faster Rust**.

The measured scientific source is commit
`9be8a50d9011438d4b61f219d60e156904659207`. The host has eight physical cores
and sixteen logical CPUs, Windows 10 build 19044, CPython 3.12.8, and NumPy 2.1.3.
The compiler and complete source/build inventory are in the allocation manifest.
Results certify this Windows host. Native wheel correctness on other platforms
does not establish their performance.

After measurement, running every standalone example found one obsolete assertion
expecting zero for an invalid raster cell. Only that assertion changed to expect
NaN. Neither library nor the benchmark/oracle changed. The
[example-only attestation](2026-11-09-reviewed-float64-example-attestation.json)
pins the exact one-line change and both file hashes. Original measured source
identities, binaries, reports, and complete arrays remain unchanged.

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
| uniform | surface | 10000 | 1 | 1.25 | 2.79 [2.22, 3.16] | 2.90 [2.56, 3.21] | not gated |
| uniform | surface | 100000 | 1 | 2.00 | 3.90 [3.88, 4.05] | 4.01 [3.92, 4.24] | 0.467 |
| uniform | surface | 100000 | 4 | 4.00 | 7.18 [6.63, 7.61] | 7.22 [6.97, 7.66] | 0.468 |
| uniform | surface | 1000000 | 1 | 2.00 | 5.48 [4.96, 5.67] | 5.52 [5.08, 5.62] | 0.288 |
| uniform | surface | 1000000 | 4 | 4.00 | 13.65 [12.95, 14.20] | 13.44 [12.17, 13.64] | 0.287 |
| uniform | crown | 10000 | 1 | 1.25 | 2.74 [2.57, 2.95] | 2.71 [2.43, 2.95] | not gated |
| uniform | crown | 100000 | 1 | 2.00 | 3.45 [3.35, 3.66] | 3.27 [3.14, 3.37] | 0.600 |
| uniform | crown | 100000 | 4 | 4.00 | 6.72 [6.58, 7.21] | 6.47 [6.34, 6.77] | 0.601 |
| uniform | crown | 1000000 | 1 | 2.00 | 4.81 [4.59, 4.90] | 4.73 [4.55, 4.84] | 0.475 |
| uniform | crown | 1000000 | 4 | 4.00 | 11.92 [11.52, 12.12] | 11.63 [11.24, 12.06] | 0.474 |
| uniform | typed-crown | 10000 | 1 | 1.00 | 2.89 [2.64, 3.00] | 2.76 [2.63, 2.94] | not gated |
| uniform | typed-crown | 100000 | 1 | 2.00 | 3.58 [3.26, 3.83] | 3.48 [3.41, 3.54] | 0.607 |
| uniform | typed-crown | 100000 | 4 | 4.00 | 6.96 [6.87, 7.13] | 6.70 [6.05, 6.97] | 0.607 |
| uniform | typed-crown | 1000000 | 1 | 2.00 | 4.71 [4.48, 5.04] | 4.81 [4.64, 4.90] | 0.482 |
| uniform | typed-crown | 1000000 | 4 | 4.00 | 12.06 [11.36, 12.67] | 11.83 [11.43, 12.01] | 0.481 |
| uniform | typed-crown-b | 10000 | 1 | 1.00 | 2.82 [2.68, 3.15] | 2.98 [2.90, 3.05] | not gated |
| uniform | typed-crown-b | 100000 | 1 | 2.00 | 3.48 [3.17, 3.74] | 3.46 [3.12, 3.62] | 0.607 |
| uniform | typed-crown-b | 100000 | 4 | 4.00 | 6.89 [6.43, 7.08] | 6.53 [6.24, 7.06] | 0.608 |
| uniform | typed-crown-b | 1000000 | 1 | 2.00 | 4.97 [4.75, 5.10] | 5.02 [4.86, 5.16] | 0.484 |
| uniform | typed-crown-b | 1000000 | 4 | 4.00 | 12.13 [11.12, 12.55] | 11.91 [11.50, 12.20] | 0.483 |
| heterogeneous | surface | 10000 | 1 | 1.25 | 2.68 [2.52, 2.79] | 2.60 [2.51, 2.72] | not gated |
| heterogeneous | surface | 100000 | 1 | 2.00 | 3.43 [3.28, 3.49] | 3.48 [3.37, 3.59] | 0.467 |
| heterogeneous | surface | 100000 | 4 | 4.00 | 6.81 [5.78, 6.96] | 6.93 [6.61, 7.01] | 0.468 |
| heterogeneous | surface | 1000000 | 1 | 2.00 | 4.58 [4.53, 4.90] | 4.49 [4.34, 4.69] | 0.289 |
| heterogeneous | surface | 1000000 | 4 | 4.00 | 11.74 [11.42, 12.10] | 11.57 [11.15, 12.25] | 0.288 |
| heterogeneous | crown | 10000 | 1 | 1.25 | 2.72 [2.54, 2.81] | 2.64 [2.29, 2.78] | not gated |
| heterogeneous | crown | 100000 | 1 | 2.00 | 3.45 [3.18, 3.52] | 3.26 [3.12, 3.35] | 0.920 |
| heterogeneous | crown | 100000 | 4 | 4.00 | 6.46 [6.30, 6.55] | 6.17 [6.01, 6.26] | 0.919 |
| heterogeneous | crown | 1000000 | 1 | 2.00 | 4.71 [4.46, 4.92] | 4.76 [4.55, 4.82] | 0.888 |
| heterogeneous | crown | 1000000 | 4 | 4.00 | 11.50 [10.95, 11.91] | 11.65 [10.83, 11.80] | 0.888 |
| heterogeneous | typed-crown | 10000 | 1 | 1.00 | 2.83 [2.42, 2.98] | 2.72 [2.63, 2.79] | not gated |
| heterogeneous | typed-crown | 100000 | 1 | 2.00 | 3.26 [3.15, 3.60] | 3.18 [3.12, 3.44] | 0.922 |
| heterogeneous | typed-crown | 100000 | 4 | 4.00 | 6.47 [6.23, 7.02] | 6.27 [6.12, 6.43] | 0.923 |
| heterogeneous | typed-crown | 1000000 | 1 | 2.00 | 4.75 [4.60, 5.06] | 4.90 [4.68, 5.02] | 0.894 |
| heterogeneous | typed-crown | 1000000 | 4 | 4.00 | 11.52 [11.22, 12.26] | 11.45 [11.06, 11.72] | 0.894 |
| heterogeneous | typed-crown-b | 10000 | 1 | 1.00 | 2.82 [2.73, 2.89] | 2.75 [2.65, 2.82] | not gated |
| heterogeneous | typed-crown-b | 100000 | 1 | 2.00 | 3.33 [3.20, 3.43] | 3.27 [3.14, 3.57] | 0.922 |
| heterogeneous | typed-crown-b | 100000 | 4 | 4.00 | 5.81 [5.47, 6.14] | 6.06 [5.98, 6.48] | 0.922 |
| heterogeneous | typed-crown-b | 1000000 | 1 | 2.00 | 4.79 [4.63, 5.14] | 4.72 [4.60, 4.93] | 0.892 |
| heterogeneous | typed-crown-b | 1000000 | 4 | 4.00 | 11.62 [11.29, 12.10] | 11.28 [10.90, 11.42] | 0.892 |

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
| uniform | surface | 5.471 | 3.429 | 4.064 | 2.783 |
| uniform | crown | 6.997 | 3.965 | 5.876 | 2.889 |
| uniform | typed-crown | 20.304 | 4.221 | 18.253 | 3.116 |
| uniform | typed-crown-b | 15.629 | 4.978 | 12.576 | 3.336 |
| heterogeneous | surface | 5.359 | 3.278 | 4.253 | 2.653 |
| heterogeneous | crown | 6.003 | 3.854 | 5.486 | 2.697 |
| heterogeneous | typed-crown | 14.198 | 4.276 | 11.085 | 3.261 |
| heterogeneous | typed-crown-b | 13.120 | 3.753 | 11.670 | 2.996 |

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
| uniform | surface | 1 | 8.982 | 99.035 | 64.059 | 8.144 | 0.070 |
| uniform | surface | 4 | 9.302 | 32.284 | 61.654 | 8.063 | 0.067 |
| uniform | crown | 1 | 11.050 | 135.017 | 91.022 | 9.876 | 0.092 |
| uniform | crown | 4 | 10.955 | 46.351 | 95.073 | 10.128 | 0.084 |
| uniform | typed-crown | 1 | 11.186 | 127.173 | 84.175 | 9.565 | 0.075 |
| uniform | typed-crown | 4 | 11.815 | 48.053 | 90.720 | 10.621 | 0.076 |
| uniform | typed-crown-b | 1 | 11.399 | 132.880 | 87.090 | 9.966 | 0.088 |
| uniform | typed-crown-b | 4 | 11.021 | 51.254 | 98.837 | 11.792 | 0.077 |
| heterogeneous | surface | 1 | 9.496 | 125.286 | 87.068 | 8.831 | 0.071 |
| heterogeneous | surface | 4 | 10.309 | 45.811 | 97.329 | 9.987 | 0.083 |
| heterogeneous | crown | 1 | 15.899 | 142.677 | 93.918 | 10.530 | 0.088 |
| heterogeneous | crown | 4 | 13.671 | 49.896 | 96.669 | 10.592 | 0.077 |
| heterogeneous | typed-crown | 1 | 13.595 | 128.633 | 83.266 | 9.385 | 0.079 |
| heterogeneous | typed-crown | 4 | 14.035 | 48.342 | 90.638 | 10.487 | 0.098 |
| heterogeneous | typed-crown-b | 1 | 13.922 | 129.405 | 85.838 | 9.663 | 0.080 |
| heterogeneous | typed-crown-b | 4 | 13.454 | 47.611 | 94.029 | 10.783 | 0.089 |

## Portable evidence

- [Acceptance records](2026-11-09-reviewed-float64-acceptance-results.json) and
  [strict checker result](2026-11-09-reviewed-float64-acceptance-check.json).
- [Matched builds](2026-11-09-reviewed-float64-allocation-builds.json),
  [raw allocation observations](2026-11-09-reviewed-float64-allocation-evidence.json), and
  [assembled allocation evidence](2026-11-09-reviewed-float64-allocations.json).
- [Preoptimization scalar baseline](2026-11-09-reviewed-float64-preoptimization-scalar.json),
  [scheduling pilot](2026-11-09-reviewed-float64-scheduling-pilot.json),
  [continuous checks](2026-11-09-reviewed-float64-continuous-correctness.json),
  [additional layout/containment cases](2026-11-09-reviewed-float64-auxiliary.json), and
  [separate profiles](2026-11-09-reviewed-float64-profiles.json).
- [Bundle and report hashes](2026-11-09-reviewed-float64-evidence-manifest.json).

Full actual and independent-reference arrays are compressed in the neighboring
`2026-11-09-reviewed-float64-*-arrays.tar.xz` bundles. Extract them at the repository root
with `tar -xf <bundle>` before running the checker on another checkout. Original
execution-relative paths and every NPZ digest remain unchanged. The reports are
byte-for-byte copies, not reconstructed measurements or checksums in place of
scientific arrays. Never overwrite an existing campaign with a rerun.

```text
python tools/check_performance.py --results docs/performance/2026-11-09-reviewed-float64-acceptance-results.json --requirements tools/performance_requirements.json --baseline-source-hash 66685bb82a32fa27cece497f559775acbd13ab1c6a4c6e31ce4948f54518e10f --baseline-extension-hash 8cc1a5543e1d36891cdeba4d5d4723311d3aabdab581272e389c3ffad7683c8a
```
