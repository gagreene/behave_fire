# Wind-cache core confirmation checkpoint

**STOP before API4.** The integrated cache passed the new three-pair screen (32/32 bounds), then missed one bound in the required seven-new-pair confirmation (31/32 bounds). All Crown, mixed Crown A, and mixed Crown B bounds pass. Uniform Surface at 100k misses its first-call lower confidence requirement.

## Frozen candidate and procedure

- Measured commit: `2a02304ceecc0e7ba0a3e6149b0f2543ea997e9f`. This checkpoint adds evidence only; the measured scientific source stays unchanged.
- Source SHA256: `b3ad00f7adc1222c0a38aab3348087b506dc8c7be5de4fdf5ed74df4f8e29a6e`.
- Release probe SHA256: `6d6696deaf9a5154dc101d012ff183897ed581ad101937c6fba1f85465c9af72`.
- Same Windows x86-64 host, configured CPython 3.12.8, NumPy 2.1.3, compiler, binary, and sources for both phases. Three fresh pairs/four warm calls in the screen; seven NEW pairs/five warm calls in confirmation. No pooling, dropped outliers, extra candidates, or repeated confirmation.
- Both phases cover Surface/Crown/A/B, uniform and heterogeneous, 10k and 100k. Every stored output passes independent complete f64/NoData validation. Continuous layouts are correctness diagnostics. Python timing retains the original unnormalized public inputs.
- [Fresh screen raw evidence](2026-11-07-wind-cache-core-screen.json), [independent confirmation raw evidence](2026-11-07-wind-cache-core-confirmation.json), and [bounded investigation/candidate evidence](2026-11-07-crown-profile.md). All worker logs and typed exports remain in the ignored workspace paths in the reports.
- This is actual-core timing with complete final storage. Public Python/Rust dispatch, boundary copy/normalization, output transfer, and wheel certification have not earned speed approval.

## 100k confirmation

Ratios are public Python time divided by native core time: greater than one means the core is faster. Both first and warm lower 95% bounds must reach 2.5x, leaving 25% headroom above the unchanged 2x public serial target.

| Dataset | Route | Phase | Median ratio | Lower 95% | Required |
| --- | --- | --- | ---: | ---: | ---: |
| uniform | surface | first | 3.814x | 2.241x | 2.500x |
| uniform | surface | warm | 3.798x | 3.357x | 2.500x |
| uniform | crown | first | 3.906x | 3.644x | 2.500x |
| uniform | crown | warm | 4.077x | 3.756x | 2.500x |
| uniform | typed-crown | first | 4.547x | 4.031x | 2.500x |
| uniform | typed-crown | warm | 4.230x | 3.612x | 2.500x |
| uniform | typed-crown-b | first | 4.001x | 3.728x | 2.500x |
| uniform | typed-crown-b | warm | 4.282x | 3.504x | 2.500x |
| heterogeneous | surface | first | 3.407x | 3.251x | 2.500x |
| heterogeneous | surface | warm | 3.620x | 3.109x | 2.500x |
| heterogeneous | crown | first | 4.013x | 3.613x | 2.500x |
| heterogeneous | crown | warm | 4.127x | 3.837x | 2.500x |
| heterogeneous | typed-crown | first | 4.326x | 4.157x | 2.500x |
| heterogeneous | typed-crown | warm | 4.488x | 4.187x | 2.500x |
| heterogeneous | typed-crown-b | first | 4.295x | 4.170x | 2.500x |
| heterogeneous | typed-crown-b | warm | 4.216x | 4.024x | 2.500x |

All 10k bounds pass. Every Crown bound passes in this independent confirmation. The overall gate still fails because every required bound must pass.

## Remaining Surface variance

| Block | Python first ms | Rust core first ms | Python warm median ms | Rust warm median ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 381.87 | 100.19 | 448.17 | 101.57 |
| 1 | 411.81 | 95.67 | 393.19 | 96.38 |
| 2 | 372.43 | 100.44 | 381.36 | 95.90 |
| 3 | 375.21 | 167.41 | 382.65 | 113.99 |
| 4 | 383.03 | 93.84 | 384.83 | 103.52 |
| 5 | 388.73 | 181.56 | 398.75 | 116.88 |
| 6 | 424.09 | 101.73 | 470.20 | 103.58 |

Two native first calls are 167.4ms and 181.6ms; the others are 93.8-101.7ms. Those observations remain included. The first-call median speedup is 3.814x, but its lower 95%bound is 2.241x, below 2.5x. Warmed Surface passes with lower 3.357x. The confirmation alone does not identify the cause of its latency spikes; the separate Crown diagnostics showed that CPU pinning did not eliminate variance.

## Verification and scope

The exact measured commit passed 6,149 Python tests with zero skips and 27 subtests in 197.21s. Stable/MSRV core tests, 36 native production-module tests, Clippy, rustdoc, standalone tiles, and 12 independent full-output recipes pass. Three new wind-cache tests cover exact ordinary arithmetic, thresholds/NaNs/infinities/signed zero, physical edits, dynamic fallback, and actual fuelbed key invalidation. Real configured kernel/encoder allocation scopes remain zero. One initial full-suite run encountered a Windows temporary-venv executable cleanup permission error; the unchanged isolated test and full retry passed, and both logs are retained.

Tasks 0/1 are complete. Task 2 has a validated faster Crown implementation but remains incomplete because the mandatory overall confirmation fails. Tasks 3-7 remain unstarted, public dispatch stays API3, and the PR stays draft. No targets or tolerances changed.

## Concrete next candidate for owner decision

The current Crown path passes prepared wind terms to the shared Surface driver. The normal `SurfacePlan::run_available_cell` path still passes `None`. Uniform Surface uses static model 10, so the same immutable physical cache is eligible. A narrow follow-up would connect that existing context at this call site, without new equations or cache machinery:

```diff
@@ SurfacePlan::run_available_cell
             tweaks,
-            None,
+            fuel.wind_terms().as_ref(),
         )
```

This proposed change has not been applied or timed. Dynamic fuel remains uncached; supplied fuelbeds remain dependency-checked. If authorized, validate scalar composition, overrides, stale-key/NaN behavior and allocations, then freeze one new candidate and run the entire unchanged screen/confirmation matrix. Do not use the current failed confirmation as a pass, discard its slow samples, or claim a Surface improvement without measurement.

The approved plan requires owner authorization for another candidate after a miss. The bounded Crown pass stops here. An explicit waiver of only the Surface core-headroom prerequisite is another owner decision; it would not waive any public speed target and could expose insufficient boundary margin.
