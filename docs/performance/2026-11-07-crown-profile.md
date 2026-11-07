# Bounded Crown profiling and variance investigation

## Result

One isolated immutable wind-term cache reduced warmed Crown core time by about 24%, or 1.31-1.32x relative to the current Rust core. This is a candidate/control diagnostic comparison, not a Python/Rust speed ratio or a passed performance gate. The existing failed screen is unchanged.

## Scope and evidence

- Base: `2d88105a5a5efd4d1f9c5810ae7f074c630d6e1c`; targets unchanged. No production scientific source was changed during diagnosis.
- Two sampled stage-profile processes (uniform Crown and heterogeneous mixed Crown A, 100k cells), followed by six fresh Python/control pairs per case: three default-affinity blocks, then three pinned to the lowest allowed logical CPU. Four warmed calls per process. This was fixed before observing results.
- One cache candidate only, compared in three new control/candidate pairs per case with four warmed calls, default affinity. No samples were pooled with the failed screen, and no acceptance checker was run on these diagnostic records.
- Every native process stores all outputs and passes independent Python float64/NoData validation after timing. The prototype also passes all 12 recipe/layout correctness cases and the full Rust 1.78 core suite, including scalar composition and missing-data tests. Native finite comparisons are bit-exact; Python f64 checks keep the established tolerance.
- CPU time comes from Windows process accounting; elapsed time surrounds only the actual native run including final storage. Windows accounting can be coarse relative to short calls; CPU/elapsed ratios slightly above one reflect quantization. Host utilization covers the entire child process, including setup/validation, so it is context rather than a per-call causal measurement.
- [All raw diagnostics, calibration, summaries, identities, and snapshot hashes](2026-11-07-crown-profile.json). Local worker logs and complete typed exports remain at the repo-relative paths in the reports. No raw evidence was discarded.

## Actual execution costs

Stages were sampled every 64th cell inside the actual configured model. Times below are warmed sampled nanoseconds per cell, inclusive of nested instrumentation. Empty individual scope overhead measured about 66-74ns. Nested scopes add overhead to their parents; these are diagnostic costs, not exact percentages of uninstrumented runtime. Tiny cache-check values near that floor do not show a dominant cache-key bottleneck.

| Sampled group | Uniform Crown ns/cell | Mixed Crown A ns/cell |
| --- | ---: | ---: |
| kernel | 3891 | 3592 |
| encoder | 134 | 108 |
| crown_spread | 2154 | 2005 |
| fuelbed | 431 | 394 |
| reaction | 222 | 196 |
| surface_forward | 987 | 943 |
| backsolve | 816 | 765 |
| wind_coefficients | 418 | 387 |
| wind_vectors | 539 | 521 |
| fuel_cache_checks | 135 | 124 |
| savr_cache_checks | 86 | 82 |

Forward Surface work and the crowning backsolve dominate. The wind coefficient group executes twice per Crown cell. For static physical fuel, both calls recompute two sigma powers and two exponentials; two wind-amplification calls and one effective-wind calculation also repeat packing powers. These are independent of per-cell weather and moisture for static fuel. Encoding is comparatively small.

## Variance

Pinning one logical CPU did not remove variance. The worst pinned native first call took 288.5ms elapsed and 250.0ms CPU while whole-process host utilization was 64.4%. The elapsed excess is real, but time spent executing on CPU also increased; the data do not establish the exact frequency, cache, memory, or scheduling cause. Affinity alone is not a sound remedy.

Both original-input Python and native calls varied. Future campaigns should record environment/load and use a declared quiet-host protocol consistently for both backends, never discard slow samples after observing ratios, and keep the original confidence/first-call requirements.

## One general candidate

The prototype stores sigma-derived wind coefficients plus positive and negative packing powers in immutable prepared physical fuel state. Static fuel uses those terms in forward spread and backsolve; dynamic fuel falls back to the existing per-cell calculation. Physical edits invalidate the prepared-fuel key, and actual fuelbed sigma/packing bit checks prevent reuse after injections. Positive and negative powers are computed independently; floating product order and thresholds stay unchanged. No dataset-specific constants, dtype dispatch, global caches, SIMD, or new dependencies are introduced.

| 100k case | Phase | Current Rust ms | Prototype ms | Current/prototype | Time reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform crown | first | 185.60 | 130.15 | 1.426x | 29.9% |
| uniform crown | warm | 178.00 | 134.52 | 1.323x | 24.4% |
| heterogeneous typed-crown | first | 179.70 | 129.76 | 1.385x | 27.8% |
| heterogeneous typed-crown | warm | 172.77 | 131.46 | 1.314x | 23.9% |

First-call gains have wider paired variation than warmed gains. These three pairs support investigating the candidate; they do not certify the seven-pair public target or justify multiplying historical ratios by this speedup.

The [Windows CPU accounting patch](2026-11-07-crown-cpu-instrumentation.patch.json) applies first to a disposable copy; the sampled patch applies on top of that copy. The exact experimental [cache patch](2026-11-07-crown-wind-cache-prototype.patch.json) and [sampled instrumentation patch](2026-11-07-crown-stage-instrumentation.patch.json) are retained as artifacts. Instrumentation uses temporary snapshots only. Each JSON diff artifact preserves verbatim unified-diff lines; reconstruct the patch with `''.join(document['lines'])`. The prototype patch is not the shipped interface: integration keeps wind context private and shares the existing wind products instead of duplicating formulas or adding cache transport to public result records.

## Integrated candidate gate result

The single candidate was integrated with private wind context, shared ordinary equations, and three stale-key/threshold/missing-value tests. It passed all 32 bounds in a fresh screen. The required seven-new-pair/five-warm confirmation passed 31 of 32 bounds: every Crown/A/B bound passes, but uniform Surface at 100k misses first-call lower confidence (2.241x versus 2.5x; median 3.814x).

The [complete checkpoint, raw results, and concrete Surface follow-up](2026-11-07-wind-cache-core-checkpoint.md) preserve that failure. The current normal Surface route still passes no prepared wind context. A proposed follow-up would connect the same existing cache there; it has not been applied or timed. The bounded pass stops with the PR draft and API4 still gated. No target, tolerance, sample count, or required route was dropped.