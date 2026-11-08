# API4 public serial checkpoint: 55/56 bounds, STOP

The complete three-pair public screen passed 55 of 56 first/warmed confidence bounds. Every 100k-cell endpoint passed the unchanged 2x requirement. The only miss was heterogeneous Surface at 10k cells, warmed: its median-ratio estimate was 2.234x, but its lower 95% bound was 0.954x against a 1.25x target. The required seven-new-pair confirmation was not launched. Task 3 remains incomplete; Tasks 4-7 have not started.

## Candidate and method

- Scientific/API4 checkpoint: `df8fb4020e1c583021f270da5c4249bb910aac64`, draft PR23.
- Source SHA256: `7ea2e6cd1ef060f1d94c492a344ceac00db6eb2f85b89f58b99da0a98341c11b`.
- Extension SHA256: `4e201d77817a6f74508538e98f140d421a78d30e7cac2397fbee590a543eb1d1`.
- Harness SHA256: `3931f24fce9e99064f35299d4e39c8b9151d58e3b7391d326178e4068832bb0c`.
- Configured CPython 3.12.8, NumPy 2.1.3, Windows 10 x86-64; compiler and build/thread environment are retained in the raw report.
- Three fresh paired processes, four warmed calls per process, original-input public Python timing, explicit serial Rust. Bootstrap units are paired fresh processes, not individual warmed calls.
- Separate correctness workers check all outputs on widened Python inputs plus the reviewed NaN/selection rules. Timing workers retain the original dtypes and masks. No profiling runs are acceptance evidence.
- All 28 preliminary diagnostics passed: seven continuous 1,031-cell cases and seven routes at 1/16/1,024 cells. Every common model used one native compute boundary with no direction calibration.
- All 28 acceptance cases passed complete-output validation. No samples were discarded, pooled with earlier campaigns, or repeated to obtain a pass.
- Targets remain 1.25x for common routes at 10k, 1x for typed routes at 10k, and 2x for all routes at 100k. The additional C layout uses the typed small-input category; original Surface/Crown/A/B targets are unchanged.

## Complete screen

At 100k cells, median first/warmed speedups span 2.558x to 3.631x; their lower bounds span 2.234x to 3.631x. These are screen observations, not completed seven-pair public certification.

All values below are lower 95% bounds of Python time divided by Rust time. Greater than 1 means Rust is faster.

| Dataset | Route |10k first|10k warmed|100k first|100k warmed|
|---|---|---:|---:|---:|---:|
| uniform | surface | 2.364 | 2.845 | 3.631 | 3.452 |
| uniform | crown | 1.516 | 2.032 | 2.949 | 2.846 |
| uniform | typed-crown | 2.087 | 2.526 | 2.718 | 2.713 |
| uniform | typed-crown-b | 1.615 | 1.912 | 3.248 | 3.156 |
| uniform | masked-crown | 1.991 | 2.079 | 2.703 | 2.759 |
| uniform | masked-false-crown | 1.697 | 1.816 | 2.692 | 2.766 |
| uniform | typed-crown-c | 1.721 | 2.184 | 2.615 | 2.812 |
| heterogeneous | surface | 1.764 | **0.954 FAIL** | 2.907 | 3.081 |
| heterogeneous | crown | 1.836 | 2.321 | 2.234 | 2.386 |
| heterogeneous | typed-crown | 1.517 | 2.097 | 2.755 | 2.691 |
| heterogeneous | typed-crown-b | 1.963 | 2.164 | 2.667 | 2.639 |
| heterogeneous | masked-crown | 2.016 | 2.053 | 2.557 | 2.656 |
| heterogeneous | masked-false-crown | 2.320 | 2.161 | 2.594 | 2.590 |
| heterogeneous | typed-crown-c | 2.213 | 2.283 | 2.583 | 2.717 |

## The failed endpoint

The slow first Rust process stays in the assessment. The other two Rust processes were faster, but the paired-process bootstrap correctly retains the observed variation. Its cause has not been established; it must not be labeled or discarded as external noise.

| Block | Python first ms | Rust first ms | Python warmed median ms | Rust warmed median ms | Paired warmed ratio |
|---:|---:|---:|---:|---:|---:|
|0|42.370|24.018|40.220|42.173|0.954|
|1|46.268|18.639|36.755|15.795|2.327|
|2|42.386|15.868|34.927|16.452|2.123|

The campaign point estimate is the ratio of backend medians across process blocks. It differs from the median of the three paired ratios. The unmodified raw report retains every first and warmed duration.

## One permitted diagnostic profile

After the failed screen, exactly one profiled Rust process ran the same heterogeneous Surface at 10k recipe with four warmed calls. It is diagnostic only and is not appended to the acceptance samples. Its source and extension identities match the screen.

| Measured component | Warmed median ms |
|---|---:|
|Public call wall time|16.866|
|Python work outside native boundary|4.207|
|Native owned copies/setup before detach|1.861|
|Detached wall time, including setup/storage|10.607|
|Actual scientific kernel scopes|7.375|
|Actual final encoder scopes|0.772|
|NumPy final-buffer transfer|0.025|
|Concatenation / appended bytes|0 / 0|

Component medians are reported independently and need not sum exactly. Per-cell profiling adds observation cost, so these timings do not establish an uninstrumented speed ratio. Encoding is now a small part of this call; the measurements point toward wrapper/setup cost and short-call variance for a bounded follow-up. They do not explain the slow acceptance process by themselves.

## Verification and next decision

- Exact candidate full suite: 6,298 passed / 0 skipped, plus 27 subtests, 326.92 seconds with `pytest --require-rust -rA -W error`.
- Stable/MSRV core suites, Rust 1.78 all-feature binding check, all-target/all-feature Clippy, and 36 real production-module/allocation-scope tests pass.
- Eight isolated Windows native/pure wheel choice/install/scientific smokes pass with NumPy 2.1.3 and 2.5.4. Other platforms and final end-to-end allocation/memory/four-thread/million-cell certification remain pending.
- Historical Python fixtures remain unchanged. No numerical test was skipped to obtain these results.

The approved Task 3 rule is: "If missed, profile core versus encoder once and return evidence for an owner decision; no open-ended tuning." That profile is complete. No further acceptance samples, confirmation, optimization candidate, or Tasks 4-7 work are authorized by this failed checkpoint.

The proposed next authorization is one bounded follow-up on model-call ownership/normalization/registry setup and short-call variance, with at most one general implementation candidate, unchanged targets, full regression verification, one fresh complete screen, and seven NEW confirmation pairs only if that screen passes. Another miss stops the work again. No target waiver is assumed.

## Evidence

Subsequent CI integration fixes guard Rust-only test imports on pure-Python
hosts and move the model allocation observer into a separate adapter module.
The standalone counter remains dependency-free. These changes do not alter
scientific equations or normalization; they have not been performance timed.
The screen and profile above remain frozen at the recorded `df8fb40` candidate.

- [Complete immutable screen](2026-11-07-api4-public-serial-screen.json)
- [Single diagnostic profile](2026-11-07-api4-public-serial-profile.json)
- Per-process stdout/stderr, installation reports, test logs, and the ledger remain in the ignored local execution workspace.
