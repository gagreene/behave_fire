# Public serial checkpoint: confirmed GO

The single owner-approved plain-normalization candidate passed all 56 first/warmed confidence bounds in a fresh three-pair screen and again in the required seven-NEW-pair confirmation. The source, extension, harness, host, Python, and NumPy identities stayed frozen. No observations were pooled with the screen or earlier failed campaigns. This clears Task 3 public serial feasibility; it does not certify four-thread, million-cell, memory, or final release-platform targets.

## Candidate and scope

- Candidate commit: `241536621c842c6bc6680168b1dd2cf065aa967b`.
- Source SHA256: `d5a56991f0756e05b1c21bf2c4206259effe8b8e95532ff7b831a0017b5437c5`.
- Extension SHA256: `14d6c80426f863f82de648edbb799163dcd4e7b38e2a4715127e8ace1647dff5`.
- Harness SHA256: `3931f24fce9e99064f35299d4e39c8b9151d58e3b7391d326178e4068832bb0c`.
- Configured CPython 3.12.8 / NumPy 2.1.3 / Windows 10 x86-64; full compiler/build/thread identity is retained in the reports.
- Two datasets, seven layouts, and 10k/100k cells. First and warmed lower 95% bounds use fresh paired processes as bootstrap units.
- Original-input public Python is the timing baseline. Separate correctness workers use widened Python equations plus reviewed NaN rules and validate every result field.
- Both phases passed all 28 preliminary continuous/scalar diagnostics and all 28 full-output proofs. Common models used exactly one native compute boundary and no direction calibration.

## Confirmed results

At 100k cells, median first/warmed speedups range from 2.930x to 4.085x. The corresponding lower 95% bounds range from 2.455x to 3.810x, all above the unchanged 2x target.

The previously failing heterogeneous Surface 10k endpoint now passes in both phases. Confirmation values:

| Endpoint | Median ratio | Lower 95% | Target |
|---|---:|---:|---:|
|first|2.268|2.214|1.25|
|warm|2.646|2.229|1.25|

Every table value below is a lower 95% bound of original Python time divided by Rust time. Greater than 1 means Rust is faster.

| Dataset | Route | 10k first | 10k warmed | 100k first | 100k warmed |
|---|---|---:|---:|---:|---:|
| uniform | surface | 2.167 | 2.732 | 3.476 | 3.810 |
| uniform | crown | 2.279 | 2.294 | 2.985 | 3.023 |
| uniform | typed-crown | 2.420 | 2.596 | 3.275 | 3.251 |
| uniform | typed-crown-b | 2.398 | 2.742 | 3.064 | 3.168 |
| uniform | masked-crown | 2.185 | 2.626 | 2.541 | 2.970 |
| uniform | masked-false-crown | 2.365 | 2.624 | 2.609 | 2.897 |
| uniform | typed-crown-c | 2.192 | 2.502 | 2.980 | 3.087 |
| heterogeneous | surface | 2.214 | 2.229 | 3.117 | 3.025 |
| heterogeneous | crown | 2.226 | 2.260 | 3.009 | 2.937 |
| heterogeneous | typed-crown | 2.360 | 2.534 | 2.455 | 2.895 |
| heterogeneous | typed-crown-b | 2.177 | 2.600 | 3.119 | 2.952 |
| heterogeneous | masked-crown | 2.246 | 2.469 | 2.911 | 2.582 |
| heterogeneous | masked-false-crown | 2.314 | 2.580 | 2.868 | 2.825 |
| heterogeneous | typed-crown-c | 1.993 | 2.195 | 2.754 | 2.862 |

## Interpretation and verification

The candidate removes unnecessary MaskedArray intermediates and repeated copies only for trusted plain physical storage. Caller independence, masked/subclass paths, exact controls, and numerical rules remain covered by tests. The earlier slow process was not reproduced in the finite variance diagnostic; neither GC nor the sampler was established as its cause. The new passing campaign does not erase or explain the old failure by itself.

- Exact candidate full suite: 6,305 passed / 0 skipped, plus 27 subtests; 259.61 seconds with `pytest --require-rust -rA -W error`.
- Eight isolated Windows native/pure wheel choice/install/scientific smoke combinations passed with NumPy 2.1.3 and 2.5.4.
- All 52 CI checks passed on the candidate. Rust scientific sources were unchanged by this boundary optimization.
- No second candidate, target relaxation, unchanged-candidate retry, or exclusion of an unfavorable process was used.

Proceed with the approved staged-grid and remaining helper work. Its final implementation still requires the planned large/four-thread/memory/allocation and artifact acceptance checks; measurements on this host are not a speed guarantee for other hardware or workloads.

## Evidence

- [Fresh screen](2026-11-08-plain-normalization-public-screen.json)
- [Seven-new-pair confirmation](2026-11-08-plain-normalization-public-confirmation.json)
- [Finite diagnostic and single candidate](2026-11-08-plain-normalization-diagnostic.md)
- [Earlier failed checkpoint, retained](2026-11-07-api4-public-serial-checkpoint.md)
