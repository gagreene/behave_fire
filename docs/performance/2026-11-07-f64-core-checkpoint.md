# Float64 core headroom checkpoint: 2026-11-07

**STOP before Task 3.** The first three-pair core screen missed 3 of 32 required first/warm confidence bounds. No confirmation was launched. The public API remains version 3; this is not a public Rust backend speed certification.

The new configured `run_model` calculates physical quantities as float64, propagates missing inputs by field dependency, and writes all requested fields directly into final buffers. Classification stays int32. This core path removes dtype and MaskedArray emulation; the old Python-facing path is unchanged pending the gate.

## Method and identity

- Windows 10 x86-64, CPython 3.12.8, NumPy 2.1.3, 16 logical CPUs, rustc 1.99.0; default release optimization, no allocation-probe instrumentation.
- Three fresh paired process blocks and four warmed calls per process, per case. Both uniform and heterogeneous inputs, all four original routes, 10,000 and 100,000 cells: 96 timing observations/processes and 32 confidence endpoints.
- Python timings use the original public Python inputs, including the original mixed storage layouts. Native inputs are widened and masks filled with NaN before native timing; native timing includes actual `run_model` and final output allocation/storage. Construction, normalization, shape checks, correctness, and export are outside native timing.
- Every native process exports every field and passes independent complete Python f64/NoData validation after timing. Four continuous-layout diagnostic cases also pass; their timing is excluded. Final typed codes, NaN locations, infinities, and signed zeros are checked.
- Ratios are Python time divided by Rust core time: greater than 1 means the core is faster. The core gate requires 25% headroom above unchanged public serial targets: 1.5625x common/1.25x mixed at 10k and 2.5x all routes at 100k. Both first/warm lower 95% bounds must pass.
- Source SHA256: `61884757aa89e00d5df82dffdf6efe9bd6b4a88034c8ae962cfea0c4ca7703ce`.
- Probe binary SHA256: `b0f7b64deb4d61de96d7c7bc55294f10b7afc377960cb2f35df50132279d59be`.
- Native extension SHA256 (public Python reference provenance): `33a822f001f876823f90a9a57c1a65845592b0b83a15785e3c135e02b6c7fa41`.
- Raw report SHA256 (UTF-8, canonical LF line endings): `a4ea1e9a2e506bfa558a9a7ba148fa9e35632a9fbffcef6719370e36c07092a3`.
- All per-process timings, compiler/environment fields, identities, and field checksums are in [the frozen raw screen](2026-11-07-f64-core-screen.json). Raw worker logs and typed exports remain in the ignored plan workspace at the repo-relative paths recorded there.

## Failed endpoints

| Dataset / route / cells | Phase | Median ratio | Lower 95% | Required |
| --- | --- | ---: | ---: | ---: |
| uniform / crown / 100,000 | first | 2.875x | 1.986x | 2.500x |
| uniform / crown / 100,000 | warm | 2.884x | 2.409x | 2.500x |
| heterogeneous / typed-crown / 100,000 | first | 2.766x | 2.034x | 2.500x |

All 10k endpoints pass the cheap core screen. All 16 warmed endpoints have median ratios above their thresholds, but uniform Crown at 100k fails its warmed confidence bound. Heterogeneous common Crown at 100k passes this core screen; that does not certify the public backend or erase its earlier public confidence-bound failure.

## Costs and variance

| Case / backend | First-call seconds, paired blocks 0 / 1 / 2 | Warmed process medians, blocks 0 / 1 / 2 |
| --- | --- | --- |
| uniform crown / python | 0.5211 / 0.7235 / 0.6509 | 0.5710 / 0.6480 / 0.6679 |
| uniform crown / core-rust | 0.2624 / 0.2264 / 0.2240 | 0.2370 / 0.2030 / 0.2247 |
| heterogeneous typed-crown / python | 1.1898 / 0.6776 / 0.6673 | 0.8028 / 0.6245 / 0.9715 |
| heterogeneous typed-crown / core-rust | 0.2194 / 0.2450 / 0.3281 | 0.2290 / 0.2367 / 0.2751 |

The medians are promising, but observed variance and the remaining kernel cost leave insufficient measured headroom at the agreed confidence level. These observations do not establish the cause of variance or prove that boundary costs will fit within the public target. No extra sampling or numerical tuning was performed to turn a miss into a pass. This screen measures the frozen pre-review source identity. Subsequent review fixes reject missing selectors before index conversion and preserve missing fuelbed/reaction prerequisites, including cached Surface/Crown preparation; those fixes are untimed and do not waive the stop.

## Verification and next decision

The rebuilt-extension full suite passed 6,144 tests with zero skips and 27 subtests. A later added no-pooling guard and the paired-driver/checker tests pass separately (7 tests). Full stable/MSRV core tests, 36 real native-module tests on stable and Rust 1.78, Clippy, rustdoc, and the standalone tile example pass. The actual configured kernel and encoder loops each report zero absolute allocation requests and zero requested bytes on explicit-invalid/sparse-NaN 10k data with four workers.

Tasks 0 and 1 are complete. Task 2 has a validated implementation and a failed mandatory performance gate; it is not complete. Tasks 3-7 have not started. The new API4 boundary, complex auto routing/warnings, staged sessions, broad helper migration, final wheel validation, and final performance acceptance are still pending.

Owner choices:

1. Authorize one bounded investigation of the current f64 Crown kernel cost and variance, retaining the existing targets and requiring a newly identified candidate before any new campaign.
2. Explicitly waive the 25% core-headroom prerequisite and authorize only Task 3's limited model migration/public serial checkpoint. This does not waive or lower any public performance target and carries a risk that boundary costs exhaust the remaining margin.
3. Keep the implementation as a draft checkpoint while retaining current public behavior.
