# Surface/Crown core headroom: confirmed GO

All 32 required core confidence bounds pass in both the fresh screen and the independent confirmation. The Surface route now consumes the same immutable wind context already used by Crown. No equations, constants, targets, tolerances, required cases, or sampling rules changed.

## Frozen candidate and scope

- Measured commit: `b3759d99e27c58087186c37a131760f4be0f11de`.
- Source SHA256: `672bd0fdc34a3344d10610553b567a1a7f3f69803c32c4ff594461955d031255`.
- Release probe SHA256: `fd134259641bf1957a6fbe12efe770d4b4214c6c72e8b8f10ca5f87b90f21585`.
- CPython 3.12.8, NumPy 2.1.3, Windows x86-64, same compiler/build/environment for screen and confirmation. Screen: three fresh pairs and four warm calls. Confirmation: seven NEW pairs and five warm calls. No earlier observations were pooled or discarded.
- Both phases cover uniform/heterogeneous Surface, Crown, mixed Crown A/B at 10k and 100k cells. Every native process stores all outputs and passes independent complete Python f64/NoData validation after timing. Four continuous-layout correctness diagnostics per phase also pass.
- Python timing uses the original raw inputs. Native setup, normalization, shape validation, correctness, and export are outside timing; actual run_model and complete final output storage are inside.
- This is core-only headroom proof. It permits the planned limited API4 model migration; it does not certify public backend speed, one-million-cell throughput, four-thread targets, memory, or wheel platforms.
- [Fresh screen](2026-11-07-surface-cache-core-screen.json), [independent confirmation](2026-11-07-surface-cache-core-confirmation.json). All per-process timings, identities, field checksums, and validation records are retained. Raw typed exports/logs remain at the ignored repo-relative paths recorded there.

## 100k confirmation

Ratios are public Python time divided by Rust core time: greater than one means the core is faster. Both first/warm lower 95% bounds must reach 2.5x, retaining 25% headroom above the unchanged 2x public serial target.

| Dataset | Route | Phase | Median ratio | Lower 95% | Required |
| --- | --- | --- | ---: | ---: | ---: |
| uniform | surface | first | 5.213x | 4.971x | 2.500x |
| uniform | surface | warm | 4.593x | 4.471x | 2.500x |
| uniform | crown | first | 3.943x | 3.808x | 2.500x |
| uniform | crown | warm | 4.028x | 3.958x | 2.500x |
| uniform | typed-crown | first | 4.277x | 4.180x | 2.500x |
| uniform | typed-crown | warm | 4.235x | 4.153x | 2.500x |
| uniform | typed-crown-b | first | 4.195x | 4.100x | 2.500x |
| uniform | typed-crown-b | warm | 4.237x | 4.117x | 2.500x |
| heterogeneous | surface | first | 3.754x | 3.526x | 2.500x |
| heterogeneous | surface | warm | 4.107x | 3.966x | 2.500x |
| heterogeneous | crown | first | 4.149x | 3.803x | 2.500x |
| heterogeneous | crown | warm | 4.073x | 3.879x | 2.500x |
| heterogeneous | typed-crown | first | 4.465x | 4.084x | 2.500x |
| heterogeneous | typed-crown | warm | 4.379x | 4.174x | 2.500x |
| heterogeneous | typed-crown-b | first | 4.416x | 3.801x | 2.500x |
| heterogeneous | typed-crown-b | warm | 4.185x | 4.007x | 2.500x |

All 10k bounds also pass their unchanged route-specific requirements: common routes 1.5625x core headroom, mixed routes 1.25x. Heterogeneous common Crown at 100k passes both confirmation bounds; its earlier public confidence failure is retained as historical evidence and is not erased by this core result.

## Verification and continuation

The measured candidate passed the full local suite: 6,149 tests, zero skips, plus 27 subtests in 194.66 seconds. Full stable/MSRV core tests, 36 real native production-module tests, Clippy, rustdoc, standalone tiles, and 12 independent recipe/layout output comparisons pass. Expanded Surface regression coverage compares every output with uncached scalar composition for static models 1/4/10 and dynamic models 101/124, both wind orientations/heights, persistent overrides, and four workers. Missing depth/extinction tests now cover static model 10 on Surface as well as Crown. Actual configured kernel/encoder allocation scopes remain zero.

Only the prepared context argument changed in the production Surface route. Actual sigma/packing key checks still prevent stale reuse; dynamic fuel remains uncached and supplied fuelbeds retain their existing path. Earlier failed screens/confirmations remain preserved in the prior checkpoint files.

Task 2 can complete after its final verification record is written. Task 3 continues under the original implementation authorization: limited model API4 transport, explicit numeric migration manifest, complex preflight routing/fallback warnings, ownership/shape tests, and a separate mandatory public serial screen plus independent confirmation. Python public dispatch stays API3 until that coherent switch is implemented. The PR remains draft; no merges or publication are authorized.
