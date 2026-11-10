# PR review follow-up: current float64 stack

The reviews cover PR18, PR19, PR21, PR22, PR23, PR24, and PR26. Earlier branches
describe intermediate API3 implementations. This follow-up applies live fixes to
the combined API4 code in PR26, without rewriting predecessor branches or
restoring retired dtype/mask machinery. No review replies or resolutions are
posted automatically.

## Confirmed live fixes

- A Python-preferred auto backend now calls Python directly. An absent or
  unusable extension no longer restricts Python to the Rust complex whitelist or
  copies raster arguments through native preflight. Python exceptions propagate.
- Explicit Rust avoids the unused auto registry snapshot. Auto with usable Rust
  still freezes a registry for verified later migration. Copy/pickle behavior
  remains covered. NumPy complex scalar elapsed times have their own proven area
  fallback; the first complex operation warns once per process across operations.
- Staged fuel IDs accept finite integral floats, with range/missingness checks.
  Classification codes retain integer storage. Grid flags require boolean or
  integer zero/one; component helpers retain their existing real truth convention.
- Supplied fuelbeds reject unknown columns at Python and native boundaries and
  no longer require an unused fuel-ID input. The source is selected before fuel-ID
  validation; no registry lookup is evaluated for the unused structural default.
- Grid shape inference uses the existing tuple-based broadcast planner. Native
  output transfer flattens ranks above rust-numpy's 32-axis ceiling, then the
  Python getter restores logical axes. Empty and nonempty 35-axis Surface/Crown
  sessions are covered.
- `PreparedFuel` equality compares its fourteen public properties, independent
  of private cache history. NaN and signed-zero conventions remain ordinary float
  equality. Cache-helper comments and three corrupted Crown unit comments are fixed.
- The isolated allocation-counter test now includes a real unrelated allocating
  thread. Outside-scope counts use lower bounds; kernel and encoder tuples still
  require exact counts. No allocation-certification threshold is relaxed.

The auto cases reproduced failures before correction; the full focused auto
suite then passed 58 tests. Grid regressions reproduced 12 failures with seven
already-correct cases; the corrected focused suite passed 132 tests. Fuel
equality reproduced a failure, then all three equality cases and the full
572-test standalone core suite passed. The outside counter reproduced seven
requests/218 bytes where the old assertion demanded one request/11 bytes; all
three isolated counter tests pass with exact scientific/encoder checks retained.
See the PR's exact-head test record for final combined validation and CI.

## Complete comment disposition

Numbers below match each review's numbered items.

| Review | Item | Disposition and evidence |
| --- | --- | --- |
| PR18 | 1 | Fixed: public fuel equality excludes caches. Private caches remain private; public construction through `default`, edits, and `prepared` is supported. |
| PR18 | 2 | Cache and named particle/reaction helper comments added. Broad Rust alphabetization is not imposed: the repository's ordering rule explicitly covers top-level Python functions. |
| PR18 | 3 | Retired: the old typed `raster_probe.rs` dumper was deleted with dtype emulation. Current complete-output comparisons verify field inventories. |
| PR18 | 4 | Documented: continuous diagnostics vary fuels/weather and keep canopy inputs constant. No completed dataset or measurement is changed. |
| PR18 | 5 | Already fixed: API4 model/component producers enter actual encoder scopes. The recorded 96 allocation samples include real kernel and encoder observations. |
| PR18 | 6 | Format retained: existing allocation rows are independently portable and embed their proof. Removing duplicated evidence would require a separate report-format change. Recorded reports and digests are immutable; size/repeated validation cost is a nonblocking limitation. |
| PR18 | 7 | Fixed and reproduced: unrelated thread allocations no longer break an exact outside-count assertion; scientific/encoder assertions remain exact. |
| PR19 | 1 | Retired: old `prepared/fields.rs`, `masked.rs`, and scalar metadata interpretation were removed. Primitive API4 columns have no colliding `_scalar.*` metadata. |
| PR19 | 2 | Already corrected in surviving documentation: normal heading separation and final newlines. |
| PR21 | 1 | Already fixed: the profile wrapper's `_array_bytes` handles nested mappings; current API4 transfers have no legacy schema dict in their result columns. Recorded profiles execute successfully. |
| PR21 | 2 | Fixed: Crown comments use correct UTF-8 square/cubic units. |
| PR21 | 3 | Already fixed: conftest requires and reports binding API4. |
| PR21 | 4 | Already fixed: backend docs describe API4 and rebuilt extensions; API2 and API3 are incompatible. Historical measurements retain their original API labels. |
| PR21 | 5 | Old parallel/column writers were retired. Rust-wide mechanical reordering is advisory under a Python-specific rule. Current capability/operation/schema inventory tests guard the 89-operation contract. |
| PR22 | 1 | Retired: the old prepared conversion helper no longer exists. Current API4 buffer/control length checks precede workers. |
| PR22 | 2 | Retired: the old `_recompute_field`/pipeline replay path is gone; outputs use dependency-local API4 groups. |
| PR22 | 3 | Old pipeline module retired; no obsolete helper rearrangement is needed. |
| PR22 | 4 | Old prepared helpers retired; current probe modules follow the surviving primitive producers. No broad Rust ordering rewrite. |
| PR22 | 5 | Retired: no empty scientific prototype/capture remains. Initialized zero-cell outputs and skipped callbacks have direct tests. |
| PR22 | 6 | Compact weak/scalar emulation tests are obsolete after the approved f64 contract. Short nonempty Rust registries deliberately clamp through `PreparedFuels::for_grid`, as documented and confirmed in the later PR23 review; imposing an 11-row minimum would change that accepted API. Empty registries are rejected. |
| PR23 | 1 | Fixed: Python preference bypasses the Rust whitelist and argument snapshots. Actual absent/unusable model and namespace cases retain Python complex behavior. |
| PR23 | 2 | Fixed: ordinary NumPy complex64/complex128/long-double scalar elapsed times pass an independently derived ellipse-area check and retain imaginary output. Custom scalar subclasses remain opaque and unverified for Rust-preferred fallback. |
| PR23 | 3 | Clarified: the complex warning is once per process, at the first complex operation, across operations. It is separate from extension-unavailable warning state. |
| PR23 | 4 | Fixed: only auto with a usable native preference creates the neutral registry snapshot. Forced-Rust metadata copy-hook and pickle cases pass. |
| PR24 | 1 | Fixed: staged fuel identifiers use `normalize_identifier`; classification injection remains strictly integral. |
| PR24 | 2 | Already fixed in the earlier PR26 correction, with persistent/transient invalid-mask and restoration regressions in `test_final_review_contract_array.py`. Those tests still pass. |
| PR24 | 3 | Fixed: unknown supplied-bed fields rejected at both boundaries; omitted fuel IDs allowed with a supplied source. |
| PR24 | 4 | Fixed: grid flags reject integers outside zero/one, before native calculation. |
| PR24 | 5 | Fixed: high-rank inference and native export/getter both support logical axes above 32. |
| PR24 | 6 | Named native helpers now have comments. Rust-wide alphabetical rearrangement is outside the explicit Python ordering rule. |
| PR24 | 7 | Public paths already validate controls in `_contracts.normalize_arguments` before primitive casting. Missing/fractional iteration/lookup controls and missing flags have regressions; no extra duplicate normalization is added to the hot transport. |
| PR24 | 8 | Already guarded: the public particle route normalizes integral float IDs before lookup indexing. A real public-call regression confirms float/integer IDs agree. |
| PR24 | Test gaps | Added Python lifecycle coverage for both clear operations, output/worker changes, UserInput WAF, and RelativeToNorth aspect invalidation. Invalid pinned fields were already covered by the earlier final-review regressions. |
| PR26 | 1 | Fixed: the binding sentence uses an explicit subject. |
| PR26 | 2 | Clarified: the prior function reorder affected only a test-reference module, not either benchmark oracle. Historical attestations retain their exact original scope; see the provenance table below. |
| PR26 | 3 | Fixed: the matched-build sentence uses an explicit subject. |

## Measurement and attestation scope

| Record | Continuing meaning |
| --- | --- |
| `2026-11-09-float64-acceptance.*` and `float64-metadata-attestation.json` | Historical campaign at `fe43dce` and its narrow diagnostic-label change. Superseded as a current-code performance result; immutable raw observations remain valid historical evidence. |
| `2026-11-09-float64-review.md` | Closed earlier code-review findings and their corrections. This follow-up adds later findings; it does not invalidate the fixes already reviewed. |
| `2026-11-09-reviewed-float64-acceptance.*` | Dated corrected scientific-source measurement at `9be8a50`: all 40 cases/80 bounds passed on the declared Windows host. |
| `2026-11-09-reviewed-float64-example-attestation.json` | Exact one-line example assertion change after that measurement. It never authorizes arbitrary later library or boundary changes. |
| This follow-up | Later input-domain, structural-validation, high-rank transfer, public-equality and test/documentation fixes. Historical timing data is not relabeled as a measurement of this newer source. |

The numerical equations, original Python production science, finite tolerances,
benchmark workloads and timing harness are unchanged. No full new timing
campaign is claimed for the boundary follow-up. The previously reported
3.18–5.52x serial and 6.06–13.44x four-worker ranges remain measurements of the
explicit source/host named in the dated report. Raw JSON attestations and their
manifest digests are preserved rather than edited to add status fields.
