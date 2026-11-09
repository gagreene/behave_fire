# Whole-stack review and corrections (2026-11-09)

One independent read-only review covered `693916c..d7ec497` against the approved
float64 design. The review found no Critical issues, five Important issues, and
one warning-guidance omission. The same reviewer confirmed the focused fix round;
all findings are closed. Final corrected-source artifact and timing verification
remain separate from this code-review verdict.

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| Public scalar/helpers retained false-default NaN policies | Public plans use the existing availability-aware equations; required unknown thresholds propagate, while known-false/unselected branches remain available | Public helper and standalone Rust NaN tests; cached/ordinary wind equivalence; original finite corpus |
| Retained raster methods returned zeros for invalid cells | Cell, full-raster, and tile routes return NaN physical outputs and classification -1 | Direct Rust invalid-cell tests and unchanged valid-cell comparisons |
| Branch-based unit selectors erased NaN/complex values | Validate original real/finite/unmasked controls before interpreting accepted finite selector conventions | Missing/nonreal tests for all branch-based families; original selector convention suite |
| Pinned session values bypassed validity | Derived views mask invalid cells while retaining raw pins for later validity edits | Persistent/transient, fused/staged, invalid-to-valid restoration cases |
| Classification flag injections had no final consumers | Final dependencies read Surface, passive, and active flags with that priority; all false means unavailable | Persistent/transient final-selection and dependency graph tests |
| Fallback warning lacked remediation | Warning includes the existing installation/rebuild guidance | Backend warning and installed-artifact suites |

The focused confirmation caught a guard-order regression during the fix round:
missing wind coefficients were checked before a known-false `phi`. New tests
first reproduced the failure, then verified zero for `phi <= 0` and NaN when the
actual required conjunction remains unknown. Torch spotting similarly preserves
known zero wind without requiring missing flame/cover inputs. Sheltered WAF does
not require an unused missing fuel depth.

Original Python implementation files, scientific JSON fixtures, and finite
tolerances remain unchanged. Tests explicitly pinning the superseded API3 scalar
or zero-filled invalid-cell behavior now assert the approved missing-data rules.
Independent Python references apply documented conditional availability rules
after evaluating the original widened equations.

The original frozen performance campaign and arrays remain immutable evidence
for commit `fe43dce`. Corrections change the core binary, so final-source timing
and allocation verification use fresh artifacts and observations. Previous source
identities are never relabeled as measurements of the corrected implementation.

The [reviewed-source acceptance](2026-11-09-reviewed-float64-acceptance.md)
now passes all 40 cases and 80 first/warmed confidence bounds on the corrected
scientific source at `9be8a50`. Its matched allocation builds, complete output
arrays, continuous and additional-layout checks, and attribution profiles are
retained separately from the historical campaign. The measured production and
benchmark-source inventory remains unchanged in the evidence-only follow-up.

The reviewer accepted the existing source/array hashes and narrow post-measurement
metadata attestation. Historical dtype parity, unreleased API preservation,
explicitly unsupported grid pickle/complex fallback, and non-Windows speed were
set aside according to the owner-approved scope. No behaviors were set aside in
the focused confirmation.
