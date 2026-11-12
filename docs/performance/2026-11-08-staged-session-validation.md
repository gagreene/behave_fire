# Staged raster sessions: validation and memory

Task 4 adds an owned Rust `GridSession`, a checked borrowed `FieldView`, and the
Python `SurfaceGrid` and `CrownGrid` APIs. Physical calculations reuse scalar
equations from the full model. The stateless model does not construct a session.

Sessions capture the actual inputs, selected wind options, registry or supplied
Surface fuelbed, validity flags, and persistent overrides. A transient injection
is a pinned source, including NaN. Automatic completion preserves it and can
recover released ancestors. Explicit producer runs replace all transient values
they produce; persistent overrides survive. A fused restart clears transient
values and uses the configured fused kernel when the output/override request
permits it. Requests for other intermediates or arbitrary persistent fields use
a fresh field DAG. Python getters copy retained data; reading an unavailable
field raises instead of triggering computation. New sessions cannot be pickled.

The Python constructors accept an input mapping with base-unit values and an
optional common shape. Scalars broadcast independently; unlike the legacy
facade, moisture columns need not have the identifier array's original shape.
Rust-only grid classes reject complex/object input and require a usable extension.
Examples are `src/examples/staged_raster.py` and the crate's `staged_raster` example.

## Correctness and ownership

The full required-native default suite passed **6,347 tests, zero skipped, plus
27 subtests** with warnings treated as errors (Python 3.12.8, NumPy 2.1.3,
Windows x86-64). The 42 new Python cases cover scalar, empty, and common grids;
one/four workers; staged/fused equivalence; individual missing inputs and selected
Crown branches; NaN injection and independent siblings; registry/fuelbed ownership;
typed edits; returned-buffer mutation; explicit producer/reset behavior; automatic
namespace access; and pickle rejection. Rust integration tests additionally cover
freed-ancestor recovery with multiple injections, exact worker execution,
persistent overrides, partial output retention, and configured full-run parity.

The standalone core passed Rust 1.78 tests. Strict workspace/all-target Clippy and
rustdoc passed. An archive generated with Rust 1.78 passed the standalone package
checker: **1,461,959 compressed bytes, 158 files, zero normal/build dependencies**.
Extracted-package consumers passed with stable Rust and Rust 1.78, with Python
deliberately unavailable to the build. Those consumers exercised embedded data,
custom fuels, scalar models, raster workers, reaction-intensity injection, and
Crown-fraction injection.

The production allocation observer measured **zero allocation requests and zero
requested bytes in both scientific and output-write scopes** for staged Crown
execution, with one and four workers. Explicitly invalid cells skipped scientific
callbacks and retained initialized unavailable outputs. The complete diagnostic
probe suite passed 37 tests. Buffer and worker setup allocations remain outside
these per-cell scopes.

## Separate staged-memory diagnostic

These are descriptive observations from eight fresh processes at 100,000 cells,
with a varying moisture column and otherwise scalar inputs. Staged execution ran
every explicit stage, then automatic completion. Fused execution used the same
owned session inputs. Each row records the process's lifetime peak working set,
including imports, session construction, and calculation. No timings were taken.

| Model | Execution | One worker (MiB) | Four workers (MiB) |
|---|---|---:|---:|
| Surface | Fused restart | 56.64 | 57.07 |
| Surface | Every explicit stage, then completion | 104.87 | 105.21 |
| Crown | Fused restart | 62.05 | 62.54 |
| Crown | Every explicit stage, then completion | 122.39 | 122.75 |

Explicit stages retain intermediate columns until automatic completion applies
liveness, so their peak storage exceeds the fused kernel's working set. The raw
record includes preconstruction, owned-input, and postcompletion working sets,
environment versions, and the extension hash in
[`2026-11-08-staged-session-memory.json`](2026-11-08-staged-session-memory.json).
These observations do not certify final memory or speed targets. The broader
performance, artifact, and platform acceptance gates remain separate.
