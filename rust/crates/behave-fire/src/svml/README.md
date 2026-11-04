# NumPy SVML float32 atan2

`atan2f.s` comes from NumPy/SVML revision
`bd9d46e0c627ab16f3d2641f7cdb1bc15d303ba4`, file
`linux/avx512/svml_z0_atan2_s_la.s`, as used by NumPy 2.5.4.
Upstream SHA256:
`d61191ed59d36fa905d91c0d0bd89a49f4feab29eb2e39db031b083c5c9d7823`.

NumPy 2.1.3 pins SVML revision `32bf2a98420762a63ab418aaa0a7d6e17eb9627a`.
Its corresponding file has SHA256
`b5d5a1831168f3342ebc69202693c3dd2750f9850721e96cda7c8e180380a3fa`.
The two upstream versions have identical mathematical instructions and tables.
The newer version adds only a `.file` directive, an `endbr64` instruction, and
a GNU property note declaring IBT/SHSTK.

The Intel copyright/SPDX header remains intact. The repository license and Intel
notice are reproduced in the crate's `LICENSES/Intel-SVML-BSD-3-Clause.txt`.

Local changes are limited to assembler linkage and annotations:

- The vector entry, rare-case helper, tables, and labels are local to the object
  and have an atan2-specific prefix. No NumPy symbols are exported or resolved at
  runtime.
- The GNU IBT/SHSTK property note is removed so this object does not impose those
  process-wide CPU properties. The mathematical instructions and tables are
  unchanged. The non-executable GNU stack note is retained.
- The scalar SysV wrapper lives in `../svml.rs`, with a hidden symbol containing
  the Cargo package version. This separates different core package versions.

The wrapper broadcasts initialized scalar arguments into all sixteen lanes and
tail-jumps to the original kernel. The original prologue restores its stack and
callee-saved registers on every normal/rare return path. The sole assembly call
targets the self-contained rare-case helper; there are no external math/runtime
calls. Rust uses the low result lane as its scalar return value.

The private adapter is the only module allowed to use unsafe code. It compiles
only on Linux x86_64 and checks AVX2, FMA, AVX512F, CD, BW, DQ, and VL before
entering assembly. Rust's feature detection also checks operating-system vector
state support. Unsupported explicit requests return a structured error through
the masked engine. The Python binding identifies the linked atan2 kernel with
a bounded calibration over sixteen fixed finite constants and exact native
candidate payloads. CPU target names alone cannot identify SVML linkage.
Configuration is cached under a lock, unknown or ambiguous implementations fail
explicitly, and no user data enters NumPy through this mechanism. Exponential
selection remains independent; ordinary libm calculations remain unchanged.

Validation must include Linux execution on supported and unsupported CPUs,
NumPy 2.1.3 and current NumPy, special values, signed zero, concurrent calls, and
stable/MSRV object compilation. Windows compilation alone cannot validate the
assembly, which is excluded there by target configuration.
