# Backend compatibility contract

## Float64 Rust calculations

Rust uses float64 for physical calculations across models, component helpers,
unit conversions, and scalar state tools. Real boolean, integer, float16,
float32, and float64 inputs normalize to float64 before scientific arithmetic.
Masked physical values become NaN. Physical results are ordinary float64 arrays;
Rust does not preserve MaskedArray identity, fill values, raw masked payloads,
boxed scalar types, fixed-width integer overflow, or weak-scalar promotion.

These differences also apply under `backend="auto"`: installations with usable
Rust use this contract, while Python-only installations use the unchanged Python
implementation and emit a fallback warning. Select `backend="python"` to retain
Python's original dtype, mask, and arithmetic behavior.

Physical integers above 2^53 can round when converted to float64. This relaxation
applies to physical data. Identifiers, lookup indices, unit selectors, iteration
limits, classification/status codes, and flags retain separate integral or
boolean storage. Missing or nonintegral lookup controls fail before workers run.
Fuel identifiers have a separate rule: IDs outside 1–256, including NaN or
infinite fuel IDs supplied through Python, select undefined row zero. Lookup
tables contain 257 rows, with row zero always undefined. Both backends preserve
the existing undefined-fuel calculation path and zero spread, without clamping
to a real fuel or adding a NaN/code/flag/storage policy. Crown potential-fire
calculations retain their existing FM10 source and use supplied Surface results
for final behavior. Custom registration outside 1–256 returns `False`, so custom
models 257–260 are no longer supported. Model 256 remains supported.
See the official [C++ definition bounds](https://github.com/firelab/behave/blob/f3b2943e69abca942e795c5b1b45276f6338abb5/src/behave/fuelModels.cpp#L863)
and [undefined-fuel spread path](https://github.com/firelab/behave/blob/f3b2943e69abca942e795c5b1b45276f6338abb5/src/behave/surface.cpp).

Existing valid unit-selector conventions and finite invalid-FDFM-index sentinel
values remain unchanged. Classification arrays are int32;
-1 means unavailable. Classification flags are false for code -1, so check the
code before interpreting those flags.

## Missing values and physical domains

NaN means unavailable physical data. A missing condition or denominator does not
silently select a zero fallback. Availability follows actual field dependencies:
unrelated outputs and valid selected branches can remain available. Explicit
native `validity=false` skips scientific work and leaves initialized unavailable
outputs. Individual masked values do not imply whole-cell invalidity.

Finite equations, thresholds, units, clipping, and undefined-fuel conventions
remain scientific rules. Physical arithmetic follows float64/IEEE behavior,
including infinity, signed zero, and NaNs produced by real-domain arithmetic.
Rust scalar tools therefore can return NaN or infinity where Python's scalar
`math` operations raise an exception. Structural and invalid-control errors
remain exceptions.

## Automatic selection and complex data

Auto selects usable Rust for real inputs, including masked arrays. An unavailable
or incompatible extension selects Python with a once-per-process
`BehaveBackendWarning`, including normal pure-wheel installations. Discovery and
explicit Python selection do not consume that warning.

When Python is already selected because no usable extension exists, calls retain
Python's original complex/object input handling. Rust-specific complex preflight
and input snapshotting are bypassed; Python's own exceptions still propagate.

Only independently tested complex counterparts may route to Python before any
Rust conversion or execution. The first such complex operation emits a separate
once-per-process warning; later complex operations share that warning state.
Native errors and general object/custom arithmetic do not trigger replay.
Explicit Rust rejects unsupported complex/object inputs without discarding
imaginary values. See `src/behave_fire/_auto_routing.py` for parameter-specific
component proofs.

Automatic FireSize instances snapshot all physical fields and caller edits before
verified complex backing/flanking calculations or direct state getters move to
Python. The instance then stays on Python, including after serialization; later
real inputs do not switch it back. Unsupported or lossy complex state methods
raise before mutation. A warning converted to an error leaves the delegate
unchanged. Explicit Rust tools retain their strict real-data contract.

## Grids, ownership, and serialization

Existing public model output names, independent dependency shapes, parameter
names/defaults, and unit conventions remain intact. Native `SurfaceGrid` and
`CrownGrid` have one explicit common grid, with scalar or per-cell inputs.
Strided, readonly, endian-swapped, empty, and scalar arrays are copied in logical
order before detached work. Getters and results own their data independently.

Grid inputs and injections use base units; reaction intensity is BTU/ft^2/min.
`run_remaining` preserves transient injections and rematerializes freed
ancestors. Explicit producer runs replace their own transient values. Reset and
fused restart clear transient injections while retaining input edits and
persistent overrides. Fused runs use the configured whole-model kernel when
possible. New grid sessions require Rust, have no Python fallback, and reject
pickling explicitly. Existing facade/tool serialization remains API4 and stores
values rather than native pointers.

Explicit validity=false hides every derived output, including pinned overrides,
as NaN, -1, or false. The original injected values remain stored and reappear if
validity is enabled later. Classification flags can select final Crown behavior:
Surface takes precedence over passive, then active. All-false flags mean unknown;
flag injection does not rewrite an independently injected fire_type code.

The package-root BehaveRun uses composition and can have a different class
identity/repr from `behave_fire.behave.BehaveRun`. Use the explicit Python class
for legacy subclassing or class-identity checks. All 89 numerical operation
entries now advertise the same `f64` contract. The core is standalone Rust with
no normal/build dependencies and forbids unsafe code; NumPy belongs to the
Python wrapper, not the standalone crate.
