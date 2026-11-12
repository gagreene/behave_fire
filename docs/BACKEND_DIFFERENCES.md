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
Existing valid unit-selector conventions and finite invalid-FDFM-index sentinel
values remain unchanged. Classification and containment status arrays are int32;
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

Containment retains fixed distance steps, `dt = dist_step / ros`, terminal
conditions, and iteration limits. A missing value needed by an executed step
makes termination unavailable. Inputs belonging to unselected branches do not
invalidate known results. Zero requested steps retain the existing time-limit
and zero-geometry result without evaluating scientific steps.

## Automatic selection and complex data

Auto selects usable Rust for real inputs, including masked arrays. An unavailable
or incompatible extension selects Python with a once-per-process
`BehaveBackendWarning`, including normal pure-wheel installations. Discovery and
explicit Python selection do not consume that warning.

Only independently tested complex counterparts may route to Python before any
Rust conversion or execution. They emit a separate complex-fallback warning.
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

The package-root BehaveRun uses composition and can have a different class
identity/repr from `behave_fire.behave.BehaveRun`. Use the explicit Python class
for legacy subclassing or class-identity checks. All 89 numerical operation
entries now advertise the same `f64` contract. The core is standalone Rust with
no normal/build dependencies and forbids unsafe code; NumPy belongs to the
Python wrapper, not the standalone crate.