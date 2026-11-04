# Backend compatibility contract

The Python backend at commit `7933789f00470c6bbfbe942dc22bdd7f74e192f9` is the behavioral reference for the Rust port. Native backend implementation is in progress; this document is not a claim that native execution is already available. The full API, ownership, selection, and artifact contracts are in [RUST_BACKEND.md](RUST_BACKEND.md).

## Intentional numerical differences

The planned Rust backend widens float16 inputs to float32 before applying each entry point's ordinary normalization. Helpers retaining input precision therefore return float32; operations that already normalize to float64 continue returning float64. Rust acceptance uses explicit Python with the same normalized inputs. Original float16 results are diagnostic only: widening can change rounding, overflow, underflow, non-finite results, exceptions, and classification thresholds. For example, `calculate_eccentricity(np.array([300], dtype=np.float16))` returns infinity, while the float32 route returns approximately 0.9999944. Select Python for exact original float16 semantics.

This policy also applies when NumPy would create a float16 intermediate from
other input dtypes. NumPy's `arctan2` chooses a float16 loop for some boolean,
int8, and uint8 combinations in `calculate_direction_of_max_spread`. Rust
widens those x/y operands to float32 before that operation, using structural
ufunc dtype resolution rather than executing the Python calculation. The
acceptance oracle applies the same normalization. Upslope direction may return
float32 instead of float16; north-oriented results retain their later float64
promotion but can differ numerically from original float16 intermediates.
Other integer operations retain their own wrapping and promotion contracts.

Float64 and float32 routes preserve the reference within `rtol=1e-9, atol=1e-12` and `rtol=1e-6, atol=1e-7`, respectively. Shapes, statuses, masks, NaN positions, and exception classes match the applicable oracle; message wording is not guaranteed. Any further difference must be recorded with an executable scenario before being accepted.

## Existing limitations retained

- Containment uses a simplified fixed-step growth/line-production loop, not the full C++ ContainSim solver or RK4. The spread-rate input defines the batch shape, and report size sets the overrun threshold rather than the initial ellipse.
- Some surface entry points require moisture arrays matching the fuel grid rather than arbitrary mutual broadcasting.
- Masked-array handling varies by entry point. Some functions discard the mask; some calculations affect values under masked positions. The port must reproduce each observed contract rather than impose a new universal policy.
- Lightning ignition is deprecated for scientific reliability reasons. Compatibility values remain available with the existing warning.
- Custom fuel lookups in an existing BehaveRun are snapshots built at construction, not live views of later registry edits.

## Planned Python API differences

The root automatic BehaveRun uses composition and therefore changes exact class identity. The historical `behave_fire.behave.BehaveRun` and explicit Python alias retain legacy identity/subclassing. Backend selection is fixed per instance. A missing extension falls back silently for auto; an unusable extension causes a once-per-process BehaveBackendWarning and Python fallback. Forced Rust raises. Calculation failures never cause fallback.

The Python runtime floor will become 3.11 at the coordinated packaging transition. Rust is required for source builds after that transition; native and pure-Python wheels themselves require no project compilation. Complete compiler-free installation also depends on binary dependency availability. See the artifact matrix in the implementation contract.
