# Rust backend implementation contract

## Float64 raster migration

Binding API4 routes public Surface/Crown models through the standalone float64
raster engine. Physical inputs widen once; masked values become NaN. Physical
outputs are plain float64 arrays, and classifications are int32 with -1 for
unavailable values. Each output retains its public shape. Empty union grids use
actual dependency-restricted native groups for nonempty independent Crown
outputs. Common models use one compute boundary and transfer final core buffers
directly; workers do not concatenate output vectors.

The per-operation `__numeric_contracts__` manifest identifies migrated models,
already-float64 stateless families, and pending component/state helpers. The
remaining compatibility implementation is temporary and does not define the
new model contract. Old API3 extensions and native serialized states are rejected.

Auto prefers usable Rust and warns once per process when native unavailability
requires Python. Proven compatible complex inputs select Python before any
coercion or scientific calculation, with a separate warning. Masked real inputs
stay on Rust. Requested policy, preferred available backend, and last completed
call backend are separate properties. See [BACKEND_DIFFERENCES.md](BACKEND_DIFFERENCES.md).

Core feasibility and the [public serial checkpoint](performance/2026-11-08-plain-normalization-public-checkpoint.md)
have passed. The fresh screen and seven-new-pair confirmation both pass all 56
bounds. Staged grid APIs and remaining helper migration can proceed; final
large/four-thread, memory, allocation, and artifact acceptance still remain.

## Goal and reference

Port the current Python calculation library to a self-sufficient Rust crate, provide Python bindings, and let Python callers select Python or Rust implementations. Put the Cargo workspace in root `rust/`. Deliver a mixed Python/Rust wheel and an independently usable crates.io package, following the local `cffdrs` repository's architecture and release process.

Behavior reference: `behave_fire` commit `7933789f00470c6bbfbe942dc22bdd7f74e192f9`. Architecture reference: sibling `../cffdrs` at `ec7d01909b7ef00ddc359cf4ff02f15b9786a380`. Read-only reference files include its `rust/Cargo.toml`, both crate manifests, `src/cffdrs/cffbps/auto.py`, Rust facade, `tools/smoke_test_wheel.py`, and `.github/workflows/{ci,release,release-crate}.yml`.

This tracked contract defines the implemented Rust backend. Native execution is available through the explicit and automatic interfaces when the compiled extension is installed; the pure wheel retains Python execution. Release and registry operations require separate authorization.

## Implementation status

- **Complete:** every public Python calculation and stateful helper has a Rust core equivalent and an explicit native binding route, and the extension advertises a complete backend. The automatic and explicit selectors, serialization, ownership, and packaging contracts below are implemented and tested.
- **Delivered:** the work landed on `pre_release` as an eight-layer dependent PR stack (contracts, core, surface and crown, other domains, bindings, dispatch, wheels, crate packaging), followed by the performance measurement harness, typed scalar stages, and raster columns described in [RUST_ENGINE.md](RUST_ENGINE.md).
- **In progress:** the binding uses the float64 raster engine for Surface/Crown models; the new public speed targets are not yet certified. See [RUST_BACKEND_BENCHMARKS.md](RUST_BACKEND_BENCHMARKS.md).
- **Not published:** neither the Python package nor the Rust crate has been uploaded to PyPI or crates.io.

## Chosen architecture

Use two crates, matching cffdrs:

- `rust/crates/behave-fire`: publishable core package; Rust import name `behave_fire`. Owns all numerical calculations, default data, typed inputs/results, custom model registration, and serial/parallel batch execution. No Python, NumPy, C++, network, filesystem-data, or sibling-repository dependency at build or runtime. Use the Rust standard library with no external runtime dependencies, as cffdrs does; Rust-only development dependencies are allowed.
- `rust/crates/behave-py`: unpublished PyO3 binding package; builds `behave_fire._rust`. Depends on the core by a relative path. Handles Python objects, array conversion, exceptions, warnings, and output ownership, not a second implementation of the equations.

A mixed Maturin wheel per supported platform installs the existing `src/behave_fire` Python package and its native extension. Each release also includes a pure-Python `py3-none-any` wheel of the same distribution/version, with matching metadata and no native extension. It provides a compiler-free route only on verified interpreter/platform combinations with binary wheels available for all required dependencies. The independently packaged core contains everything needed by a downstream Rust project after extraction from a `.crate` archive.

Alternatives considered: embedding equations directly in the PyO3 crate would violate independent Rust usability; a separate native Python distribution would complicate installation and differs from cffdrs. Keep two product crates and one Python distribution, adding the universal fallback wheel alongside its platform wheels.

## Python API and selection

Keep existing calculation implementations in `src/behave_fire/components/` as the reference. Preserve existing component import paths and `from behave_fire.behave import BehaveRun` as explicit Python compatibility routes. Expose explicit backend namespaces through `behave_fire.python` and `behave_fire.rust`, plus the automatic root facade:

```python
from behave_fire import BehaveRun, FuelModels, available_backends, get_backend
from behave_fire.python import BehaveRun as PythonBehaveRun
from behave_fire.rust import BehaveRun as RustBehaveRun

fuels = FuelModels()
model = BehaveRun(fuels, backend="auto", threads=1)
assert model.backend_in_use in {"python", "rust"}

python_model = BehaveRun(fuels, backend="python")
rust_model = BehaveRun(fuels, backend="rust", threads=4)
contain = get_backend("rust").contain.run_contain_sim_array
```

The public automatic constructor is `BehaveRun(fuel_models, *, backend="auto", threads=1)`. All current calculation method signatures, positional behavior, output dictionaries, unit arguments, and integer enum values remain compatible. `backend_in_use` is read-only and chosen at construction because this project has no separate initialization method. Construct a new instance to change it. Explicit Python and Rust classes do not accept `backend=`; Rust accepts `threads=1`. The explicit Python class remains the existing class and does not accept `threads`. Use the automatic facade for a signature-uniform constructor. On its Python route, `threads` is ignored without validation, matching cffdrs; on Rust, the binding validates it before conversion to `usize`.

`available_backends() -> list[str]` reports importable, compatible implementations. `get_backend(backend="auto")` returns an explicit backend namespace exposing `BehaveRun`, data classes, and each existing component module by name. This gives standalone functions such as ignition, containment, slope tools, and fuel moisture a selector without changing every historical component function signature. Document `from behave_fire.python import contain` and `from behave_fire.rust import contain`; do not promise unimplemented nested import paths.

Importing the root package or explicit Python namespace must never probe or import `_rust`. At selection time, use `importlib.util.find_spec("behave_fire._rust")`. Cache one probe outcome per process: absent, usable, or unusable with its cause. An absent extension, or one failing to load, API-incompatible, or incomplete, selects Python with a public `BehaveBackendWarning` (`RuntimeWarning` subclass), naming the cause and original exception text. Emit that fallback warning once per process at the first auto selection using an absent or unusable result, attributed to its caller; an earlier `available_backends()` query must not consume it. Forced Rust raises a chained actionable `ImportError` for absent or unusable extensions. Unexpected discovery exceptions propagate. Unknown selectors raise `ValueError`. Facade and calculation errors never trigger fallback.

Users wanting warnings-as-errors must configure the filter before the first warned auto selection; the once-per-process policy does not retroactively reevaluate a warning already emitted. Mark a warning as emitted only after `warnings.warn` returns successfully, so a warning promoted to an exception cannot let the next construction silently bypass strictness. Use fresh subprocesses to test cached outcomes and strictness. Existing model instances never change backend after selection.

Binding API version 2 carries independently selected float32 exponential and atan2 profiles. API 1 extensions and native serialized states are rejected, even when marked complete, because they cannot honor the new atan2 metadata. Rebuild or reinstall the native extension after updating the Python package. The value-state schema remains version 1.

NumPy CPU target names alone do not identify its linked float32 atan2 implementation. For a nonempty float32 direction calculation, the binding identifies that implementation once per process using sixteen fixed finite constants, one NumPy atan2 call, and exact native candidate payloads. No user data or fire formula enters this calibration. A lock protects cached success or immutable failure text; unknown or ambiguous implementations raise an actionable error. Float64, empty, and earlier invalid-shape paths do not require this float32 configuration. Scientific calculations still run entirely in the native core, with no Python calculation fallback. The standalone core needs no Python or NumPy.

Use separate extension markers: `__binding_api_version__` for marshalling/ABI compatibility, `__capabilities__` for implemented operation names, and `__backend_complete__` for completed inventory coverage. Explicit `behave_fire.rust` imports and `--require-rust` tests check the API version but permit incomplete development builds. An unimplemented explicit operation raises `NotImplementedError` naming it, never calls Python. Automatic and forced selectors require a true completeness marker and every operation in the static versioned capability contract (additional advertised operations are allowed). An absent, malformed, or partial capability inventory is unusable even when the completeness marker is true. The static contract imports no native code and is checked against the tracked API inventory. Automatic and forced selectors require completeness: auto warns and selects Python for an incomplete extension; forced `BehaveRun(..., backend="rust")` and `get_backend("rust")` raise `ImportError`. `available_backends()` lists Rust only when usable and complete; absent/unusable results return only Python without consuming the auto-fallback warning. This allows Tasks 7–9 to test explicit native functionality before Task 10 enables selection. Native release artifacts require complete coverage; the pure wheel intentionally has no extension.

The Rust facade snapshots custom `FuelModels` inputs at construction, matching the current Python facade's cached lookup. Changing the original registry afterward does not update an existing model. Explicit Rust registries support their own typed add/remove/get operations. Python adapters may retain Python metadata containers and convert them into Rust snapshots, but every native calculation must call the core. Expose standard Rust fuel and species tables independently; Rust consumers never need to supply Python-generated tables.

### Identity, registries, and serialization

The root automatic facade uses composition, not inheritance from the Python scientific implementation. Exact class identity, its default repr, and `isinstance(auto_model, behave_fire.behave.BehaveRun)` therefore change intentionally. Document this migration; callers needing legacy type identity or subclassing retain `behave_fire.behave.BehaveRun` or its explicit-Python alias. Existing root `FuelModels`, species types, and unit enums keep their identities. Do not claim blanket class-identity compatibility. Historical static calls on the root class retain the Python implementation without probing the extension; calls on model instances forward to the selected backend. All 11 public calculation signatures remain discoverable on the class and its instances.

| Constructor | Python FuelModels | Rust FuelModels wrapper |
| --- | --- | --- |
| Legacy/explicit Python BehaveRun | Accepted unchanged | Convert explicitly using `to_python()` first |
| Explicit Rust BehaveRun | Snapshot to native records | Snapshot native records |
| Automatic facade, either selected backend | Accepted | Convert a snapshot for Python; snapshot natively for Rust |

Provide explicit `RustFuelModels.from_python(registry)` and `to_python()` conversions that preserve all fields and reserved/custom status. Never expose a mutable Python dictionary as a live alias of native storage. Test the same snapshot-conversion contract for species registries and coefficient-building entry points.

Preserve working pickle and deepcopy behavior for built-in model/registry classes. Implement versioned serialized state using values/arrays, never pointers or a live thread pool. Persist the selected backend, thread setting, and actual cached numerical snapshot, not just a reference to a registry that may have changed since construction. Unpickling a native-selected model without its compatible extension raises an actionable `ImportError`; it must not silently reselect Python. Test shallow-copy behavior against the reference, independent deepcopy state, protocol round trips, and Windows multiprocessing spawn. Existing pickles referencing historical Python class paths must remain loadable. Arbitrary user subclasses keep the legacy explicit-Python route; no promise to serialize custom native subclass machinery is made.

## Coverage and compatibility

Completion means a Rust equivalent for every public calculation and stateful helper in the current components, plus the constructor and all 11 public facade methods: surface, crown, scorch height, crown-scorch mortality, three spotting methods, and four fire-geometry methods. The explicit API inventory (`tests/data/rust_api_inventory.json`) lists each public name and its native route; the original pytest suite and legacy harness alone were not complete coverage of the library.

| Python component | Core module and responsibility |
| --- | --- |
| `behave_units.py` | `units.rs`: enum values and every conversion direction |
| `fuel_models.py`, `fuel_models_array.py` | `fuel_models.rs`, `generated/fuel_models.rs`: standard/reserved/custom records and lookup construction |
| `species_master_table.py` | `species.rs`, `generated/species.rs`: records, region availability, equation identifiers, and lookup methods |
| `surface.py` | `surface/`: particles, fuelbed intermediates, reaction, wind/slope, geometry, and orchestration |
| `crown.py` | `crown.rs`: each stage and composed calculation |
| `mortality.py` | `mortality.rs`: coefficient tables, scorch, and implemented mortality equations |
| `spot.py` | `spot.rs`: surface fire, burning pile, and torching-tree spotting |
| `ignite.py` | `ignite.rs`: fuel temperature, firebrand, and deprecated compatibility lightning |
| `safety.py` | `safety.rs`: zone geometry |
| `fine_dead_fuel_moisture_tool.py` | `fine_dead_fuel_moisture.rs`: embedded lookup tables and indexing rules |
| `vapor_pressure_deficit_calculator.py` | `vapor_pressure_deficit.rs`: equations and units |
| `slope_tool.py` | `slope.rs`: scalar/stateful helper behavior |
| `fire_size.py` | `fire_size.rs`: FireSize state and methods, distinct from surface geometry functions |
| `contain.py` | `contain.rs`: current fixed-step compatibility model and statuses |

Port Python behavior, not the C++ reference as an alternative oracle. Preserve the recent 18 metric record fixes, 23 static flags, fractional moisture exceptions, and upstream-aligned heat constants. Do not add missing science, an actual RK4 containment solver, or optional new upstream outputs during the port. Record existing limitations and intentional differences in `docs/BACKEND_DIFFERENCES.md`. Discovery of a Python defect requires an explicit separate fix decision, not silently different Rust results.

The main model pipeline and standalone Rust default APIs use `f64`. Public helpers that retain float32 use native `f32` paths with the reference operation order. Rust does not implement float16 arithmetic or emulate its intermediate rounding. At the Python boundary, widen float16 inputs to float32, preserving their values, shapes, and any mask metadata; then apply the normal per-entry-point normalization. A helper retaining its input precision returns float32, while an API normally normalizing to float64 still returns float64. Mixed-input promotion follows the reference after this normalization. The Python backend remains unchanged and preserves its existing float16 behavior.

The acceptance oracle for Rust is the explicit Python implementation given the same normalized inputs, with float16 inputs widened to float32 first. Also widen operands of a NumPy operation whose resolved loop would create float16 internally: currently the direction helper's `arctan2` for some bool/int8/uint8 combinations. Resolve that loop structurally, preserve masks/scalar kind, and widen only the relevant operands; do not globally cast integer arrays or erase wrapping behavior. Compare float64 outputs at `rtol=1e-9, atol=1e-12` and float32 outputs at `rtol=1e-6, atol=1e-7`. Against that oracle, match integer/status values, threshold decisions, dtype, shape, masks, NaN positions, infinity signs, and exception types exactly. Keep paths that need no widening strictly aligned with the original Python contract.

Separately compare the widened route with original float16 behavior as a diagnostic, not an acceptance gate. Widening can change overflow/underflow, finite versus infinite results, NaNs, exceptions, and classification thresholds as well as dtype. For example, the existing eccentricity helper returns `inf` for a float16 array containing 300, while its float32 route returns approximately `0.9999944`. Record such differences in `docs/BACKEND_DIFFERENCES.md`; do not loosen acceptance tolerances or claim exact original-float16 classifications. Finite-value diagnostics may report a `5e-3, 5e-4` comparison, but failures against original float16 do not invalidate correct normalized behavior. Users requiring original float16 semantics must select Python.

The frozen non-finite-to-int32 diagnostics retain their original x86 conversion
results. NumPy can convert NaN and positive infinity differently on another host.
Only four exact original input records qualify for portability handling:
`coerce_crown_fire_inputs/nan`,
`coerce_crown_fire_inputs/positive-infinity`,
`calculate_lightning_ignition_probability/positive-infinity`, and
`calculate_fine_dead_fuel_moisture/nan`. Their entry points and encoded input
hashes must match, and the current host must demonstrate a different int32 cast.
Native execution then must match live explicit Python on those same original
inputs, retaining the existing float tolerances and exact integer, dtype, shape,
mask, exception, and warning requirements. A separate replay with the original
finite `INT_MIN` cast must still match the unchanged frozen diagnostic. Other
cases continue to use their original frozen acceptance checks.

The exporter deliberately canonicalizes those four cast-dependent outputs by
replaying the original finite `INT_MIN` conversion. Default generation and
`--check` therefore use the same reviewed snapshots across hosts. These snapshots
are canonical invalid-cast diagnostics, rather than raw current-host outputs.
`tools/probe_platform_oracle.py` records the raw original-input Python and optional
native results; `tools/compare_backends.py` reports frozen and live outputs
separately when portability handling activates. Neither path rewrites scientific
fixtures or relaxes numeric tolerances.

Characterize float16/32/64, signed/unsigned integers, booleans, and object inputs for each public helper. Preserve successful numeric coercion and existing exception types against the selected oracle. Additional floating dtypes require an explicit documented contract before claiming support. Match NumPy's clipping, remainder, promotion, tie-breaking, and branch semantics deliberately; do not globally relax tolerances to conceal defects.

The supported native arithmetic boundary is bool, fixed-width signed/unsigned
integers, and float32/float64 after normalization. Preserve characterized
numeric-object coercion, output boxing, and exception behavior; do not reject an
input solely because its dtype is object. Preserve large-Python-integer inputs
when the reference explicitly coerces them to a supported real dtype before
arithmetic. General arbitrary-precision integer outputs and opaque custom
Python-object arithmetic require explicit Python; native calls reject unsupported
inputs with `TypeError` before detached execution. This includes the object result
of `calculate_fire_length(2**64, 0, 1)`. Do not silently approximate such results
or fall back during a calculation. Preserve exact fixed-width wrapping and weak
integer prefixes on supported paths, and complex intermediates generated from
supported real inputs. See [BACKEND_DIFFERENCES.md](BACKEND_DIFFERENCES.md).

Characterize scalar, length-one, 1D/2D/3D, empty, mixed-fuel, and non-contiguous input behavior by entry point. Preserve documented shape restrictions, including containment's report-spread-rate-defined shape. Exception classes match the reference; exact message text is not an API guarantee. Malformed boundary shapes/types must raise `ValueError` or `TypeError`, never a Rust panic or PyO3 `PanicException`. Add malformed-input/fuzz cases.

Masked-array policy is to reproduce each entry point's observed behavior, including its output type and whether the mask affects values, disappears, or survives. Local probes show differing behavior: unit conversion discards the mask and uses underlying values, while eccentricity produces an ordinary array with masked positions affected by the masked computation. Do not globally strip masks during copying or invent uniform mask propagation. Pass data and mask metadata separately when the native implementation needs them. Freeze these cases in Task 1 and verify them in Tasks 7–8.

## Native API, memory, and parallelism

Provide public typed scalar input/result structs and computational functions for each domain, a `FuelModels` registry, a `SpeciesMasterTable`, and a `BehaveRun` Rust facade with the same calculation coverage. Units and status enums are public. Batch APIs use shape metadata plus owned results and scalar-or-slice fields, with checked element counts and lengths. The standalone core offers the same models as the binding, not just private routines that PyO3 can call.

Propose an initial core version of `0.1.0`, subject to release authorization. Mark extensible result structs and error/status enums `#[non_exhaustive]` where practical; provide constructors/accessors and document public stages as versioned API. Use named per-field vectors for high-volume surface/crown batch results so bindings can transfer each field into NumPy without transposing an array of structs. Retain `GridOutput<R>` only for small generic results if useful; benchmark any remaining copy cost explicitly.

The Python binding accepts dynamically dimensioned NumPy arrays. Normalize supported inputs and copy logical row-major values into owned buffers before releasing the GIL with `Python::detach`. Never retain Python-owned mutable array slices across detached execution. Support C-contiguous, Fortran-contiguous, sliced, reversed, broadcast, and read-only arrays according to the documented contract. Return fresh arrays with the normalized reference shapes and dtypes, including the stated float16 widening. Python inputs must remain unchanged.

A public calculation spanning multiple native stages snapshots its caller arrays,
masks, and nested containers before the first stage. The facade, field partitions,
empty-output reconstruction, and result metadata restoration share that snapshot.
Nested stages reuse owned buffers and views. Direction-kernel selection uses the
original normalized operand layout before copying, preserving reversed and
strided float32 atan2 behavior.

`threads=1` is serial; `threads=0` uses available parallelism; a positive count requests that many workers, capped by actual work. Reject negative and non-integer values in Rust-facing APIs. The automatic Python route ignores `threads`, matching cffdrs. Parallelize independent cells without changing within-cell arithmetic or reduction order. Use scoped standard-library threads in the core, bounded per call; avoid a global thread pool and accidental oversubscription. Thread counts must produce identical per-cell results. Tune serial thresholds from measurements rather than copying cffdrs's 4,096-cell threshold without evidence.

## Wheel process

Use cffdrs's Maturin/PyO3 process for all seven tested native platforms: Linux x86-64 and aarch64 using manylinux 2_17, Alpine x86-64 using musllinux 1_2, macOS x86-64 and arm64, and Windows x86-64 and ARM64. Each platform must pass real installation checks before certification. No PyPy, free-threaded CPython, GPU, or WebAssembly support is proposed.

The selected Python floor is >=3.11, with `abi3-py311` / `cp311-abi3`, matching cffdrs. Python 3.10 support is dropped. The target certification matrix is CPython 3.11, 3.12, 3.13, and 3.14, explicitly tested by Linux Python-only CI from Task 2 onward; support is certified only after relevant artifact/dependency checks pass. Retain NumPy >=2.1.3 generally and use >=2.3.3 for Python >=3.14 or Windows ARM64, validating platform markers on the actual runner. The universal fallback wheel does not extend certification to PyPy, free-threaded builds, untested platforms, or future interpreters.

Use cffdrs's existing PyO3/rust-numpy 0.27 pairing and Maturin >=1.7,<2.0 as candidate versions, with Rust 1.78 / edition 2021 as candidate MSRV and edition. Confirm them with actual pinned dependency resolution and MSRV checks before claiming support. Validate exact applicable NumPy floors and current allowed binary versions on each certified row. Generate and commit `rust/Cargo.lock` in Layer 2 under the recorded stack authorization, before that layer's CI runs `cargo test --locked`. Root `uv.lock` is introduced with the Python packaging/dependency workflow.

Use `uv` and root `uv.lock` for Python development/CI dependency locking, matching cffdrs; the lock does not replace published dependency metadata or exact-floor tests. Commit lockfiles only when commit permission is given. Python-only CI installs dependencies from the lock with the project itself excluded (`uv sync --locked --no-install-project` with appropriate extras), then runs tests from the checkout using `pytest.ini`'s relative `pythonpath = src`; it does not invoke Maturin or require Rust. Ensure the selected extras are actually installed by that command and that `_rust` is absent in this job.

Configure `python-source = "src"`, `module-name = "behave_fire._rust"`, `manifest-path = "rust/crates/behave-py/Cargo.toml"`, and `sdist-generator = "git"`. Task 7 introduces only `[tool.maturin]` for native development builds; preserve working setuptools packaging until Task 11 switches the build backend and manifest rules together. If development builds require the earlier switch, move the complete packaging transition into that layer. The sdist must contain both crates, their manifests, lockfile, tables, and build inputs. Rebuild a wheel from the extracted sdist without a Git checkout or sibling repositories. Exclude `development/`, `.claude/`, build outputs, and caches. The ignored plan is never a build input.

Use dynamic Python metadata from the binding crate, following cffdrs. Preserve the current version unless the owner approves a release bump; verify Cargo's prerelease spelling maps to the intended PEP 440 value (currently 1.0.0b2). Derive `__version__` from installed distribution metadata with a tested source-checkout fallback. Expose native core and binding version identifiers for diagnostics. `tools/build_pure_wheel.py` stages the Python package in a temporary build project using `setuptools.build_meta`, with the same distribution/version, dependencies, extras, Python requirement, license, classifiers, and readme as the native wheels. Reject compiled leftovers in the staged source. The helper does not modify root metadata or require Rust.

Publish the pure wheel alongside platform wheels. Installing that behave-fire artifact itself never requires compiling behave-fire or installing Rust; NumPy's binary availability determines whether dependency installation also avoids compilation. On a covered platform pip must prefer the compatible native wheel; where only the pure wheel is compatible, auto selects Python. Compiler-free installation of the whole environment is promised only where `pip install --only-binary=:all:` succeeds for the project and required dependencies. Optional extras have separate matrix claims. Editable, git, sdist, and `--no-binary` project builds still require Rust. State both project and dependency requirements clearly in contribution/release documentation.

Alpine and Windows ARM64 now have native wheels alongside the original five targets. They also retain independent pure-wheel installation checks:

| Fallback target | Execution environment | Python matrix | NumPy floor to verify |
| --- | --- | --- | --- |
| Alpine/musl Linux x86-64 | `ubuntu-latest` host running `python:3.11-alpine` through `python:3.14-alpine` containers | 3.11, 3.12, 3.13, 3.14 | 2.1.3 for 3.11–3.13; 2.3.3 for 3.14 |
| Native Windows ARM64 | `windows-11-arm`, setup-python `architecture: arm64` | 3.11, 3.12, 3.13, 3.14 | 2.3.3 |

The PyPI metadata audit found matching musllinux x86-64 wheels for NumPy 2.1.3 on 3.11–3.13, and matching musllinux x86-64 plus win_arm64 wheels for NumPy 2.3.3 on every 3.11–3.14 row. These are [published wheel files](https://pypi.org/project/numpy/2.3.3/#files), not completed runtime certification. [GitHub documents the Windows ARM runner](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run); verify availability and native interpreter architecture when implementing CI.

Run artifact checkout/download actions on the Ubuntu host, then use container Python for Alpine smoke checks with only staged artifacts/scripts mounted outside the checkout. Pin tested container digests. On Windows ARM64, verify ARM64 interpreter/pip tags rather than accidentally exercising x64 emulation. On every target, test exact conditional floors and current allowed binary dependencies. With the complete artifact set, require native-wheel selection, silent automatic Rust selection and matching calculations. In separate environments offering only `py3-none-any`, require Python-only availability, silent automatic Python selection and correct forced-Rust failure. Both routes are required on all seven platforms.

Keep an explicit result for every requested matrix row. Missing dependency wheels or runners produce a named unverified/unsupported outcome, never silent removal, source-build substitution, or `continue-on-error`. They block certification for that row and require explicit scope revision before a release omits a requested target.

Maturin's git sdist mode uses the tracked file set. Ensure required generated tables/manifests are tracked, not merely present as untracked or ignored files, before testing the sdist. Release builds additionally use a clean, committed tree. Git tracking and committing are different operations; do not claim the generator reads only committed content. Verify the unpacked artifact contents directly.

The release workflow builds/tests, checks MSRV, builds an sdist plus native and pure wheels, installs each artifact in an isolated environment outside the checkout across its tested matrix, runs native/Python parity or pure-fallback smoke checks as appropriate, and runs `twine check`. Native and pure artifacts share metadata; verify wheel selection with dependencies preinstalled from binary wheels and `pip install --no-deps --no-index --find-links ...`, separately from complete dependency-install tests. Only verified artifacts reach TestPyPI on manual release dispatch or PyPI on an approved `v*` tag. Use PyPI Trusted Publishing, protected environments, and explicit owner setup. Publishing and tags require separate authorization; provide a build-only manual mode.

Conda packaging has been removed from the repository; Python delivery is through wheels and the source distribution. CI runs for PRs targeting any branch, including the intermediate branches of a dependent stack.

## Independent crates.io delivery

Proposed registry name is `behave-fire`, library name `behave_fire`; check availability and ownership before the first upload. The core has its own version and `behave-fire-vX.Y.Z` tag, independent of Python, as cffdrs does. Use a separate `release-crate.yml`, defaulting to dry-run, with an explicit publish input and protected `crates-io` environment.

Ship core-local README, license, source, embedded/generated tables, examples, and any fixtures needed by packaged tests. Core code and doctests cannot use `include!`/`include_str!` outside the crate or launch Python. Generation tools may use the Python reference during development, but generated Rust tables are checked in and generation is never required by `cargo build`, `cargo package`, or `cargo publish`.

Retain the repository's existing MIT license and copyright notice in the core package; inventory and preserve applicable upstream source/data notices and scientific provenance. No relicensing is proposed. Any incompatible or uncertain copied material must be resolved before publication.

Set a project budget of 5 MiB for the compressed `.crate` archive, below Cargo's currently documented 10 MB registry limit, and recheck that registry limit at release. Package compact deterministic fixtures covering records and representative boundaries; place large/generated scenario sets under root `tests/data/rust_port/` for full Python-to-native parity runs. Core packaged tests must use only their own compact fixtures. Generate large shape/performance scenarios from seeds rather than serializing million-cell arrays. Add archive-size and file-inventory gates to `check_crate_package.py`.

Gate delivery on `cargo package`, `cargo publish --dry-run`, and tests of the extracted package in a clean Rust-only environment. Add a downstream consumer smoke project using only the packaged core; exercise default/custom fuels, a surface run, and another domain. No PyO3, rust-numpy, Python executable, NumPy installation, C++ compiler, or reference checkout may be necessary. After actual authorized publication, additionally test resolution from crates.io without `[patch]` or path overrides.

Follow cffdrs's first-publication bootstrap and later OIDC trusted-publishing workflow, rechecking current crates.io requirements before release. Owner setup and first upload are separate authorized operations; implementing the workflow does not publish a crate. Core version bumps and Python binding dependency/version updates have separate compatibility checks.

## Delivery and sources

A release may not claim completion while advertised native operations are missing or silently delegated to Python. The original stages (contract fixtures, standalone Rust foundation and data, domain ports, array bindings with explicit namespaces, the automatic wrapper, Python packaging and CI, and crates.io packaging) were delivered as a dependent PR stack. Procedures for any future stack, including how to land it, are in [CONTRIBUTION_PROCEDURES.md](CONTRIBUTION_PROCEDURES.md).

The selected layout and sdist settings follow the [Maturin project layout](https://www.maturin.rs/project_layout.html), [configuration documentation](https://www.maturin.rs/config.html), and [git-mode file selection](https://docs.rs/crate/maturin/1.7.4/source/maturin.schema.json). ABI choices follow [PyO3 build and distribution guidance](https://pyo3.rs/main/building-and-distribution.html). Standalone packaging follows [Cargo workspace rules](https://doc.rust-lang.org/cargo/reference/workspaces.html), [dependency rules](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html), and [publishing size guidance](https://doc.rust-lang.org/cargo/reference/publishing.html). Registry authentication must be checked against [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing) before release.
