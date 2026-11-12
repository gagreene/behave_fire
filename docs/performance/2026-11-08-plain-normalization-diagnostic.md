# Bounded call-setup and variance investigation

The owner approved one bounded follow-up after the API4 public screen missed one confidence bound. This report records diagnostic evidence and the single selected candidate. It does not certify a speedup.

## Fixed diagnostic budget

Base: `c47036f1e063a0c2c9565ab2821fa52b7745e01b`. One detailed Python-call attribution process ran heterogeneous Surface at 10k cells (first plus four warmed calls). Six additional fresh processes ran the same recipe: three with the unchanged memory sampler and three with a diagnostic-only sampler disabled. Order was shuffled with seed 20261108. GC callbacks and CPU times were recorded. All records are marked nonacceptance; the public timing protocol remains unchanged.

## Findings

- Across five profiled calls, `normalize_physical` consumed 8.301 ms cumulatively and constructed 120 MaskedArray intermediates for ordinary inputs.
- Ownership/copy setup also appears in the profile. Some physical float64 arrays are already registered independent snapshots but are copied again by normalization.
- The six fresh diagnostic processes did not reproduce the earlier 42.173 ms warmed process median. Sampled and unsampled durations overlap; observed timed GC events were at most 0.112 ms.
- Windows main-thread CPU timing was quantized in 15.625 ms increments, so it cannot attribute variation precisely for these short calls. No cause is assigned to the original slow process, and no original sample is discarded.

| Diagnostic mode | First ms | Four warmed calls, ms |
|---|---:|---|
|sampled|21.063|16.513, 17.668, 14.294, 14.706|
|no-sampler|22.406|14.044, 14.286, 14.480, 14.099|
|no-sampler|15.802|13.394, 14.492, 15.065, 16.977|
|sampled|16.946|13.755, 13.682, 18.004, 14.944|
|no-sampler|16.362|14.028, 14.048, 14.716, 14.470|
|sampled|15.968|14.726, 14.089, 14.854, 16.279|

## Single implementation candidate

Normalize ordinary real scalars and exact ndarrays directly with NumPy instead of constructing a MaskedArray. Reuse float64 storage only when it is an exact ndarray, C-contiguous, aligned, owns its allocation, and is already registered as an independent snapshot in the current public-call owner. External caller arrays still copy. Masks, sequences, subclasses, byte order, and layouts requiring conversion keep their safe paths. Rust equations, output shapes, missing-data rules, and the acceptance harness are unchanged.

Structural regression tests first observed five unnecessary MaskedArray constructions and one redundant owned copy. All six failures passed after this candidate; focused ownership/mask/fallback tests also passed. The candidate remains unmeasured until the approved fresh complete public screen and, only if it passes, seven NEW confirmation pairs. Another miss stops the work; no second candidate or target relaxation is authorized.

## Evidence

[Diagnostic records](2026-11-08-plain-normalization-diagnostic.json). Raw pstats, stdout, stderr, and CPU/GC records remain in the ignored execution workspace. Absolute profiler paths are normalized in this published copy; the original report hash is retained. The previous failed screen/profile are untouched.
