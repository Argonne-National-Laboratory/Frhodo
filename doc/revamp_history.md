# Optimization revamp — development history

Compact record of the bucketed development effort behind the current
optimizer (2026-06 → 2026-07). What each bucket built or decided, with
verdicts for work that was validated and rejected — the negative
results are load-bearing knowledge. Shipped behavior is documented in
[optimization_changelog.md](optimization_changelog.md); this file
records the arc and the evidence.

## Buckets

- **B0 — opendsm re-vendor.** Re-vendored from upstream main, local
  modifications re-applied, provenance recorded.
- **B1 — Objective rebuild.** Per-shock standardized robust losses,
  user × quality × coverage weight composition, stationary anchor,
  known-truth synthetic harness. The objective specification lives in
  [objective.md](objective.md).
- **B2 — Optimization-outcome views.** Typed per-iteration events
  behind a view router; misfit map, Arrhenius-with-bounds, ratios,
  improvement, time offsets, objective trace.
- **B3 — Bayesian path disposition: cut.** The CheKiPEUQ vehicle
  (relative-% objective, per-iteration object rebuild, abandoned
  upstream, zero coverage) was deleted; the objective is
  residual-only. The replacement idea — a native soft-bounds prior in
  parameter space — was built in B5, validated against over four
  registered rounds, and deleted: box constraints contain runaway
  parameters and the post-fit audit reports flat directions, covering
  both motivations without distorting the objective.
- **B4 — Hygiene sweep.** Style, u_incident → u2, SI docs, zone-5
  validity caveat, vendored-binary provenance with pinned SHA256s.
- **B5 — Sensitivity rung 1: screening + balance.** Pre-fit reaction
  screening (importance, leverage, identifiability), background
  ranking runs with score sorting, hybrid coverage ×
  footprint-uniqueness experiment balance as default (4-round
  pre-registered validation), post-fit audit + band-utilization view.
- **B6 — Sensitivity rung 2: adjoint gradient. Rejected.** The adjoint
  cost gradient was proven exact against an FD oracle but plateaued at
  an oracle-bias floor and lost the registered Pareto criterion to
  Subplex; the gradient path and LD_* optimizer entries were deleted
  per the pre-registered rule. Kept: the rate-form-agnostic multiplier
  sensitivity backends (screening, weighting, and Smurf consume them).
- **B7 — Sensitivity rung 3: Smurf.** Sensitivity Multistart Rate
  Fitting: per-experiment residual attribution via time-resolved
  multiplier sensitivity, rate-shape fits, LM-damped steps from Sobol
  multistarts under a shared eval budget. Promoted to default global
  at 16 starts / 400 evals (high-n median 0.019722 vs RBFOpt 0.020692
  at 0.18x wall; worst-seed 1.9% letter-miss disclosed and accepted).
  Field-basis Subplex (per-reaction 4-basis field coefficients) won
  the local shootout (median 0.018939 vs whitened 0.019206, ~7x faster
  to whitened-final quality) and is the default local. Racing and the
  standalone multistart driver were deleted at closeout.
- **B8 — Direct Troe-space local stage. Rejected.** Optimizing
  transformed Troe parameters directly (bypassing the per-eval Troe
  refit) lost at equal evals, equal wall, in legs ablation, and with
  an SLSQP spike. Key finding: refit noise measures 2–4 orders above
  the refit-free objective but is NOT the binding constraint — the
  refit's per-eval projection onto the Troe manifold is load-bearing.
  Pivot executed: the numba-fused Troe-fit objective shipped instead.
- **B9 — Quick-filter mode. Rejected as a mode, restored as presets.**
  A replacement-grade multi-fidelity ladder failed its registered
  validation and was deleted; the filter re-scope failed its wall gate
  by shared-coarse arithmetic. Both survive as explicit triage presets
  ("Smurf (quick, low fidelity)", "Subplex (quick, field basis,
  multi-fidelity)"), not defaults, with measured quality documented:
  13/14 within 10% of full-pipeline answers, median +3%, ~half wall.

An interstitial workstream (not a bucket) followed B9: persistent
worker pool with in-place re-init and bench-derived sizing, shared
sensitivity cache, screening solves farmed across the pool
(steady-state optimization prep ~2 min → seconds), program settings
window, and optimization-view navigation fixes.

## Methodology notes

- Delete-on-fail with pre-registered criteria was enforced throughout;
  every rejected artifact above was deleted, not shelved.
- Cross-mechanism comparison requires a shared nominal anchor: scoring
  mechanisms under objectives anchored at different mechanisms
  produced invalid rankings three separate times. A refit-free
  common-anchor mechanism scorer is the recorded prerequisite for any
  future comparison tooling.
- Post-mortems ask "wrong artifact or wrong gate?" before a deletion
  executes — the B9 wall gate was arithmetically unreachable, a
  criterion-design failure rather than an artifact failure.
