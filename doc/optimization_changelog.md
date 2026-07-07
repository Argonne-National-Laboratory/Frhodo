# Optimization & objective improvements — history

Running history of shipped changes to the optimizer, cost objective,
sensitivity infrastructure, and optimization GUI. Newest first. Each
entry: what changed, why, and how it was validated. Experimental work
not yet in prod is tracked under `development/revamp_pt_2/` and noted at
the end.

## 2026-07 — Sensitivity ladder & local optimizer

### Smurf coarse global stage (DEFAULT global algorithm)
"Smurf (Sensitivity Multistart Rate Fitting)": a global-stage
optimizer that attributes each experiment's residual across reactions
via the time-resolved multiplier sensitivity (a small per-experiment
weighted least-squares), fits each reaction's per-experiment
log-multiplier cloud across the campaign temperature spread to a rate
shape, and takes Levenberg-Marquardt-damped steps scored by the true
aggregate. Reaches the optimum neighborhood in a handful of sensitivity
sweeps and hands off to the local stage.
- Why: replaces the high-dimensional blind global search (RBFOpt at 66
  slots is far too sparse to build a surrogate); ~25x faster at the
  global stage, feeding the same local optimum.
- Fidelity + multistart bridges (fair full-stack, same-anchor
  comparison, high-n final): pre-fix 0.0260 -> matching the objective's
  per-point weighting, Barron IRLS, Gram-diagonal leverage, and a Troe/
  Plog lnP shape term -> 0.0245 -> adding the multistart wrapper
  (incumbent + Sobol perturbations, 4 starts) -> 0.0216 -> 8 starts ->
  0.0202, crossing under RBFOpt's 0.0206 at ~856s global vs RBFOpt's
  ~2231s: on high-n (seed 0) Smurf is now both more accurate and ~1.8x
  faster. Base is a near-tie (0.02768 vs 0.02751, within noise). Each
  bridge made Smurf's per-experiment rate derivation match the real
  objective more faithfully; the multistart addressed local-descent
  basin coverage. Promoted to default at 16 starts: median 0.019722 vs
  RBFOpt 0.020692 (4.7% better) across internal seeds, best seed
  0.017445 (campaign record), at 0.18x RBFOpt's global walltime with
  the pooled sweep; worst seed within 2% of RBFOpt's worst. Default
  config: 16 starts, 400-eval total budget, first in the dropdown.
- Collinear (unidentifiable) reaction pairs are held at the
  minimum-norm split by a ridge on the per-experiment solve.

### Smurf eval-budget contract + multistart-count setting
Smurf's iteration-maximum stop is now a hard evaluation budget shared
across all multistart descents, matching every other algorithm's
max-eval contract (it previously derived only a per-descent cap and
the starts spent freely). The multistart count is a first-class
setting (`multistart_count`, default 16) with its own GUI spinbox:
selecting Smurf swaps the Initial Population Multiplier row for a
Multistart Count row (the multiplier only applies to population
algorithms, and was previously left disabled-but-shown for Smurf, so
the count was not settable from the GUI at all). Pinned by a
budget-contract test. The coarse linearization sweep is pool-parallel
(sim + sensitivity per shock dispatched to workers), removing the
stage's dominant serial cost on multi-core hosts (~8x coarse wall).

### Per-algorithm settings resets + population-multiplier fixes
Each optimization settings box's reset value (right-click Reset /
Ctrl+R) now follows the selected algorithm's recommended configuration
(e.g. Smurf: 400-eval budget, 16 starts; RBFOpt: 600 evals; population
algorithms: 2500 evals, multiplier 1). The multistart-count box uses
the same scientific spinbox as every other setting. The initial
population multiplier is enabled for the pygmo genetic algorithms
(DE/SaDE/PSO/GWO) and — fixing a silent gap — those algorithms now
actually scale their population size by it (previously only the nlopt
population algorithms consumed it).

### Field-basis Subplex (default local stage)
"Subplex (field basis)": the local stage searches per-reaction 4-basis
field coefficients (delta_lnk = c0 + c1 lnT - c2/(Ru T) + c3 lnP, 4 per
reaction) applied on top of the stage start point, instead of the raw
per-grid-point scalers. The search dimension is 4 per reaction
regardless of slot count, and the subspace excludes the
refit-degenerate (Troe-gauge) directions of the full slot space.
- Validation: same-handoff shootout at equal 1500-eval budget: 0.0197
  vs whitened Subplex's 0.0216 (+8.9%), ~7x faster to whitened-final
  quality, identical per-eval cost; seed-paired runs: wins 2 of 3
  internal seeds and the median (0.018939 vs 0.019206), campaign-best
  0.016960. Default pipeline: Smurf (16 starts) -> field-basis Subplex.
- Whitened Subplex stays available (the best full-dimensional engine).

### Whitened Subplex with curvature auto-switch (available local)
A local-stage algorithm "Subplex (whitened)": a one-time diagonal
finite-difference curvature probe at the stage start rescales the
search coordinates, then Subplex runs in the whitened space. An
auto-switch measures the per-slot curvature spread and falls through
to plain Subplex when it is below 1e3 (near-isotropic problems, where
whitening only costs the probe). Now the default local algorithm.
- Why: at high dimensionality the optimized slots mix parameter
  families of very different stiffness (Troe falloff terms vs
  Arrhenius legs); equalizing their scales converts Subplex's subspace
  cycling into far faster progress.
- Validation: registered Pareto criterion, seeds 0-2. At 66 slots,
  2.7-6.6% better objective / ~4x faster to equal accuracy vs plain
  Subplex; at 9 slots it correctly falls through (spread ~52) so the
  few-reaction case is unharmed.

### Analytic objective-gradient path removed
Deleted the analytic gradient (`cost/gradient.py`, `CostFunction.
gradient`, the refit Jacobian, the per-parameter forward/adjoint
sensitivity backends) and the LD_* gradient optimizers from the
algorithm registries.
- Why: the gradient was proven exact but plateaued at an oracle-bias
  floor and lost to Subplex under the registered accuracy-vs-walltime
  criterion; it also only covered plain-Arrhenius targets. Per the
  pre-registered delete-on-fail rule.
- Kept: the rate-form-agnostic multiplier sensitivity backends
  (`compute_adjoint_sensitivity`, `compute_forward_sensitivity`), which
  the screening/weighting path uses.

### Pre-fit reaction screening
Screens every reaction against the campaign at the start mechanism
(two solves per shock, outside the optimization loop): importance,
signed objective leverage, capability footprint, and identifiability
(spectral-gap rank of the stacked slope matrix). Per-shock solves are
cached on a mechanism-version stamp. Surfaced in the GUI with
background runs and score sorting.
- Why: rank reactions for target selection and feed the experiment
  weighting; give the user a pre-fit view of what each reaction can do.

### Experiment balance: coverage x information uniqueness
Cost setting `experiment_weighting` ("uniqueness" | "coverage" |
"none"), default "uniqueness". Uniqueness = geometric condition-space
coverage x an information-uniqueness factor from sensitivity
footprints frozen at the run start; experiments whose simulation
responds to no optimized reaction are dropped. Residual-free, so
contaminated data cannot masquerade as unique information.
- Validation: 4-round validation; default after contamination-risk
  testing.

### Post-fit audit + band-utilization view
After a fit: reports coefficients saturated at bounds, flat
(unconstrained) directions, and a band-utilization view (per-reaction
R# labels) of how much of each reaction's uncertainty band the
optimizer used.

## 2026-07 — Objective rebuild & GUI

### Residual objective + optimizer stage handoff rebuilt
Reworked the per-shock Bessel-weighted RMSE with per-point adaptive
Barron weights, legacy adaptive aggregation, solved per-shock time
shifts, and the sigma-total error floor; the global->local handoff
re-anchors the error floor at the stage optimum.

### Bayesian/CheKiPEUQ objective path removed
Cut the Bayesian optimization path; the objective is Residual-only.
- Why: the relative-% normalization was suspect and a native
  soft-bounds prior in the rebuilt residual objective covered the use
  case; per the B3 decision.

### Optimization-outcome views + sim overlay
Added optimization-outcome plots and a start/best/current simulation
overlay on the signal plot; live Home/autoscale behavior on nav.

## Bug fixes (prod)
- Deep-copy coefficients on mechanism reset so writes cannot corrupt
  the pristine snapshot (fixed a reset-aliasing product bug).
- Keep optimization-view Home live and re-autoscale on nav restores;
  keep GUI-run mechanism artifacts out of the example library.
- Bit-exact pool-vs-serial objective parity, pinned by a regression
  test (an earlier ~5.6% discrepancy did not reproduce and was a
  cross-configuration measurement artifact).

## Pending / not yet in prod
- Quick-filter triage mode (fixed-budget fidelity ladder): built and
  DELETED per its registered rule — the wall gate (<=0.25x) is
  arithmetically unreachable while both modes share the coarse stage
  (measured 0.52x), and the rank gates went unmeasured because scoring
  saved mechanisms on a common anchor is currently impossible (the
  evaluate path anchors at the loaded mechanism; s-reconstruction is
  refit-invalid). Coarse-only triage (~0.1x wall) already exists by
  disabling the local stage. A refit-free common-anchor mechanism
  scorer is the recorded prerequisite for future comparison tooling.

- Multi-fidelity ladder local stage (ODE-tolerance + D-optimal-subset
  escalation): built, failed its registered validation twice (large
  wall savings but 3-8% accuracy loss; the noise-floor stall detector
  is structurally premature on this objective — the step-scale
  roughness it gates on is exactly the noise Subplex descends
  through), and DELETED per the pre-registered rule. Detail in
  `development/revamp_pt_2/multifidelity_plan.md`.

- Direct transformed-Troe local optimization (bypassing the per-eval
  refit): investigated and REJECTED under pre-registered gates. The
  refit's step-scale noise was measured at 2-4 orders above the
  refit-free objective, but removing it lost to the field-basis local
  on every seed at equal evaluations (median 0.0201 vs 0.0189), and a
  legs-only ablation exonerated the Troe-quadruple geometry — the
  refit's per-eval projection onto the Troe manifold is load-bearing.
  An equal-walltime extension (3x evaluations, matching the refit-free
  variant's cheaper evals) also failed: converged, not budget-limited.
  Kept findings: the noise measurement, and that production optima
  carry Troe parameters far outside physicality guardrails (A_fc=-5,
  T=1e+/-30) — campaign-level guardrail enforcement is a roadmap
  candidate. Detail in `development/revamp_pt_2/b8_plan.md`.

- Racing (first-k global-stage pruning): DELETED per the registered
  delete-on-fail rule (per-eval Troe-refit walltime floor made the
  probe savings a no-op). Revisit only if the local-stage
  parameterization work removes that floor.
- Standalone multistart driver: DELETED — Smurf's built-in multistart
  (Sobol starts, budget-shared) subsumes it and nothing imported it.
  The D-optimal subsetting engine code stays; its re-validation feeds
  the multi-fidelity ladder bucket.
- Smurf default-flip: DONE at 16 starts (see the Smurf entry above).
  The 8-start config had failed the median criterion with a 2.8x-wider
  spread; 16 starts (cheap after the pool-parallel sweep) flipped the
  distribution — median beats RBFOpt by 4.7%, fat tail eliminated,
  0.18x global walltime.
- Field-parameterized local stage: PROMOTED TO DEFAULT (see the entry
  above). The direct experiment overturned the earlier "DOF-capped /
  local stage structurally necessary" claim, and the same investigation
  established that the scaler->mechanism Troe-refit map is severely
  ill-conditioned near the optimum (roundtrip 0.0202->0.0795) — the
  refit-degenerate directions the field subspace excludes are why
  model-based DFO fails here and why the field wins. Detail in
  `development/revamp_pt_2/smurf_plan.md`.
