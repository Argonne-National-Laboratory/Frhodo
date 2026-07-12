# The Residual objective, written down

What the optimizer minimizes when Objective Function Type = Residual,
as of the B1 closeout. One run = one mechanism proposal; everything
below is a pure function of the trial parameters x.

## Within one experiment (shock s)

1. Simulate the observable at the trial x; interpolate to the
   experiment's time base, allowing a per-shock time shift τ_s solved
   within ±t_unc (t=0 is relaxation-blurred; the solve is
   deterministic).
2. Residual in the configured scale (Bisymlog by default, calibrated
   per shock).
3. Divide by the shock's standardizing scale σ_total,s =
   √(σ̄_s² + E²): σ̄_s is the measurement-noise scale estimated once
   from the data over the weight window; E is a campaign model-error
   floor estimated from a probe at the optimizer start (25th
   percentile of the excess of achieved residual scale over σ̄) and
   re-anchored at the global→local stage boundary, where near-optimal
   residuals give it its intended meaning.
4. Within-shock adaptive loss (Barron, user setting; Adaptive by
   default) reweights points; the per-shock loss l_s is the
   Bessel-corrected weighted RMSE of the standardized residual. The
   division by σ_total cancels in step 5's multiplication, so the
   objective consumes raw residual scales; standardization exists for
   the report layer and cross-shock diagnostics.

## Across experiments

5. Raw per-shock losses L_s = l_s · σ_total,s.
6. Adaptive experiment-level reweighting about the campaign minimum
   (the pre-rebuild aggregation): excursions (L_s − L_min)² weighted
   by the Barron adaptive loss whose shape solves on full bounds every
   evaluation (stationary — no history narrowing).
7. The objective is the weighted average of the reweighted losses
   under user × coverage weights. Coverage weights are the inverse
   Gaussian-KDE density of campaign conditions in (1000/T, log10 P),
   clipped to [0.2, 5]× mean — representation in condition space, not
   shot count, sets each experiment's voice. CostSettings knob,
   default ON (GUI: Loss Function tab checkbox).

## The report layer (never feeds the optimizer)

Per shock, the σ-standardized losses go through a robust M-location
solve (asymmetric Barron, α ∈ [1, 2], suspicion-threshold floor on its
scale): the resulting z and IRLS weights flag experiments that misfit
beyond their noise. Validated against expert noise labels on the real
campaign. Shown in the optimization views and the progress payload;
acting on a flag is the researcher's decision.

## Why the M-location is not the objective

The B1 comparison protocol (comparison_protocol.md,
protocol_results.md) found its truth-recovery win was an artifact —
accidental trimming compensating for a broken stage handoff — and
under a fixed engine it lost to this aggregation on every
pre-registered test. Parked; the report layer is what survived.
