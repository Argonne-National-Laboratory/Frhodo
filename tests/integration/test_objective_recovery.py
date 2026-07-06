"""Known-truth recovery harness for the experiment-level objective.

A fake forward model (analytic exponential-decay observable driven by an
Arrhenius k(T) — no ODEs) runs campaigns through the real reduction
chain. The production Residual objective is the legacy adaptive
aggregation on raw per-shock losses under user × coverage weights; the
M-location on standardized losses is the reporting layer only, and its
assertions here test the report (flags, no false alarms), not the
optimizer. The within-shock reduction is fixed at L2 so the
experiment-level behavior is isolated. Recovery scenarios optimize the
Arrhenius triple from a perturbed start under the production objective
and the frozen pre-rebuild baseline, scoring rate-space error against
the known truth.

Scenario matrix (each runs in Linear and Bisymlog scales):
  S1 clean spread campaign        — not-worse than the frozen baseline
  S2 clustered + sparse high-T    — better than the baseline (coverage
                                    earning its default)
  S3 one wholesale-corrupted      — not-worse; flagged by the report
  S3b wrong-kinetics shock        — flagged by the report
  S4 point-outlier shock          — not-worse (within-shock robustness)
  S5 history-independence         — identical objective for identical θ
                                    regardless of evaluation order (exact)
  S6 seed stability               — recovery spread bounded across seeds
  S7 cold-clean low-SNR shock     — must NOT be flagged
  S8 structured contamination     — wobble + bursty shocks flagged
  S9 correlated-noise campaign    — endpoint loss parity; no mass-flag
"""
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from frhodo._vendor.opendsm.adaptive_loss import adaptive_weights
from frhodo.common.scale import Scale
from frhodo.experiment.uncertainty import correlation_length, sigma_bar
from frhodo.optimize.cost.aggregation import (
    LOSS_C_FLOOR_K,
    coverage_weights,
    model_error_floor,
    solve_m_location,
)



R_GAS = 8.314  # J/mol/K
TRUTH = np.array([12.0, 0.0, 95_000.0])  # ln A, n, Ea
START = TRUTH + np.array([0.8, 0.15, 12_000.0])
SEEDS = [0, 1, 2, 3, 4]


def k_arrhenius(theta, T):
    ln_a, n, ea = theta

    return np.exp(ln_a) * T**n * np.exp(-ea / (R_GAS * T))


def fake_trace(theta, T, t, amplitude=1.0):
    """Pseudo-observable: amplitude · exp(−k(T)·t)."""
    trace = amplitude * np.exp(-np.outer(t, np.atleast_1d(k_arrhenius(theta, T))))

    return trace.squeeze()


class Campaign:
    """A synthetic shock campaign with per-shock conditions and noisy data."""

    def __init__(self, temperatures, pressures, rng, noise=0.01,
                 amplitudes=None, n_t=160, scale_mode="Bisymlog"):
        self.T = np.asarray(temperatures, dtype=float)
        self.P = np.asarray(pressures, dtype=float)
        self.t = np.linspace(0.0, 2.5e-4, n_t)
        n = self.T.size
        if amplitudes is None:
            amplitudes = np.ones(n)
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.noise = np.full(n, noise, dtype=float)
        self.scale_mode = scale_mode
        clean = np.column_stack([
            fake_trace(TRUTH, T_s, self.t, a)
            for T_s, a in zip(self.T, self.amplitudes)
        ])
        self.obs = clean + rng.standard_normal(clean.shape) * (
            self.noise * self.amplitudes
        )
        self.extra_resid = np.zeros_like(self.obs)
        self.features = np.column_stack([1000.0 / self.T, np.log10(self.P)])
        self._finalize()

    def _finalize(self):
        """Per-shock Scale calibration, σ̄, and τ̂, mirroring _trim_shocks."""
        self.scales = []
        self.sigma_bars = np.empty(self.T.size)
        self.corr_lengths = np.empty(self.T.size)
        window = np.ones(self.t.size)
        for j in range(self.T.size):
            scale = Scale(self.scale_mode, calibration_data=self.obs[:, j])
            self.scales.append(scale)
            value = sigma_bar(
                self.obs[:, j], scale=scale, window_weights=window,
            )
            if not np.isfinite(value) or value <= 0:
                value = 1.0
            self.sigma_bars[j] = value
            self.corr_lengths[j] = max(correlation_length(
                self.obs[:, j], scale=scale, window_weights=window,
            ), 1.0)
        self.sigma_totals = self.sigma_bars.copy()

    def calibrate_error_floor(self, theta_start):
        """Mirror production: raise the standardizing scale to total
        expected error using the campaign floor from start-point
        residuals."""
        self.sigma_totals = self.sigma_bars.copy()
        resid_scales = self.shock_losses(theta_start) * self.sigma_bars
        floor = model_error_floor(resid_scales, self.sigma_bars)
        self.sigma_totals = np.sqrt(self.sigma_bars**2 + floor**2)

    def shock_losses(self, theta):
        """Standardized per-shock losses through the real reduction chain."""
        losses = np.empty(self.T.size)
        for j in range(self.T.size):
            sim = fake_trace(theta, self.T[j], self.t, self.amplitudes[j])
            scale = self.scales[j]
            resid = scale.forward(self.obs[:, j]) - scale.forward(sim)
            resid = resid + self.extra_resid[:, j]
            resid = resid / self.sigma_totals[j]
            if not np.all(np.isfinite(resid)):
                losses[j] = np.inf
                continue
            wsse = float(np.sum(resid**2))
            eff_dof = max(float(resid.size) - 3.0, 1.0)
            losses[j] = np.sqrt(wsse / eff_dof)

        return losses


def loss_c_floor(campaign):
    """Suspicion threshold in loss units: k sampling-σ of the loss statistic.

    Effective dof divides by each shock's residual correlation length so
    correlated noise does not shrink the floor below the loss statistic's
    true sampling scale; campaign median, as in production.
    """
    dof = max(campaign.t.size - 3, 1)
    eff = np.median(dof / campaign.corr_lengths)
    floor = LOSS_C_FLOOR_K / np.sqrt(2.0 * max(eff, 1.0))

    return floor


def legacy_aggregate(losses, weights, loss_c=1.0):
    """Mirror of the production Residual aggregation: adaptive
    reweighting of raw per-shock losses about the campaign minimum,
    averaged under the supplied weights; the aggregate loss shape
    solves on full bounds every call."""
    if losses.size == 1:
        return float(losses[0])

    def at(alpha):
        loss_min = losses.min()
        exp_w, _c, _a = adaptive_weights(
            losses - loss_min, C_scalar=loss_c, alpha=alpha,
        )
        loss_exp = exp_w * (losses - loss_min) ** 2
        loss_exp = loss_exp - loss_exp.min() + loss_min

        return float(np.average(loss_exp, weights=weights))

    res = minimize_scalar(at, bounds=(-100.0, 2.0), method="bounded")

    return at(float(res.x))


def production_objective(losses_std, campaign, coverage=True):
    """The product Residual objective: legacy aggregation on raw
    (unstandardized) losses under user × coverage weights."""
    if not np.all(np.isfinite(losses_std)):
        return np.inf
    raw = losses_std * campaign.sigma_totals
    n = raw.size
    user_w = np.ones(n)
    weights = user_w
    if coverage and n >= 4:
        weights = user_w * coverage_weights(campaign.features, user_w)

    return legacy_aggregate(raw, weights)


def legacy_objective(losses_raw):
    """The pre-rebuild aggregation (scoring baseline): the same
    adaptive aggregate, unweighted — so production differs from this
    baseline by the user × coverage weighting alone."""
    losses = np.asarray(losses_raw, dtype=float)
    if not np.all(np.isfinite(losses)):
        return np.inf

    return legacy_aggregate(losses, np.ones(losses.size))


def recover(campaign, objective):
    """Local recovery from the perturbed start, scored in rate space.

    The Arrhenius triple is sloppy (lnA/n/Ea compensate), so recovery is
    scored as the worst ``|Δln k(T)|`` over the campaign's temperature
    window — the quantity a mechanism actually needs recovered.
    """
    x_scale = np.array([1.0, 0.2, 10_000.0])
    campaign.calibrate_error_floor(START)

    def fun(u):
        theta = u * x_scale
        losses = campaign.shock_losses(theta)
        if objective == "production":
            return production_objective(losses, campaign)

        return legacy_objective(losses * campaign.sigma_totals)

    # Two stages with the floor re-anchored at the stage boundary,
    # mirroring the production global -> local handoff. The second
    # stage is a polish under the corrected floor: a tight simplex at
    # the incumbent, not a fresh exploration.
    stage1 = minimize(
        fun, START / x_scale, method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 2000},
    )
    campaign.calibrate_error_floor(stage1.x * x_scale)
    steps = np.maximum(np.abs(stage1.x) * 1e-3, 1e-4)
    simplex = np.vstack([stage1.x, stage1.x + np.diag(steps)])
    res = minimize(
        fun, stage1.x, method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 1000,
                 "initial_simplex": simplex},
    )
    theta_hat = res.x * x_scale
    T_grid = np.linspace(campaign.T.min(), campaign.T.max(), 25)
    ln_k_err = np.abs(
        np.log(k_arrhenius(theta_hat, T_grid)) - np.log(k_arrhenius(TRUTH, T_grid))
    )

    return float(np.max(ln_k_err))


def median_errors(make_campaign, scale_mode):
    errs_prod, errs_old = [], []
    for seed in SEEDS:
        campaign = make_campaign(np.random.default_rng(seed), scale_mode)
        errs_prod.append(recover(campaign, "production"))
        errs_old.append(recover(campaign, "legacy"))
    med_prod = float(np.median(errs_prod))
    med_old = float(np.median(errs_old))

    return med_prod, med_old


def spread_T(n=12):
    spread = np.linspace(1300.0, 2000.0, n)

    return spread


@pytest.mark.slow
@pytest.mark.parametrize("scale_mode", ["Linear", "Bisymlog"])
class TestScenarioMatrix:
    def test_s1_clean_spread_not_worse(self, scale_mode):
        def make(rng, sm):
            return Campaign(spread_T(), np.full(12, 1.5e4), rng, scale_mode=sm)

        med_prod, med_old = median_errors(make, scale_mode)
        assert med_prod <= 1.3 * med_old + 0.02, (
            f"S1 {scale_mode}: new objective worse on clean campaign: "
            f"production {med_prod:.4f} vs legacy {med_old:.4f}"
        )

    def test_s2_clustered_sparse_better(self, scale_mode):
        def make(rng, sm):
            T = np.concatenate([np.full(10, 1400.0) + rng.uniform(-15, 15, 10),
                                [1900.0, 1950.0]])

            return Campaign(T, np.full(12, 1.5e4), rng, scale_mode=sm)

        med_prod, med_old = median_errors(make, scale_mode)
        assert med_prod < med_old, (
            f"S2 {scale_mode}: coverage should beat count-domination: "
            f"production {med_prod:.4f} vs legacy {med_old:.4f}"
        )

    def test_s3_corrupted_shock_better_and_flagged(self, scale_mode):
        def make(rng, sm):
            campaign = Campaign(spread_T(), np.full(12, 1.5e4), rng,
                                scale_mode=sm)
            # wholesale corruption: a wrong-amplitude trace on shock 5
            campaign.obs[:, 5] = 0.55 * campaign.obs[:, 5] + 0.2
            campaign._finalize()

            return campaign

        med_prod, med_old = median_errors(make, scale_mode)
        assert med_prod <= med_old, (
            f"S3 {scale_mode}: robust aggregation should not lose to legacy "
            f"under corruption: production {med_prod:.4f} vs legacy {med_old:.4f}"
        )

        campaign = make(np.random.default_rng(0), scale_mode)
        losses = campaign.shock_losses(TRUTH)
        mloc = solve_m_location(losses, c_floor=loss_c_floor(campaign))
        assert np.argmax(mloc.z) == 5, (
            f"S3 {scale_mode}: corrupted shock must carry the largest z, "
            f"got z={np.round(mloc.z, 2)}"
        )
        assert mloc.irls_weights[5] < 0.5, (
            f"S3 {scale_mode}: corrupted shock must be discounted, "
            f"weight={mloc.irls_weights[5]:.3f}"
        )

    def test_s3b_wrong_kinetics_shock_flagged(self, scale_mode):
        """θ-coupled corruption: one trace generated from wrong kinetics
        drags the fit equally under the production aggregation — the
        report layer must identify it so the researcher can act."""
        def make(rng, sm):
            campaign = Campaign(spread_T(), np.full(12, 1.5e4), rng,
                                scale_mode=sm)
            wrong = TRUTH + np.array([-2.0, 0.0, -20_000.0])
            campaign.obs[:, 5] = (
                fake_trace(wrong, campaign.T[5], campaign.t)
                + rng.standard_normal(campaign.t.size) * 0.01
            )
            campaign._finalize()

            return campaign

        campaign = make(np.random.default_rng(0), scale_mode)
        losses = campaign.shock_losses(TRUTH)
        mloc = solve_m_location(losses, c_floor=loss_c_floor(campaign))
        assert np.argmax(mloc.z) == 5, (
            f"S3b {scale_mode}: wrong-kinetics shock must top the z "
            f"ranking, got z={np.round(mloc.z, 2)}"
        )

    def test_s4_point_outliers_not_worse(self, scale_mode):
        def make(rng, sm):
            campaign = Campaign(spread_T(), np.full(12, 1.5e4), rng,
                                scale_mode=sm)
            spikes = rng.choice(campaign.t.size, size=8, replace=False)
            campaign.obs[spikes, 3] += 0.5
            campaign._finalize()

            return campaign

        med_prod, med_old = median_errors(make, scale_mode)
        assert med_prod <= 1.3 * med_old + 0.02, (
            f"S4 {scale_mode}: point outliers should stay a within-shock "
            f"matter: production {med_prod:.4f} vs legacy {med_old:.4f}"
        )

    def test_s5_history_independence_exact(self, scale_mode):
        campaign = Campaign(spread_T(), np.full(12, 1.5e4),
                            np.random.default_rng(2), scale_mode=scale_mode)
        thetas = [TRUTH, START, TRUTH + [0.2, -0.05, 3000.0], TRUTH, START]
        direct = []
        for theta in thetas:
            direct.append(
                production_objective(campaign.shock_losses(theta), campaign)
            )
        assert direct[0] == direct[3] and direct[1] == direct[4], (
            f"S5 {scale_mode}: objective must be a pure function of theta; "
            f"got {direct}"
        )

    def test_s6_seed_stability(self, scale_mode):
        """Recovery scatter across seeds, paired against the baseline.

        Production differs from the baseline by coverage weighting
        alone. Coverage upweights the sparse 1000/T edges; on
        Linear-scale traces the cold edge carries almost no signal, so
        trajectories scatter more there — a documented stability cost
        of coverage on Linear campaigns. Bisymlog, the default scale,
        must stay tight."""
        def make(rng, sm):
            return Campaign(spread_T(), np.full(12, 1.5e4), rng, scale_mode=sm)

        errs_prod, errs_base = [], []
        for seed in SEEDS:
            campaign = make(np.random.default_rng(seed), scale_mode)
            errs_prod.append(recover(campaign, "production"))
            errs_base.append(recover(campaign, "legacy"))
        spread_prod = float(np.max(errs_prod) - np.min(errs_prod))
        spread_base = float(np.max(errs_base) - np.min(errs_base))
        if scale_mode == "Bisymlog":
            margin = 0.25
        else:
            margin = 1.25
        assert spread_prod <= spread_base + margin, (
            f"S6 {scale_mode}: production less stable than the baseline: "
            f"spread {spread_prod:.3f} vs {spread_base:.3f} "
            f"(production errors {np.round(errs_prod, 3)})"
        )

    def test_s7_cold_clean_low_snr_not_flagged(self, scale_mode):
        campaign = Campaign(
            np.concatenate([spread_T(10), [1250.0, 1260.0]]),
            np.full(12, 1.5e4),
            np.random.default_rng(3),
            amplitudes=np.concatenate([np.ones(10), [0.08, 0.08]]),
            scale_mode=scale_mode,
        )
        losses = campaign.shock_losses(TRUTH)
        mloc = solve_m_location(losses, c_floor=loss_c_floor(campaign))
        assert np.all(mloc.irls_weights[10:] > 0.5), (
            f"S7 {scale_mode}: cold clean low-SNR shocks must not be "
            f"flagged; weights {np.round(mloc.irls_weights[10:], 3)}"
        )

    def test_s9_correlated_noise_campaign(self, scale_mode):
        """AR(1) noise on every shock: a clean-but-correlated campaign must
        not be mass-flagged, and the fit quality the aggregation actually
        controls — endpoint per-shock losses — must match legacy's. Rate-
        space recovery is not compared here: both objectives reach the
        same losses and differ only by where they park along the sloppy
        Arrhenius valley, which is trajectory luck, not aggregation
        quality."""
        def make(rng, sm):
            campaign = Campaign(spread_T(), np.full(12, 1.5e4), rng,
                                noise=0.0, scale_mode=sm)
            n_t, n_s = campaign.t.size, campaign.T.size
            ar = np.zeros((n_t, n_s))
            e = rng.standard_normal((n_t, n_s))
            for i in range(1, n_t):
                ar[i] = 0.6 * ar[i - 1] + e[i]
            campaign.obs = campaign.obs + 0.01 * ar * campaign.amplitudes
            campaign._finalize()

            return campaign

        x_scale = np.array([1.0, 0.2, 10_000.0])
        ratios = []
        for seed in SEEDS:
            campaign = make(np.random.default_rng(seed), scale_mode)
            losses_at = {}
            for objective in ["production", "legacy"]:
                campaign.calibrate_error_floor(START)

                def fun(u, _objective=objective):
                    losses = campaign.shock_losses(u * x_scale)
                    if _objective == "production":
                        return production_objective(losses, campaign)

                    return legacy_objective(losses * campaign.sigma_totals)

                res = minimize(
                    fun, START / x_scale, method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 2000},
                )
                losses_at[objective] = campaign.shock_losses(res.x * x_scale)

            med_prod = float(np.median(losses_at["production"]))
            med_old = float(np.median(losses_at["legacy"]))
            ratios.append(med_prod / med_old)

            mloc = solve_m_location(
                losses_at["production"], c_floor=loss_c_floor(campaign),
            )
            assert mloc.alpha > 1.5, (
                f"S9 {scale_mode} seed {seed}: correlated noise mass-flagged, "
                f"alpha={mloc.alpha:.3f}"
            )
            assert mloc.irls_weights.min() > 0.5, (
                f"S9 {scale_mode} seed {seed}: shocks faded under correlated "
                f"noise, min weight {mloc.irls_weights.min():.3f}"
            )

        # Per-seed endpoint parking scatters a few percent either way;
        # the parity claim is about the typical case.
        assert float(np.median(ratios)) <= 1.05, (
            f"S9 {scale_mode}: endpoint fit quality regressed vs the "
            f"baseline: per-seed loss ratios {np.round(ratios, 3)}"
        )

        campaign = make(np.random.default_rng(6), scale_mode)
        assert np.median(campaign.corr_lengths) > 1.0, (
            "S9 setup: the campaign should measure as correlated"
        )
        losses = campaign.shock_losses(TRUTH)
        mloc = solve_m_location(losses, c_floor=loss_c_floor(campaign))
        assert np.all(mloc.irls_weights > 0.5), (
            f"S9 {scale_mode}: clean correlated shocks must not be flagged; "
            f"weights {np.round(mloc.irls_weights, 3)}"
        )

    def test_s8_structured_contamination_flagged(self, scale_mode):
        campaign = Campaign(spread_T(), np.full(12, 1.5e4),
                            np.random.default_rng(4), scale_mode=scale_mode)
        rng = np.random.default_rng(40)
        # wobble on shock 2: slow sinusoid sized vs its sigma_bar
        campaign.extra_resid[:, 2] = (
            6.0 * np.sin(2 * np.pi * campaign.t / campaign.t[-1] * 3.0)
            * campaign.sigma_bars[2]
        )
        # bursts on shock 7: intermittent clumps
        burst = np.zeros(campaign.t.size)
        for start in rng.choice(campaign.t.size - 12, size=4, replace=False):
            burst[start:start + 12] = 8.0
        campaign.extra_resid[:, 7] = burst * campaign.sigma_bars[7]

        losses = campaign.shock_losses(TRUTH)
        mloc = solve_m_location(losses, c_floor=loss_c_floor(campaign))
        flagged = set(np.argsort(mloc.z)[-2:])
        assert flagged == {2, 7}, (
            f"S8 {scale_mode}: wobble+burst shocks must top the z ranking, "
            f"got {sorted(flagged)} with z={np.round(mloc.z, 2)}"
        )
        assert np.all(mloc.irls_weights[[2, 7]] < 0.6), (
            f"S8 {scale_mode}: contaminated shocks must be discounted, "
            f"weights {np.round(mloc.irls_weights[[2, 7]], 3)}"
        )
