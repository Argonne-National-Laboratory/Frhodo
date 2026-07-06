"""Experiment-level aggregation: the weighted asymmetric Barron M-location.

The optimizer's scalar objective over a campaign is the M-location μ* of
the standardized per-shock losses — the robust center the losses pull on,
quadratic below μ* and Barron-robust above it, with the tail shape α
estimated jointly via the penalized NLL (the ln Z construction).

α is restricted to [1, 2]: the influence function ψ = ρ′ is monotone
non-decreasing there, which makes the M-location monotone in every sample
(worsening any shock can never lower the objective). Below α = 1, ψ
redescends and that guarantee fails.

Coverage weights live here too: inverse kernel-density of shock conditions
in standardized feature space, so condition-space representation — not
shot count — sets each shock's voice.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from frhodo._vendor.opendsm.adaptive_loss_Z import ln_Z



ALPHA_MIN = 1.0
ALPHA_MAX = 2.0
_ALPHA_XATOL = 1e-4
_QUADRATIC_EPS = 1e-9  # treat alpha within this of 2 as exactly L2
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_MIN_SAMPLES_FOR_ADAPTIVE = 4
_MAD_TO_SIGMA = 1.4826
LOSS_C_FLOOR_K = 5.0  # suspicion threshold: sampling-sigmas of loss above median
MODEL_ERROR_FLOOR_QUANTILE = 0.25  # excess quantile carried by the best-fit shocks


def ln_z_asym(alpha: float) -> float:
    """Log normalizer of the asymmetric density exp(-rho_asym).

    The lower half integrates to sqrt(2π)/2 (quadratic side), the upper
    half to Z(α)/2, so Z_asym = (sqrt(2π) + Z(α)) / 2.
    """
    z_alpha = float(np.exp(ln_Z(alpha)))
    result = float(np.log(0.5 * (_SQRT_2PI + z_alpha)))

    return result


def rho_asym(z: np.ndarray, alpha: float) -> np.ndarray:
    """Asymmetric Barron loss: quadratic for z ≤ 0, robust ρ(z; α) for z > 0."""
    z = np.asarray(z, dtype=float)
    out = 0.5 * z * z
    if alpha < ALPHA_MAX - _QUADRATIC_EPS:
        pos = z > 0
        b = abs(alpha - 2.0)
        zp2 = z[pos] * z[pos]
        out[pos] = b / alpha * ((zp2 / b + 1.0) ** (alpha / 2.0) - 1.0)

    return out


def psi_asym(z: np.ndarray, alpha: float) -> np.ndarray:
    """dρ_asym/dz — the influence function. Monotone for α in [1, 2]."""
    z = np.asarray(z, dtype=float)
    out = z.copy()
    if alpha < ALPHA_MAX - _QUADRATIC_EPS:
        pos = z > 0
        b = abs(alpha - 2.0)
        zp = z[pos]
        out[pos] = zp * (zp * zp / b + 1.0) ** (alpha / 2.0 - 1.0)

    return out


def irls_weights_asym(z: np.ndarray, alpha: float) -> np.ndarray:
    """Implied per-sample weights ψ(z)/z: 1 on the good side, ≤ 1 above.

    The reporting layer's R_s — never part of the objective itself.
    """
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    if alpha < ALPHA_MAX - _QUADRATIC_EPS:
        pos = z > 0
        b = abs(alpha - 2.0)
        zp = z[pos]
        out[pos] = (zp * zp / b + 1.0) ** (alpha / 2.0 - 1.0)

    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / np.sum(w)

    return float(v[np.searchsorted(cdf, q)])


def _penalized_nll(z: np.ndarray, w_norm: np.ndarray, alpha: float) -> float:
    result = float(np.sum(w_norm * rho_asym(z, alpha)) + ln_z_asym(alpha))

    return result


def _mu_root(losses: np.ndarray, weights: np.ndarray, c: float, alpha: float) -> float:
    """Unique root of Σ W·ψ((l−μ)/c) = 0; exists since ψ is monotone and odd-signed."""
    def pull(mu):
        return float(np.sum(weights * psi_asym((losses - mu) / c, alpha)))

    lo = float(losses.min()) - c
    hi = float(losses.max()) + c
    result = float(brentq(pull, lo, hi, xtol=1e-12 * max(c, 1.0)))

    return result


@dataclass(frozen=True)
class MLocation:
    """Result of the experiment-level aggregation at one parameter point.

    Attributes:
        mu: The objective value — robust location of the per-shock losses.
        alpha: Tail shape used ([1, 2]; 2 when fixed or degenerate).
        c: Scale (weighted MAD of the losses).
        z: Standardized deviations (l − mu) / c per shock.
        irls_weights: Implied per-shock weights for the reporting layer.
    """
    mu: float
    alpha: float
    c: float
    z: np.ndarray
    irls_weights: np.ndarray


def solve_m_location(
    losses: np.ndarray,
    weights: np.ndarray | None = None,
    alpha: float | None = None,
    c_floor: float = 0.0,
) -> MLocation:
    """Weighted asymmetric Barron M-location of per-shock losses.

    Args:
        losses: Finite standardized per-shock losses (callers apply the
            non-finite → inf-objective guard before calling).
        weights: Per-shock W_s = user × coverage; defaults to uniform.
        alpha: Fix the tail shape, or None for adaptive in [1, 2] via the
            penalized NLL (needs ≥ 4 shocks; fewer fall back to α = 2).
        c_floor: Lower bound on the scale c. Floor it at the sampling
            scale of the loss statistic (≈ 1/√(2·dof) for standardized
            losses) so ordinary fluctuations never trigger the robust
            saturation; genuine contamination still towers over it.

    Returns:
        MLocation. ``mu`` is the optimizer's scalar objective.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    if not np.all(np.isfinite(losses)):
        raise ValueError("solve_m_location requires finite losses")
    n = losses.size
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.shape != losses.shape:
        raise ValueError("weights must match losses in length")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")

    if n == 1:
        single = MLocation(
            mu=float(losses[0]), alpha=ALPHA_MAX, c=1.0,
            z=np.zeros(1), irls_weights=np.ones(1),
        )

        return single

    med = _weighted_quantile(losses, weights, 0.5)
    mad = _weighted_quantile(np.abs(losses - med), weights, 0.5)
    c = max(_MAD_TO_SIGMA * mad, float(c_floor))
    if c <= 0:
        degenerate = MLocation(
            mu=med, alpha=ALPHA_MAX, c=0.0,
            z=np.zeros(n), irls_weights=np.ones(n),
        )

        return degenerate

    w_norm = weights / weights.sum()

    if alpha is not None:
        alpha_used = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    elif n < _MIN_SAMPLES_FOR_ADAPTIVE:
        alpha_used = ALPHA_MAX
    else:
        def nll_at(a):
            mu_a = _mu_root(losses, weights, c, a)

            return _penalized_nll((losses - mu_a) / c, w_norm, a)

        res = minimize_scalar(
            nll_at, bounds=(ALPHA_MIN, ALPHA_MAX), method="bounded",
            options={"xatol": _ALPHA_XATOL},
        )
        alpha_used = float(res.x)

    mu = _mu_root(losses, weights, c, alpha_used)
    z = (losses - mu) / c
    r_weights = irls_weights_asym(z, alpha_used)

    return MLocation(mu=mu, alpha=alpha_used, c=c, z=z, irls_weights=r_weights)


def model_error_floor(
    resid_scales: np.ndarray,
    sigma_bars: np.ndarray,
    quantile: float = MODEL_ERROR_FLOOR_QUANTILE,
) -> float:
    """Campaign-level model-error scale E from start-point residuals.

    Per shock the achieved residual scale decomposes as r² ≈ σ̄² + E_s²,
    where σ̄ is measurement noise and E_s collects model-form and
    numerical error. Standardizing by σ̄ alone over-amplifies the
    cleanest shocks (E_s dominates their tiny σ̄), so the standardizing
    scale needs the floor. A low quantile of the excess across the
    campaign estimates the floor shared by the best-fit shocks —
    high-excess shocks carry optimizable misfit and must not inflate it.
    Non-finite entries (overflowed start-point traces) are ignored.
    """
    r = np.asarray(resid_scales, dtype=float)
    s = np.asarray(sigma_bars, dtype=float)
    excess = r * r - s * s
    excess = excess[np.isfinite(excess)]
    if excess.size == 0:
        return 0.0
    floor_sq = max(float(np.quantile(np.maximum(excess, 0.0), quantile)), 0.0)

    return float(np.sqrt(floor_sq))


def coverage_weights(
    features: np.ndarray,
    weights: np.ndarray | None = None,
    clip: tuple[float, float] = (0.2, 5.0),
) -> np.ndarray:
    """Inverse-density coverage weights over shock conditions.

    Args:
        features: (n_shocks, n_dims) condition coordinates — e.g. columns
            (1000/T_reactor, log10 P_reactor). Zero-variance dims are
            dropped; if none survive, all weights are 1.
        weights: KDE sample weights (user × implied quality); a discounted
            shock neither votes nor counts as covering its region.
        clip: (lo, hi) multiples of the mean inverse density.

    Returns:
        Per-shock weights normalized to mean 1.
    """
    x = np.atleast_2d(np.asarray(features, dtype=float))
    n = x.shape[0]
    if n < _MIN_SAMPLES_FOR_ADAPTIVE:
        return np.ones(n)
    if weights is None:
        weights = np.ones(n)
    w = np.asarray(weights, dtype=float).ravel()
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("coverage weights must be non-negative with positive sum")

    mean = np.average(x, axis=0, weights=w)
    var = np.average((x - mean) ** 2, axis=0, weights=w)
    live = var > 0
    if not np.any(live):
        return np.ones(n)
    xs = (x[:, live] - mean[live]) / np.sqrt(var[live])
    d = xs.shape[1]

    n_eff = w.sum() ** 2 / np.sum(w * w)
    h = n_eff ** (-1.0 / (d + 4))  # Scott bandwidth on standardized features

    sq = np.sum((xs[:, None, :] - xs[None, :, :]) ** 2, axis=2)
    density = np.sum(w[None, :] * np.exp(-0.5 * sq / (h * h)), axis=1)

    inv = 1.0 / np.maximum(density, np.finfo(float).tiny)
    inv = np.clip(inv, clip[0] * inv.mean(), clip[1] * inv.mean())
    result = inv / inv.mean()

    return result
