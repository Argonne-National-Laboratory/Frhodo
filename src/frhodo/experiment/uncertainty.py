"""Data-driven measurement-noise estimate for experimental shock traces.

σ(t) returned by :func:`estimate_pointwise_sigma` is the local scatter of
the data about a smooth centerline, measured directly in the optimizer's
residual scale. Defining σ as the data's own spread makes the ±1.96σ band
an **envelope around the data by construction** — and that property is
invariant to the scale, since containment of the data is preserved by any
monotone transform. (A model-propagated σ, e.g. a delta-method derivative,
is *not* invariant: in AbsoluteLog the derivative ``1/(|y|·ln10)`` diverges
at a zero-crossing and inflates σ to a value the finite data never has.)

Pipeline (any scale):

1. Noise std ``σ_n`` from 4th-order Hall-Müller differences
   (:func:`_noise_sigma`) — cancels any locally-cubic trend, so a clean
   feature (even a steep one) reads as small noise in every scale.
2. Centerline ``μ`` = curvature-adaptive local-linear regression
   (:func:`smooth_centerline`): the smoothing bandwidth keys off the
   signal *curvature* ``f″`` via ``h ∝ (σ²/f″²)^(1/5)`` — small at sharp
   features (tracked with no corner-rounding/lag), wide in flat-noisy
   regions. Keying off curvature (not noise scatter) is what makes it
   work in *all* scales: a scatter-driven bandwidth widens at a steep
   drop (wrong way), a curvature-driven one narrows (right way).
3. ``σ`` = local RMS of ``r = z − μ`` over a window of ``_RMS_WINDOW_FRAC·n``
   samples (floored at ``σ_n``, Schoenberg-smoothed) — a *less-local*
   scatter that captures the correlated-noise excursion scale so the
   band envelopes the data, yet stays smooth and heteroscedastic.

Bounds for the optimizer's CheKiPEUQ likelihood center on each observed
point via :func:`bounds_from_sigma`.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter, percentile_filter, uniform_filter1d
from scipy.stats import binned_statistic

from frhodo.common.scale import Scale



_Q25_TO_SIGMA = 1.0 / 0.3186  # Q25(|X|)=0.3186σ for X~N(0,σ²) → σ̂ = Q25/0.3186
_MIN_N_FOR_ESTIMATE = 16
_MIN_LEPSKI_BANDWIDTH = 32
_LEPSKI_GROWTH = 1.5
_LEPSKI_CONFIDENCE = 2.5
_SIGMA_FLOOR_FRAC = 1.0e-3  # of global σ scale, keeps σ strictly positive
_HALL_MULLER_4_VAR = 70.0  # Var of (1, -4, 6, -4, 1) applied to iid σ-noise
_BANDWIDTH_BULK_FRAC = 0.05  # flat-region local-linear bandwidth = frac·n samples
_BANDWIDTH_MIN = 2.0
_CURVATURE_SMOOTH_FRAC = 0.02  # curvature-smoothing window = frac·n samples
_CURVATURE_ITERS = 2  # bandwidth refinement passes
_FAN_GIJBELS_EXP = 0.4  # h ∝ (σ²/f″²)^(1/5) → exponent 2/5 on σ and on f″
_RMS_WINDOW_FRAC = 0.1  # σ-estimation window = frac·n samples
_RMS_WINDOW_MIN = 11
_SIGMA_SMOOTH_BINS_MIN = 8  # PCHIP control-point bins for σ smoothing
_SIGMA_SMOOTH_BINS_MAX = 40
_SIGMA_SMOOTH_BIN_WIDTH = 25  # target samples per bin
_LOG_FAMILY_MODES = ("Log", "AbsoluteLog")  # forward() is singular at a zero crossing


def _lepski_bandwidths(n: int) -> list[int]:
    """Geometric ladder of candidate bandwidths between ``_MIN_LEPSKI_BANDWIDTH`` and ``n``.

    The floor stays absolute (not n-relative): a window needs a minimum
    sample count for a stable Q25/MAD spread estimate regardless of
    trace length. For short traces this trades locality for stability —
    the correct choice, since a smaller window there is too noisy.
    """
    h_min = min(_MIN_LEPSKI_BANDWIDTH, n)
    h_max = min(max(h_min, n // 4), n)
    bandwidths: list[int] = []
    h = h_min
    while h <= h_max:
        bandwidths.append(h)
        h = max(h + 1, int(h * _LEPSKI_GROWTH))
    if bandwidths[-1] != h_max:
        bandwidths.append(h_max)

    return bandwidths


def _lepski_local_sigma(values: np.ndarray, var_multiplier: float) -> np.ndarray:
    """Lepski-adaptive moving σ via the 25th-percentile of |residual|.

    ``values`` is the output of an order-k difference operator with
    iid-noise variance ``var_multiplier · σ²``. Spread is read as the
    25th percentile of ``|values − median|`` in the window: ignores
    the upper 75% of the distribution, so one-sided contamination from
    signal leakage at sharp transitions (where the difference operator
    doesn't fully cancel the underlying smooth signal) is discarded
    instead of biasing σ̂ upward. Per-point bandwidth picked by the
    Lepski rule: largest window consistent with all smaller ones.
    """
    n = values.size
    nan_result = np.full(n, np.nan)
    if n < _MIN_N_FOR_ESTIMATE:
        return nan_result
    bandwidths = _lepski_bandwidths(n)
    K = len(bandwidths)
    q25_to_sigma = _Q25_TO_SIGMA / np.sqrt(var_multiplier)
    sigmas = np.empty((K, n))
    for k, h in enumerate(bandwidths):
        med = median_filter(values, size=h, mode="reflect")
        abs_dev = np.abs(values - med)
        q25 = percentile_filter(abs_dev, percentile=25, size=h, mode="reflect")
        sigmas[k] = q25_to_sigma * q25
    se = sigmas / np.sqrt(np.asarray(bandwidths, dtype=float))[:, None]
    chosen = np.zeros(n, dtype=np.int64)
    for k in range(1, K):
        diff = np.abs(sigmas[k] - sigmas[:k])
        threshold = _LEPSKI_CONFIDENCE * (se[:k] + se[k])
        consistent = (diff <= threshold).all(axis=0)
        chosen = np.where(consistent & (chosen == k - 1), k, chosen)
    result = sigmas[chosen, np.arange(n)]
    global_q25 = float(np.percentile(np.abs(values - np.median(values)), 25))
    floor = q25_to_sigma * global_q25 * _SIGMA_FLOOR_FRAC
    sigma = np.maximum(result, floor)

    return sigma


def _smooth_sigma(values: np.ndarray) -> np.ndarray:
    """Shape-preserving smooth of a σ-trend via PCHIP through bin medians.

    Bins the values into ``~n/_SIGMA_SMOOTH_BIN_WIDTH`` groups, takes the
    median of each (robust), and PCHIP-interpolates through the bin
    centers. PCHIP is monotone/shape-preserving, so it cannot overshoot
    below zero the way a cubic smoothing spline does on rapidly-varying
    (multi-decade) input — σ stays strictly positive by construction.
    """
    n = values.size
    if n < _MIN_N_FOR_ESTIMATE:
        return values
    n_bins = int(np.clip(n // _SIGMA_SMOOTH_BIN_WIDTH,
                         _SIGMA_SMOOTH_BINS_MIN, _SIGMA_SMOOTH_BINS_MAX))
    x = np.linspace(0.0, 1.0, n)
    medians, edges, _ = binned_statistic(x, values, statistic="median", bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = np.isfinite(medians)
    if valid.sum() < 2:
        return values
    smoothed = np.asarray(PchipInterpolator(centers[valid], medians[valid])(x), dtype=float)

    return smoothed


def _noise_sigma(z: np.ndarray, finite: np.ndarray, n: int) -> np.ndarray:
    """Noise std via 4th-order Hall-Müller differences ``d⁴z``.

    ``d⁴z = z[i+4] − 4z[i+3] + 6z[i+2] − 4z[i+1] + z[i]`` cancels any
    locally-cubic trend (``Var = 70·σ²`` for iid noise), so a *clean*
    feature — even a steep one — reads as small noise in any scale. Used
    in the curvature-adaptive bandwidth formula and as a floor on σ.
    """
    src = np.where(finite, z, np.nanmedian(z[finite]))
    d4 = src[4:] - 4.0 * src[3:-1] + 6.0 * src[2:-2] - 4.0 * src[1:-3] + src[:-4]
    raw = _lepski_local_sigma(d4, _HALL_MULLER_4_VAR)
    smooth = np.maximum(_smooth_sigma(raw), float(np.min(raw)))
    padded = np.concatenate([smooth[:2], smooth, smooth[-2:]])

    return padded


def _local_linear(z: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Local-linear regression with per-point Gaussian bandwidth ``h`` (samples).

    Local-*linear* (not local-mean) so a steep ramp is followed without
    lag where the bandwidth is small.
    """
    n = z.size
    idx = np.arange(n, dtype=float)
    out = np.empty(n)
    tiny = np.finfo(float).tiny
    for i in range(n):
        hi = h[i]
        half = int(np.ceil(3.0 * hi))
        lo, up = max(0, i - half), min(n, i + half + 1)
        d = idx[lo:up] - i
        w = np.exp(-0.5 * (d / hi) ** 2)
        sw = w.sum()
        swd = (w * d).sum()
        swdd = (w * d * d).sum()
        swz = (w * z[lo:up]).sum()
        swdz = (w * d * z[lo:up]).sum()
        det = sw * swdd - swd * swd
        if abs(det) < tiny:
            out[i] = swz / sw
        else:
            out[i] = (swdd * swz - swd * swdz) / det

    return out


def _curvature_adaptive_centerline(
    z: np.ndarray, sigma: np.ndarray, finite: np.ndarray, n: int,
) -> np.ndarray:
    """Local-linear centerline with curvature-adaptive bandwidth (Fan-Gijbels).

    The smoothing bandwidth keys off the signal *curvature* ``f″`` rather
    than the noise scatter, so it shrinks at a sharp feature (tracking it,
    no corner-rounding/lag) and widens in flat-noisy regions — and it does
    so *scale-independently*, where a scatter-driven bandwidth pushes the
    wrong way at a steep drop. Per the local-linear MSE-optimal rule,
    ``h ∝ (σ² / f″²)^(1/5)``, auto-calibrated so flat regions (median
    curvature, median σ) get the bulk bandwidth. Two refinement passes so
    the curvature at a sharp feature isn't under-read by the initial
    fixed-bandwidth rounding.
    """
    zz = np.where(finite, z, np.nanmedian(z[finite]))
    h_bulk = max(_BANDWIDTH_MIN, _BANDWIDTH_BULK_FRAC * n)
    curv_window = max(5, int(_CURVATURE_SMOOTH_FRAC * n))
    sigma_ref = float(np.median(sigma)) + np.finfo(float).tiny
    mu = _local_linear(zz, np.full(n, h_bulk))
    for _ in range(_CURVATURE_ITERS):
        curv = np.abs(np.gradient(np.gradient(mu)))
        curv = uniform_filter1d(curv, size=curv_window, mode="reflect")
        curv_ref = float(np.median(curv)) + np.finfo(float).tiny
        h = (
            h_bulk
            * (sigma / sigma_ref) ** _FAN_GIJBELS_EXP
            * (curv_ref / np.maximum(curv, curv_ref)) ** _FAN_GIJBELS_EXP
        )
        h = np.clip(h, _BANDWIDTH_MIN, h_bulk)
        mu = _local_linear(zz, h)

    return mu


def _bounded_robust_band(
    z: np.ndarray, finite: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded centerline + σ for log-family scales (``log|y|``).

    A zero crossing sends ``log|y| → −∞``, so the curvature-adaptive
    method (which tries to *track* the spurious dip) collapses the band.
    Here the centerline is a smooth fixed-bandwidth local-linear fit that
    does not chase the dip, and σ is the **Q25-robust** local scatter of
    the residual: the extreme near-zero residuals fall in the upper 75%
    the 25th percentile discards, so they don't drag σ toward ∞. The band
    is therefore bounded and captures the typical scatter, with the rare
    near-zero-crossing points as honest outliers below it.
    """
    fill = float(np.nanmedian(z[finite]))
    zz = np.where(finite, z, fill)
    h_bulk = max(_BANDWIDTH_MIN, _BANDWIDTH_BULK_FRAC * n)
    center = _local_linear(zz, np.full(n, h_bulk))
    window = max(_RMS_WINDOW_MIN, int(n * _RMS_WINDOW_FRAC))
    abs_residual = np.abs(np.where(finite, z - center, 0.0))
    q25 = percentile_filter(abs_residual, percentile=25, size=window, mode="reflect")
    sigma_raw = _Q25_TO_SIGMA * q25
    sigma = np.maximum(_smooth_sigma(sigma_raw), float(np.min(sigma_raw)))

    return center, sigma


def smooth_centerline(y: np.ndarray, *, scale: Scale) -> np.ndarray:
    """Smooth centerline of ``forward(y)`` (scaled space).

    Curvature-adaptive local-linear for Linear/Bisymlog (tracks clean
    sharp features, smooths noisy regions); a bounded fixed-bandwidth fit
    for the log family, whose ``log|y|`` is singular at a zero crossing.
    Callers feed the result to ``scale.inverse`` to render in linear units.
    """
    arr = np.asarray(y, dtype=float).ravel()
    n = arr.size
    z = scale.forward(arr)
    if n < _MIN_N_FOR_ESTIMATE:
        return z
    finite = np.isfinite(z)
    if finite.sum() < _MIN_N_FOR_ESTIMATE:
        return z

    if scale.mode in _LOG_FAMILY_MODES:
        center, _ = _bounded_robust_band(z, finite, n)
    else:
        sigma = _noise_sigma(z, finite, n)
        center = _curvature_adaptive_centerline(z, sigma, finite, n)

    return center


def estimate_pointwise_sigma(y: np.ndarray, *, scale: Scale) -> np.ndarray:
    """Per-point measurement-noise σ(t) of ``y`` in the optimizer's scale.

    Linear/Bisymlog: local scatter about the curvature-adaptive
    centerline, over a window wide enough to capture the correlated-noise
    excursion scale (so ``μ ± 1.96σ`` envelopes the data) yet still local,
    heteroscedastic, and smooth. Log family: bounded Q25-robust scatter
    (see :func:`_bounded_robust_band`).

    Returns:
        ``np.ndarray`` shaped like ``y`` with ``σ ≥ 0`` (in scaled units).
    """
    arr = np.asarray(y, dtype=float).ravel()
    n = arr.size
    nan_result = np.full(n, np.nan)
    if n < _MIN_N_FOR_ESTIMATE:
        return nan_result
    z = scale.forward(arr)
    finite = np.isfinite(z)
    if finite.sum() < _MIN_N_FOR_ESTIMATE:
        return nan_result

    if scale.mode in _LOG_FAMILY_MODES:
        _, sigma = _bounded_robust_band(z, finite, n)
    else:
        sigma_noise = _noise_sigma(z, finite, n)
        center = _curvature_adaptive_centerline(z, sigma_noise, finite, n)
        residual = np.where(finite, z - center, 0.0)
        window = max(_RMS_WINDOW_MIN, int(n * _RMS_WINDOW_FRAC))
        rms = np.sqrt(uniform_filter1d(residual**2, size=window, mode="reflect"))
        sigma_unsmoothed = np.maximum(rms, sigma_noise)
        # floor at the smallest pre-smoothing σ: the Schoenberg spline can
        # overshoot below zero on rapidly-varying (multi-decade) input.
        sigma = np.maximum(_smooth_sigma(sigma_unsmoothed), float(np.min(sigma_unsmoothed)))

    return sigma


def bounds_from_sigma(
    y: np.ndarray, sigma: np.ndarray, *, sigma_multiple: float, scale: Scale,
) -> np.ndarray:
    """Build CheKiPEUQ-shaped ``(N, 2)`` bounds in linear ``y`` units.

    ``sigma`` is the per-point std in ``scale``-space. Bounds are
    ``inverse(forward(y) ± k·σ)`` so the cost function's own forward
    transform on ``obs_bounds`` recovers exactly ``k·σ``. Centered on
    each observed point so the optimizer's likelihood matches what the
    bounds represent.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    sigma_arr = np.asarray(sigma, dtype=float).ravel()
    if y_arr.shape != sigma_arr.shape:
        raise ValueError(f"shape mismatch: y={y_arr.shape}, sigma={sigma_arr.shape}")
    k = float(sigma_multiple)
    y_scaled = scale.forward(y_arr)
    lower = scale.inverse(y_scaled - k * sigma_arr)
    upper = scale.inverse(y_scaled + k * sigma_arr)
    bounds = np.column_stack([lower, upper])

    return bounds
