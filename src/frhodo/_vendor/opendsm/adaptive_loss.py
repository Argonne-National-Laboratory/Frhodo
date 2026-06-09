#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Optional, Tuple, Union

import numba
import numpy as np
from scipy.optimize import minimize_scalar

from frhodo._vendor.opendsm.adaptive_loss_Z import ln_Z, ln_Z_numba
from frhodo._vendor.opendsm.outliers import (
    remove_outliers,
    _IQR_outlier,
)
from frhodo._vendor.opendsm.stats_basic import (
    _median_absolute_deviation,
    _weighted_quantile,
)
from frhodo._vendor.opendsm.utils import OoM_numba

# Loss function constants
LOSS_ALPHA_MIN = -100.0
LOSS_ALPHA_MAX = 100.0


@numba.jit(nopython=True, cache=True)
def sliding_window(
    arr: np.ndarray, 
    window_size: int, 
    step: int = 0,
) -> np.ndarray:
    """Create sliding windows over a time series array.

    Reference: https://giov.dev/2018/05/a-window-on-numpy-s-views.html

    Args:
        arr: Input array with time advancing along dimension 0
        window_size: Size of sliding window
        step: Step size for sliding window. If 0, uses non-overlapping
            contiguous windows (step=window_size)

    Returns:
        Array with windowed views of the input data

    Raises:
        ValueError: If window_size > array size or step < 0
    """
    n_obs = arr.shape[0]

    # validate arguments
    if window_size > n_obs:
        raise ValueError(
            "Window size must be less than or equal "
            "the size of array in first dimension."
        )
    if step < 0:
        raise ValueError("Step must be positive.")

    n_windows = 1 + int(np.floor((n_obs - window_size) / step))

    obs_stride = arr.strides[0]
    windowed_row_stride = obs_stride * step

    new_shape = (n_windows, window_size) + arr.shape[1:]
    new_strides = (windowed_row_stride,) + arr.strides

    strided = np.lib.stride_tricks.as_strided(
        arr,
        shape=new_shape,
        strides=new_strides,
    )
    return strided


def rolling_IQR_outlier(
    x: np.ndarray,
    y: np.ndarray,
    sigma_threshold: float = 3.0,
    quantile: float = 0.25,
    window: Union[float, int] = 0.05,
    step: float = 1.0,
) -> np.ndarray:
    """Calculate rolling outlier thresholds using IQR method.

    Args:
        x: X-values of the data (e.g., time)
        y: Y-values of the data (residuals or observations)
        sigma_threshold: Sigma threshold for IQR outlier detection
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        window: Window size. If <= 1, treated as proportion of data length
        step: Step size for rolling calculation (as proportion of window if < 1)

    Returns:
        2D array with shape (2, len(x)) where row 0 is lower threshold
        and row 1 is upper threshold
    """

    if window <= 1.0:
        window = int(np.floor(len(y) * window))
    else:
        window = int(window)

    step = int(np.floor(window * step))
    if step < 1:
        step = 1

    y = np.abs(y)

    x_windows = sliding_window(x, window, step=step)
    y_windows = sliding_window(y, window, step=step)

    # Vectorized computation of means and quantiles
    x_interp = np.mean(x_windows, axis=1)
    q13 = np.quantile(y_windows, [quantile, 1 - quantile], axis=1)
    q1 = q13[0]
    q3 = q13[1]

    # Empirical scaling factor to convert sigma threshold to IQR multiplier
    q13_scalar = 0.7413 * sigma_threshold - 0.5
    iqr = (q3 - q1) * q13_scalar
    outlier_bnds = [q1 - iqr, q3 + iqr]

    outlier_threshold = np.zeros((2, len(x)))
    outlier_threshold[0] = np.interp(x, x_interp, outlier_bnds[0])
    outlier_threshold[1] = np.interp(x, x_interp, outlier_bnds[1])

    # x_interp = np.arange(0, len(outlier_bnds[0]))
    # x_orig = np.linspace(0, len(outlier_bnds[0]), len(x))

    # outlier_threshold = np.zeros((2, len(x)))
    # outlier_threshold[0] = np.interp(x_orig, x_interp, outlier_bnds[0])
    # outlier_threshold[1] = np.interp(x_orig, x_interp, outlier_bnds[1])

    return outlier_threshold


@numba.jit(nopython=True, error_model="numpy", cache=True)
def get_C(
    resid: np.ndarray,
    mu: float,
    sigma: float,
    quantile: float = 0.25,
    algo: str = "mad",
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate scale parameter C for adaptive loss weighting.

    Computes a robust scale estimate using various methods to normalize
    residuals for adaptive loss functions.

    Args:
        resid: Residuals from model fit
        mu: Location parameter (typically median of residuals)
        sigma: Scale factor (typically sigma threshold for outliers)
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        algo: Algorithm to use - 'iqr_legacy', 'iqr', 'mad', or 'stdev'
        weights: Optional per-observation weights threaded through the
            scale estimator (weighted IQR / MAD / std).

    Returns:
        Scale parameter C for normalizing residuals
    """
    finite_mask = np.isfinite(resid)
    resid = resid[finite_mask]

    # Numba can't narrow Optional[ndarray] across `if/else`, so narrow once
    # at the top via reassignment and steer with a boolean flag.
    if weights is None:
        weights_arr = np.ones(0, dtype=resid.dtype)
        has_weights = False
    else:
        weights_arr = weights[finite_mask]
        has_weights = True

    if algo == "iqr_legacy":
        if has_weights:
            bounds = _IQR_outlier(
                resid - mu, weights=weights_arr, sigma_threshold=sigma, quantile=quantile
            )
        else:
            bounds = _IQR_outlier(
                resid - mu, weights=None, sigma_threshold=sigma, quantile=quantile
            )
        C = np.max(np.abs(bounds))

    elif algo == "iqr":
        resid_abs = np.abs(resid)
        if has_weights:
            bounds = _IQR_outlier(
                resid_abs - mu, weights=weights_arr, sigma_threshold=sigma, quantile=quantile
            )
        else:
            bounds = _IQR_outlier(
                resid_abs - mu, weights=None, sigma_threshold=sigma, quantile=quantile
            )
        C = np.max(np.abs(bounds))

    elif algo == "mad":
        if has_weights:
            C = sigma * _median_absolute_deviation(resid, median=None, weights=weights_arr)
        else:
            C = sigma * _median_absolute_deviation(resid, median=None, weights=None)

    elif algo == "stdev":
        if has_weights:
            w_norm = weights_arr / np.sum(weights_arr)
            mean = np.sum(w_norm * resid)
            n = float(len(resid))
            var = np.sum(w_norm * (resid - mean) ** 2) / (1 - 1 / n)
            C = sigma * np.sqrt(var)
        else:
            C = sigma * np.std(resid)

    if C == 0:
        C = OoM_numba(np.array([C]), method="floor")[0]

    return C


def rolling_C(
    T: np.ndarray,
    resid: np.ndarray,
    mu: float,
    sigma: float = 3.0,
    quantile: float = 0.25,
    window: Union[float, int] = 0.2,
    step: float = 1.0,
) -> np.ndarray:
    """Calculate rolling scale parameter C for adaptive loss weighting.

    Args:
        T: Time or x-axis values
        resid: Residuals from model fit
        mu: Location parameter (typically median)
        sigma: Sigma threshold for outlier detection
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        window: Window size (proportion if <= 1, absolute if > 1)
        step: Step size for rolling calculation

    Returns:
        Array of rolling C values
    """
    q13 = rolling_IQR_outlier(T, resid - mu, sigma, quantile, window, step)
    C = np.max(np.abs(q13), axis=0)

    return C


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_fcn(
    x: Union[float, np.ndarray], 
    alpha: float = 2.0, 
    alpha_min: float = LOSS_ALPHA_MIN,
) -> Union[float, np.ndarray]:
    """Calculate generalized loss function value.

    Implements a family of robust loss functions parameterized by alpha.
    Different alpha values correspond to different well-known loss functions.

    Args:
        x: Input value(s) - typically normalized residuals
        alpha: Shape parameter determining loss function type
        alpha_min: Minimum alpha value for Welsch/Leclerc loss

    Returns:
        Loss function value(s)

    Loss function types by alpha value:
        - alpha = 2.0: L2 (squared error) loss
        - alpha = 1.0: Smoothed L1 (Pseudo-Huber) loss
        - alpha = 0.0: Charbonnier loss
        - alpha = -2.0: Cauchy/Lorentzian loss
        - alpha <= alpha_min: Welsch/Leclerc loss
        - other: Generalized Charbonnier loss
    """

    # Defaults to sum of squared error
    x_2 = x**2

    if alpha == 2.0:  # L2
        loss = 0.5 * x_2
    elif alpha == 1.0:  # smoothed L1
        loss = np.sqrt(x_2 + 1) - 1
    elif alpha == 0.0:  # Charbonnier loss
        loss = np.log(0.5 * x_2 + 1)
    elif alpha == -2.0:  # Cauchy/Lorentzian loss
        loss = 2 * x_2 / (x_2 + 4)
    elif alpha <= alpha_min:  # at -infinity, Welsch/Leclerc loss
        loss = 1 - np.exp(-0.5 * x_2)
    else:
        loss = np.abs(alpha - 2) / alpha * ((x_2 / np.abs(alpha - 2) + 1) ** (alpha / 2) - 1)

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_derivative(
    x: Union[float, np.ndarray], 
    scale: float = 1.0, 
    alpha: float = 2.0,
) -> Union[float, np.ndarray]:
    """Calculate derivative of generalized loss function.

    Computes the gradient of the loss function with respect to the input,
    accounting for the scale parameter.

    Args:
        x: Input value(s) - typically residuals
        scale: Scale parameter for normalization
        alpha: Shape parameter determining loss function type

    Returns:
        Derivative of loss function with respect to x

    Loss function types by alpha value:
        - alpha = 2.0: L2 loss
        - alpha = 1.0: Smoothed L1 (Pseudo-Huber) loss
        - alpha = 0.0: Charbonnier loss
        - alpha = -2.0: Cauchy/Lorentzian loss
        - alpha <= LOSS_ALPHA_MIN: Welsch/Leclerc loss
        - other: Generalized loss
    """
    if alpha == 2.0:  # L2
        dloss_dx = x / scale**2
    elif alpha == 1.0:  # smoothed L1
        dloss_dx = x / scale**2 / np.sqrt((x / scale) ** 2 + 1)
    elif alpha == 0.0:  # Charbonnier loss
        dloss_dx = 2 * x / (x**2 + 2 * scale**2)
    elif alpha == -2.0:  # Cauchy/Lorentzian loss
        dloss_dx = 16 * scale**2 * x / (4 * scale**2 + x**2) ** 2
    elif alpha <= LOSS_ALPHA_MIN:  # at -infinity, Welsch/Leclerc loss
        dloss_dx = x / scale**2 * np.exp(-0.5 * (x / scale) ** 2)
    else:
        dloss_dx = x / scale**2 * ((x / scale) ** 2 / np.abs(alpha - 2) + 1)

    return dloss_dx


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_weights(
    x: np.ndarray, 
    alpha: float = 2.0, 
    min_weight: float = 0.0,
) -> np.ndarray:
    """Calculate adaptive weights based on generalized loss function.

    Computes observation weights that downweight outliers according to
    the loss function shape parameter alpha.

    Args:
        x: Normalized residuals (typically (residuals - mu) / scale)
        alpha: Shape parameter determining weight behavior
        min_weight: Minimum weight value (prevents complete downweighting)

    Returns:
        Array of weights in range [min_weight, 1.0]
    """

    dtype = numba.float64
    if numba.config.DISABLE_JIT:
        dtype = np.float64

    # Vectorized computation
    x_sq = x**2
    w = np.ones(len(x), dtype=dtype)

    abs_x = np.abs(x)

    if alpha == 2.0:
        # L2 loss: all weights are 1.0
        pass
    elif alpha == 0.0:
        # Charbonnier loss
        w = np.where(abs_x > 0, 1.0 / (0.5 * x_sq + 1.0), 1.0)
    elif alpha <= LOSS_ALPHA_MIN:
        # Welsch/Leclerc loss
        w = np.where(abs_x > 0, np.exp(-0.5 * x_sq), 1.0)
    else:
        # Generalized loss
        w = np.where(abs_x > 0, (x_sq / np.abs(alpha - 2) + 1) ** (0.5 * alpha - 1), 1.0)

    return w * (1.0 - min_weight) + min_weight


def penalized_loss_fcn(
    x: np.ndarray, 
    alpha: float = 2.0, 
    use_penalty: bool = True,
) -> np.ndarray:
    """Calculate penalized loss function with partition function penalty.

    Adds a penalty term based on the approximate partition function to
    penalize more complex loss functions (lower alpha values).

    Args:
        x: Normalized input values (typically residuals)
        alpha: Shape parameter for loss function
        use_penalty: Whether to include partition function penalty

    Returns:
        Penalized loss values

    Raises:
        Exception: If non-finite values are found in calculated loss
    """
    loss = generalized_loss_fcn(x, alpha=alpha)

    if use_penalty:
        # Approximate partition function penalty for C=1, tau=10
        penalty = ln_Z(alpha, LOSS_ALPHA_MIN)
        loss += penalty

        if not np.isfinite(loss).all():
            # print("alpha: ", alpha)
            # print("x: ", x)
            # print("penalty: ", penalty)
            raise Exception("non-finite values in 'penalized_loss_fcn'")

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=True)
def alpha_scaled(
    s: float, 
    alpha_max: float = 2.0,
) -> float:
    """Convert scaled parameter s to alpha value.

    Transforms a bounded input s to the alpha parameter space using
    nonlinear scaling to provide smooth optimization behavior.

    Args:
        s: Scaled input value (typically in [0, 1] for optimization)
        alpha_max: Maximum alpha value (determines scaling method)

    Returns:
        Alpha value in range [LOSS_ALPHA_MIN, alpha_max] (approximately)
    """
    if alpha_max == 2.0:
        a = 3
        b = 0.25

        # Clip s to valid range
        if s < 0:
            s = 0
        if s > 1:
            s = 1

        # Nonlinear scaling using power law
        s_max = 1 - 2 / (1 + 10**a)
        s = (1 - 2 / (1 + 10 ** (a * s**b))) / s_max

        alpha = LOSS_ALPHA_MIN + (2.0 - LOSS_ALPHA_MIN) * s

    else:
        # Alternative scaling using logistic function
        x0 = 1.0
        k = 1.5

        if s >= 1:
            return LOSS_ALPHA_MAX
        elif s <= 0:
            return LOSS_ALPHA_MIN

        A = (np.exp((LOSS_ALPHA_MAX - x0) / k) + 1) / (
            1 - np.exp(2 * LOSS_ALPHA_MAX / k)
        )
        K = (1 - A) * np.exp((x0 - LOSS_ALPHA_MAX) / k) + 1

        alpha = x0 - k * np.log((K - A) / (s - A) - 1)

    return alpha


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _numba_penalized_loss(x_sq, obs_weights, alpha, alpha_min):
    """Weighted penalized loss for a given alpha. Fully numba."""
    n = len(x_sq)
    penalty = ln_Z_numba(alpha, alpha_min)
    total = 0.0
    if alpha == 2.0:
        for i in range(n):
            total += obs_weights[i] * (0.5 * x_sq[i] + penalty)
    elif alpha <= alpha_min:
        for i in range(n):
            total += obs_weights[i] * (1.0 - np.exp(-0.5 * x_sq[i]) + penalty)
    else:
        abs_a_m2 = np.abs(alpha - 2.0)
        coeff = abs_a_m2 / alpha
        exp = alpha / 2.0
        for i in range(n):
            total += obs_weights[i] * (coeff * ((x_sq[i] / abs_a_m2 + 1.0) ** exp - 1.0) + penalty)
    return total


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _numba_loss_weights(x_sq, alpha, alpha_min, min_weight):
    """Generalized loss weights for a given alpha. Fully numba."""
    n = len(x_sq)
    w = np.empty(n)
    if alpha == 2.0:
        for i in range(n):
            w[i] = 1.0
    elif alpha == 0.0:
        for i in range(n):
            w[i] = 1.0 / (0.5 * x_sq[i] + 1.0) if x_sq[i] > 0 else 1.0
    elif alpha <= alpha_min:
        for i in range(n):
            w[i] = np.exp(-0.5 * x_sq[i]) if x_sq[i] > 0 else 1.0
    else:
        exp = 0.5 * alpha - 1.0
        abs_a_m2 = np.abs(alpha - 2.0)
        for i in range(n):
            w[i] = (x_sq[i] / abs_a_m2 + 1.0) ** exp if x_sq[i] > 0 else 1.0
    if min_weight > 0.0:
        scale = 1.0 - min_weight
        for i in range(n):
            w[i] = w[i] * scale + min_weight
    return w


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _numba_brent_alpha(x_sq: np.ndarray, obs_weights: np.ndarray, alpha_min: float = LOSS_ALPHA_MIN) -> float:
    """Find optimal alpha via Brent's method, fully in numba.

    Minimizes the weighted penalized loss over alpha in scaled space [0, 1].
    Supports both uniform and kernel-weighted observations.
    """
    # Brent's method on [a, b] with golden section fallback
    a = -1e-5
    b = 1.0 + 1e-5
    tol = 1e-5
    golden = 0.3819660112501051  # (3 - sqrt(5)) / 2

    def _loss(s):
        return _numba_penalized_loss(x_sq, obs_weights, alpha_scaled(s), alpha_min)

    # Initialize
    x = a + golden * (b - a)
    w = x
    v = x
    e = 0.0
    fx = _loss(x)
    fw = fx
    fv = fx

    for _ in range(50):
        midpoint = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
            break

        # Try parabolic interpolation
        p = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e

        if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
            # Parabolic step
            e = p / q
            u = x + e
            if (u - a) < tol2 or (b - u) < tol2:
                e = tol1 if x < midpoint else -tol1
        else:
            # Golden section step
            e = (b if x < midpoint else a) - x
            e = golden * e

        u = x + (e if abs(e) >= tol1 else (tol1 if e > 0 else -tol1))
        fu = _loss(u)

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w; fv = fw
            w = x; fw = fx
            x = u; fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w; fv = fw
                w = u; fw = fu
            elif fu <= fv or v == x or v == w:
                v = u; fv = fu

    return alpha_scaled(x)


def adaptive_loss_fcn(
    x: np.ndarray,
    mu: float = 0.0,
    scale: float = 1.0,
    alpha: Union[str, float] = "adaptive",
    replace_nonfinite: bool = True,
    obs_weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Calculate adaptive loss function and optimal alpha parameter.

    Computes the total loss and optionally optimizes the alpha parameter
    to minimize the penalized loss function.

    Args:
        x: Input residuals
        mu: Location parameter for normalization
        scale: Scale parameter for normalization
        alpha: Shape parameter ('adaptive' for optimization, or fixed value)
        replace_nonfinite: Replace non-finite values with max finite value
        obs_weights: Optional per-observation weights for the loss sum.
            When provided, the loss becomes sum(obs_weights * penalized_loss)
            instead of sum(penalized_loss).  Used by kernel_adaptive_weights
            to localize alpha estimation.

    Returns:
        Tuple of (total loss value, alpha parameter used)
    """
    # Standardize residuals if needed
    if np.all(mu != 0.0) or np.all(scale != 1.0):
        x = (x - mu) / scale

    if replace_nonfinite:
        x[~np.isfinite(x)] = np.max(x[np.isfinite(x)])

    def _loss_for_alpha(alpha_val: float) -> float:
        """Compute total penalized loss for given alpha."""
        loss = penalized_loss_fcn(x, alpha=alpha_val, use_penalty=True)
        if obs_weights is not None:
            return (obs_weights * loss).sum()
        return loss.sum()

    if alpha == "adaptive":
        # Fully numba Brent optimization — single Python→numba call
        # eliminates per-iteration dispatch overhead.
        x_sq = x ** 2
        if obs_weights is not None:
            w = obs_weights
        else:
            w = np.full(len(x), 1.0 / len(x))
        loss_alpha = _numba_brent_alpha(x_sq, w, LOSS_ALPHA_MIN)
        loss_fcn_val = _loss_for_alpha(loss_alpha)
    else:
        loss_alpha = alpha
        loss_fcn_val = _loss_for_alpha(alpha)

    return loss_fcn_val, loss_alpha


def adaptive_weights(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: Union[str, float] = "adaptive",
    sigma: float = 3.0,
    quantile: float = 0.25,
    min_weight: float = 0.0,
    C_algo: str = "mad",
    replace_nonfinite: bool = True,
    C_scalar: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """Calculate adaptive weights for robust regression.

    Computes observation weights that downweight outliers based on
    the adaptive loss function. The scale and alpha parameters are
    automatically determined from the data.

    Args:
        x: Input residuals (not standardized)
        weights: Optional per-observation weights propagated to the
            outlier rejection, location (mu), scale (C), and alpha
            estimators.  Length must match ``x``.  None or empty array
            falls back to the unweighted path.
        alpha: Shape parameter ('adaptive' for optimization, or fixed value)
        sigma: Sigma threshold for outlier detection
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        min_weight: Minimum weight value (prevents complete downweighting)
        C_algo: Algorithm for scale estimation ('iqr_legacy', 'iqr', 'mad', 'stdev')
        replace_nonfinite: Replace non-finite values with max finite value
        C_scalar: Post-multiplier on C; smaller values tighten outlier
            rejection (1 / sensitivity).

    Returns:
        Tuple of (weights array, scale parameter C, alpha parameter)
    """
    if weights is not None and len(weights) == 0:
        weights = None

    x_no_outlier, idx_no_outlier = remove_outliers(
        x, weights=weights, sigma_threshold=sigma, quantile=0.25
    )

    if weights is not None:
        w_no_outlier = weights[idx_no_outlier]
        mu = _weighted_quantile(
            x_no_outlier,
            np.array([0.5]),
            weights=w_no_outlier,
            values_presorted=False,
        )[0]
    else:
        mu = np.median(x_no_outlier)

    C = get_C(x, mu, sigma, quantile, C_algo, weights=weights) * C_scalar
    x_normalized = (x - mu) / C

    if alpha == "adaptive":
        # `_numba_brent_alpha` weights the per-observation penalized loss;
        # it expects obs_weights pre-normalized to sum to 1.
        if weights is not None:
            obs_w = weights / weights.sum()
        else:
            obs_w = None
        _, alpha = adaptive_loss_fcn(
            x_normalized,
            alpha=alpha,
            replace_nonfinite=replace_nonfinite,
            obs_weights=obs_w,
        )

    return generalized_loss_weights(x_normalized, alpha=alpha, min_weight=min_weight), C, alpha


def _fast_weighted_alpha(r_normalized, obs_weights, n_candidates=20):
    """Fast weighted alpha optimization via inlined grid search.

    Avoids the overhead of repeated numba-jitted ``penalized_loss_fcn`` calls
    by inlining the generalized loss computation.  ~7x faster than
    ``adaptive_loss_fcn`` with ``minimize_scalar`` for typical array sizes.

    Args:
        r_normalized: Scale-normalized residuals.
        obs_weights: Per-observation kernel weights (sum to 1).
        n_candidates: Number of alpha candidates to evaluate.

    Returns:
        Optimal alpha value.
    """
    alpha_grid_s = np.linspace(0, 1, n_candidates)
    alpha_grid = np.array([alpha_scaled(s) for s in alpha_grid_s])

    x_sq = r_normalized ** 2
    best_loss = np.inf
    best_alpha = 2.0

    for a in alpha_grid:
        if a == 2.0:
            loss_arr = 0.5 * x_sq
        elif a <= LOSS_ALPHA_MIN:
            loss_arr = 1.0 - np.exp(-0.5 * x_sq)
        else:
            abs_a_m2 = np.abs(a - 2.0)
            loss_arr = abs_a_m2 / a * ((x_sq / abs_a_m2 + 1.0) ** (a / 2.0) - 1.0)
        total = (obs_weights * (loss_arr + ln_Z(a, LOSS_ALPHA_MIN))).sum()
        if total < best_loss:
            best_loss = total
            best_alpha = a

    return best_alpha


class KernelWeightCache:
    """Pre-computed kernel geometry for repeated adaptive weight calls.

    The kernel weight matrices depend only on the temperature array x and the
    knot configuration — not on the residuals.  By constructing this cache
    once per component and reusing it across adaptive iterations, we avoid
    recomputing bandwidth, evaluation grids, and the Gaussian kernel matrices
    on every call.
    """

    __slots__ = (
        "N", "x", "M", "M_alpha", "eval_x", "alpha_x",
        "scale_kernels", "alpha_kernels", "_q50",
    )

    def __init__(
        self,
        x: np.ndarray,
        zone_knot_count: int = 10,
        min_knot_spacing_pct: float = 0.025,
        n_eff_min: int = 15,
    ):
        N = len(x)
        self.N = N
        self.x = x
        self._q50 = np.array([0.5])

        if N < 3 or (x[-1] - x[0]) <= 0:
            self.M = 0
            self.M_alpha = 0
            self.eval_x = np.empty(0)
            self.alpha_x = np.empty(0)
            self.scale_kernels = []
            self.alpha_kernels = []
            return

        x_range = x[-1] - x[0]

        # Bandwidth selection
        base_points = 12
        knot_points = zone_knot_count * 3
        h_model = x_range / max(base_points + knot_points, 1)
        h_spacing = min_knot_spacing_pct * x_range
        h_data = n_eff_min * x_range / (N * np.sqrt(2 * np.pi))
        h = max(h_model, h_spacing, h_data)

        # Scale evaluation grid
        M = max(4, int(np.ceil(x_range / h)))
        self.M = M
        self.eval_x = np.linspace(x[0], x[-1], M)

        # Alpha evaluation grid (coarser)
        M_alpha = max(4, min(M, int(np.ceil(x_range / (h * 3)))))
        self.M_alpha = M_alpha
        self.alpha_x = np.linspace(x[0], x[-1], M_alpha)

        # Pre-compute and normalize kernel weight vectors
        self.scale_kernels = []
        for i in range(M):
            w = np.exp(-0.5 * ((x - self.eval_x[i]) / h) ** 2)
            w_sum = w.sum()
            if w_sum < 1e-12:
                self.scale_kernels.append(None)
            else:
                self.scale_kernels.append(w / w_sum)

        h_alpha = h * 3
        self.alpha_kernels = []
        for i in range(M_alpha):
            w = np.exp(-0.5 * ((x - self.alpha_x[i]) / h_alpha) ** 2)
            w_sum = w.sum()
            if w_sum < 1e-12:
                self.alpha_kernels.append(None)
            else:
                self.alpha_kernels.append(w / w_sum)

    def compute_weights(
        self,
        residuals: np.ndarray,
        min_weight: float = 0.0,
        return_local_scale: bool = False,
    ) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, np.ndarray]:
        """Compute adaptive weights using cached kernel geometry.

        Args:
            residuals: Model residuals (same length and order as x used
                to construct this cache).
            min_weight: Minimum weight value.
            return_local_scale: If True, also return per-observation local
                scale (MAD-based C) as a third element.

        Returns:
            Tuple of (weights, median_alpha) or (weights, median_alpha, C_local).
        """
        from scipy.interpolate import PchipInterpolator

        N = self.N
        if self.M == 0:
            return np.ones(N, dtype=float), 2.0

        # --- Pass 1: Scale (mu, C) at M points ---
        M = self.M
        eval_mu = np.empty(M)
        eval_C = np.empty(M)

        for i in range(M):
            w = self.scale_kernels[i]
            if w is None:
                eval_mu[i] = 0.0
                eval_C[i] = 1.0
                continue
            eval_mu[i] = _weighted_quantile(
                residuals, self._q50, weights=w, values_presorted=False
            )[0]
            C_i = _median_absolute_deviation(residuals, median=eval_mu[i], weights=w)
            eval_C[i] = max(C_i, 1e-10)

        # Interpolate mu, C to all observations
        x = self.x
        mu_interp = PchipInterpolator(self.eval_x, eval_mu, extrapolate=True)(x)
        C_interp = np.clip(
            PchipInterpolator(self.eval_x, eval_C, extrapolate=True)(x),
            1e-10, None,
        )

        # --- Pass 2: Alpha at M_alpha points ---
        r_normalized_all = (residuals - mu_interp) / C_interp
        r_normalized_all[~np.isfinite(r_normalized_all)] = 0.0

        M_alpha = self.M_alpha
        eval_alpha = np.empty(M_alpha)

        r_sq_all = r_normalized_all ** 2

        for i in range(M_alpha):
            w = self.alpha_kernels[i]
            if w is None:
                eval_alpha[i] = 2.0
                continue
            eval_alpha[i] = _numba_brent_alpha(r_sq_all, w, LOSS_ALPHA_MIN)

        # Interpolate alpha, clamp
        alpha_interp = np.clip(
            PchipInterpolator(self.alpha_x, eval_alpha, extrapolate=True)(x),
            LOSS_ALPHA_MIN, 2.0,
        )

        # --- Per-observation weights via numba (by alpha bucket) ---
        weights = np.ones(N, dtype=float)
        alpha_rounded = np.round(alpha_interp, 2)
        for alpha_val in np.unique(alpha_rounded):
            mask = alpha_rounded == alpha_val
            weights[mask] = _numba_loss_weights(
                r_sq_all[mask], float(alpha_val), LOSS_ALPHA_MIN, min_weight,
            )

        median_alpha = float(np.median(eval_alpha))
        if return_local_scale:
            return weights, median_alpha, C_interp
        return weights, median_alpha


def kernel_adaptive_weights(
    x: np.ndarray,
    residuals: np.ndarray,
    zone_knot_count: int = 10,
    min_knot_spacing_pct: float = 0.025,
    n_eff_min: int = 15,
    min_weight: float = 0.0,
    return_local_scale: bool = False,
    _cache: Optional["KernelWeightCache"] = None,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, np.ndarray]:
    """Kernel-weighted adaptive weights with bandwidth tied to model resolution.

    Computes per-observation adaptive weights where both the scale (C) and
    shape (alpha) of the generalized loss function are estimated locally via
    Gaussian kernel weighting.

    When called repeatedly on the same temperature array (e.g., across
    adaptive iterations), pass a ``KernelWeightCache`` via ``_cache`` to
    avoid recomputing bandwidth, evaluation grids, and kernel matrices.

    Args:
        x: Temperature values, sorted ascending (typically standardized).
        residuals: Model residuals corresponding to x (same length).
        zone_knot_count: Maximum interior knots per zone.
        min_knot_spacing_pct: Minimum knot spacing as fraction of data range.
        n_eff_min: Minimum effective neighbors per evaluation point.
        min_weight: Minimum weight value.
        return_local_scale: If True, return per-observation local scale (C)
            as a third element.
        _cache: Pre-computed kernel geometry.  If None, created internally.

    Returns:
        Tuple of (weights, median_alpha) or (weights, median_alpha, C_local).
    """
    if _cache is None:
        _cache = KernelWeightCache(x, zone_knot_count, min_knot_spacing_pct, n_eff_min)
    return _cache.compute_weights(residuals, min_weight=min_weight, return_local_scale=return_local_scale)


# Pre-compile all Numba JIT functions at import time to eliminate first-call
# compilation overhead during model fitting.
def _warmup_numba():
    _x = np.ones(20, dtype=np.float64)
    adaptive_weights(_x, alpha="adaptive", C_algo="mad")


_warmup_numba()