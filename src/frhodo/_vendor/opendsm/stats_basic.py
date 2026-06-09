"""Weighted quantiles, MAD, and helpers for adaptive loss / robust stats.

Vendored from OpenDSM (Apache-2.0) with `to_np_array` inlined and the
import path collapsed under `frhodo.common`.
"""
from typing import Literal, Optional, Union

import numba
import numpy as np
from scipy.special import (
    stdtrit,
    erfinv,
)


# MAD-to-stdev scale factor for normal data: 1 / norm.ppf(0.75)
MAD_k = 1 / (erfinv(2 * 0.75 - 1) * np.sqrt(2))


def to_np_array(x):
    """Convert scalar/sequence to a 1D numpy array.

    None passes through unchanged.
    """
    if x is None:
        return None

    if not hasattr(x, "__len__"):
        x = [x]

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim == 0:
        x = np.array([x])

    return np.array(x)


def t_stat(alpha: float, n: int, tail: Union[int, str] = 2) -> float:
    """t-statistic for hypothesis testing at significance level alpha."""
    degrees_of_freedom = n - 1
    if (tail == "one") or (tail == 1):
        perc = np.asarray(1 - alpha)
    elif (tail == "two") or (tail == 2):
        perc = np.asarray(1 - alpha / 2)
    else:
        raise ValueError(f"Invalid tail parameter: {tail}. Must be 1/'one' or 2/'two'")

    return stdtrit(degrees_of_freedom, perc)


def z_stat(alpha: float, tail: Union[int, str] = 2) -> float:
    """z-statistic for hypothesis testing at significance level alpha."""
    if (tail == "one") or (tail == 1):
        perc = np.asarray(1 - alpha)
    elif (tail == "two") or (tail == 2):
        perc = np.asarray(1 - alpha / 2)
    else:
        raise ValueError(f"Invalid tail parameter: {tail}. Must be 1/'one' or 2/'two'")

    return erfinv(2 * perc - 1) * np.sqrt(2)


def unc_factor(
    n: int, interval: Literal["PI", "CI"] = "PI", alpha: float = 0.10
) -> float:
    """Uncertainty factor for confidence (CI) or prediction (PI) intervals."""
    if interval == "CI":
        return t_stat(alpha, n) / np.sqrt(n)
    elif interval == "PI":
        return t_stat(alpha, n) * (1 + 1 / np.sqrt(n))
    else:
        raise ValueError(f"Invalid interval: {interval}. Must be 'CI' or 'PI'")


@numba.jit(nopython=True, cache=True)
def weighted_std(
    x: np.ndarray,
    w: np.ndarray,
    mean: Optional[float] = None,
    w_sum_err: float = 1e-6,
) -> float:
    """Weighted standard deviation; renormalizes weights when sum != 1."""
    n = float(len(x))

    w_sum = np.sum(w)
    if w_sum < 1 - w_sum_err or w_sum > 1 + w_sum_err:
        w /= w_sum

    if mean is None:
        mean = np.sum(w * x)

    var = np.sum(w * np.power((x - mean), 2)) / (1 - 1 / n)

    return np.sqrt(var)


def fast_std(
    x: np.ndarray,
    weights: Optional[Union[np.ndarray, float, int]] = None,
    mean: Optional[float] = None,
) -> float:
    """Standard deviation; dispatches to weighted variant when weights vary."""
    if isinstance(weights, (int, float)):
        weights = np.array([weights])

    if weights is None or len(weights) == 1 or np.allclose(weights - weights[0], 0):
        if mean is None:
            return np.std(x)
        else:
            n = float(len(x))
            var = np.sum(np.power((x - mean), 2)) / n
            return np.sqrt(var)
    else:
        if mean is None:
            mean = np.average(x, weights=weights)

        return weighted_std(x, weights, mean)


@numba.jit(nopython=True, cache=True)
def _weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    values_presorted: bool = False,
    old_style: bool = False,
) -> np.ndarray:
    """Numba-jitted weighted quantile.

    Reference:
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    for q in quantiles:
        if not 0 <= q <= 1:
            raise ValueError("quantiles should be in [0, 1]")

    finite_idx = np.where(np.isfinite(values))
    values = values[finite_idx]

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = weights[finite_idx]

    if not values_presorted:
        sorted_idx = np.argsort(values)
        values = values[sorted_idx]
        weights = weights[sorted_idx]

    res = np.cumsum(weights) - 0.5 * weights
    if old_style:
        res -= res[0]
        res /= res[-1]
    else:
        res /= np.sum(weights)

    return np.interp(quantiles, res, values)


def weighted_quantile(
    values: Union[np.ndarray, list],
    quantiles: Union[np.ndarray, list, float],
    weights: Optional[Union[np.ndarray, list]] = None,
    values_presorted: bool = False,
    old_style: bool = False,
) -> np.ndarray:
    """Weighted quantile with input coercion to numpy arrays."""
    values = to_np_array(values)
    quantiles = to_np_array(quantiles)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = to_np_array(weights)

    try:
        res = _weighted_quantile(values, quantiles, weights, values_presorted, old_style)
    except Exception as e:
        print("Error in weighted_quantile:")
        print(f"  values shape: {values.shape}, dtype: {values.dtype}")
        print(f"  quantiles: {quantiles}")
        print(f"  weights shape: {weights.shape}, dtype: {weights.dtype}")
        raise Exception(f"Error in weighted_quantile: {str(e)}") from e

    return res


@numba.jit(nopython=True, cache=True)
def _median_absolute_deviation(
    x: np.ndarray,
    median: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Median Absolute Deviation scaled to match stdev under normality.

    1D only.  Pre-computed `median` skips the median pass.
    """
    mu = median
    if weights is None:
        if mu is None:
            mu = np.median(x)

        sigma = np.median(np.abs(x - mu))

    else:
        if mu is None:
            mu = _weighted_quantile(x, np.array([0.5]), weights=weights, values_presorted=False)[0]

        sigma = _weighted_quantile(
            np.abs(x - mu), np.array([0.5]), weights=weights, values_presorted=False
        )[0]

    return sigma * MAD_k


def median_absolute_deviation(
    x: Union[np.ndarray, list],
    median: Optional[float] = None,
    weights: Optional[Union[np.ndarray, list]] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """MAD scaled to standard deviation; supports per-axis evaluation."""
    x = to_np_array(x)

    if weights is not None:
        weights = to_np_array(weights)

    if axis is None:
        x_flat = x.ravel()
        weights_flat = weights.ravel() if weights is not None else None
        return _median_absolute_deviation(x_flat, median=median, weights=weights_flat)
    else:
        def mad_1d(x_slice):
            return _median_absolute_deviation(x_slice, median=None, weights=None)

        return np.apply_along_axis(mad_1d, axis, x)
