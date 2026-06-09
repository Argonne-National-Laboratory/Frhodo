"""IQR-based outlier detection.

Vendored from OpenDSM (Apache-2.0).
"""
import numba
import numpy as np

from frhodo._vendor.opendsm.stats_basic import _weighted_quantile, to_np_array


def IQR_outlier(data, weights=None, sigma_threshold=3, quantile=0.25):
    data = to_np_array(data)

    if weights is not None:
        weights = to_np_array(weights)

    return _IQR_outlier(data, weights, sigma_threshold, quantile)


@numba.jit(nopython=True, cache=True)
def _IQR_outlier(data, weights=None, sigma_threshold=3, quantile=0.25):
    if weights is None:
        q13 = np.nanquantile(data[np.isfinite(data)], [quantile, 1 - quantile])
    else:
        q13 = _weighted_quantile(
            data[np.isfinite(data)], np.array([quantile, 1 - quantile]), weights=weights
        )

    # 0.7413 * sigma - 0.5 fits the IQR multiplier corresponding to a sigma threshold
    q13_scalar = 0.7413 * sigma_threshold - 0.5
    iqr = np.diff(q13)[0] * q13_scalar
    outlier_threshold = np.array([q13[0] - iqr, q13[1] + iqr])

    return outlier_threshold


def remove_outliers(x, weights=None, sigma_threshold=3, quantile=0.25):
    if len(np.unique(x)) == 1:
        return x, np.arange(len(x))

    # Loosen the threshold until at least one inlier survives
    for sigma_added in range(10):
        outlier_bnds = _IQR_outlier(x, weights, sigma_threshold + sigma_added, quantile)
        idx_no_outliers = np.argwhere((x >= outlier_bnds[0]) & (x <= outlier_bnds[1])).flatten()

        if idx_no_outliers.size > 0:
            break

    # Fallback: keep the single point closest to the bounds
    if len(idx_no_outliers) == 0:
        dist = -np.minimum(x - outlier_bnds[0], outlier_bnds[1] - x)
        idx_no_outliers = np.array([np.argmin(dist)])

    x_no_outliers = x[idx_no_outliers]

    return x_no_outliers, idx_no_outliers
