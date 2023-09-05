import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import BSpline
import numba

from .adaptive_loss_tck import tck
from calculate.convert_units import OoM_numba


numba_cache = False
loss_alpha_min = -100.0


@numba.jit(nopython=True, cache=numba_cache)
def weighted_quantile(
    values, quantiles, weights=np.array([]), values_presorted=False, old_style=False
):
    """https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_presorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.quantile.
    :return: numpy.array with computed quantiles.
    """
    finite_idx = np.where(np.isfinite(values))
    values = values[finite_idx]
    if len(weights) == 0:
        weights = np.ones_like(values)
    else:
        weights = weights[finite_idx]

    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_presorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    res = np.cumsum(weights) - 0.5 * weights
    if old_style:  # To be convenient with numpy.quantile
        res -= res[0]
        res /= res[-1]
    else:
        res /= np.sum(weights)

    return np.interp(quantiles, res, values)


def remove_outliers(data, weights=np.array([]), sigma_threshold=3, quantile=0.25):
    outlier_bnds = IQR_outlier(data, weights, sigma_threshold, quantile)
    idx_no_outliers = np.argwhere(
        (data >= outlier_bnds[0]) & (data <= outlier_bnds[1])
    ).flatten()
    data_no_outliers = data[idx_no_outliers]

    return data_no_outliers, idx_no_outliers


@numba.jit(nopython=True, cache=numba_cache)
def IQR_outlier(data, weights=np.array([]), sigma_threshold=3, quantile=0.25):
    # only use finite data
    if len(weights) == 0:
        q13 = np.quantile(data[np.isfinite(data)], np.array([quantile, 1 - quantile]))
    else:  # weighted_quantile could be used always, don't know speed
        q13 = weighted_quantile(
            data[np.isfinite(data)], np.array([quantile, 1 - quantile]), weights=weights
        )

    q13_scalar = (
        0.7413 * sigma_threshold - 0.5
    )  # this is a pretty good fit to get the scalar for any sigma
    iqr = np.diff(q13)[0] * q13_scalar
    outlier_threshold = np.array([q13[0] - iqr, q13[1] + iqr])

    return outlier_threshold


# TODO: uncertain if these C functions should use np.min, np.mean, or np.max
@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def get_C(resid, mu, sigma, weights=np.array([]), C_scalar=1, quantile=0.25):
    q13 = IQR_outlier(
        resid - mu, weights=weights, sigma_threshold=sigma, quantile=quantile
    )
    C = np.max(np.abs(q13))

    if C == 0:
        C = OoM_numba(np.array([np.max(q13)]), method="floor")[0]

    return C*C_scalar # decreasing outliers increases outlier rejection


@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def generalized_loss_fcn(
    x, a=2, a_min=loss_alpha_min
):  # defaults to sum of squared error
    x_2 = x**2

    if a == 2.0:  # L2
        loss = 0.5 * x_2
    elif a == 1.0:  # smoothed L1
        loss = np.sqrt(x_2 + 1) - 1
    elif a == 0.0:  # Charbonnier loss
        loss = np.log(0.5 * x_2 + 1)
    elif a == -2.0:  # Cauchy/Lorentzian loss
        loss = 2 * x_2 / (x_2 + 4)
    elif a <= a_min:  # at -infinity, Welsch/Leclerc loss
        loss = 1 - np.exp(-0.5 * x_2)
    else:
        loss = np.abs(a - 2) / a * ((x_2 / np.abs(a - 2) + 1) ** (a / 2) - 1)

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def generalized_loss_derivative(x, c=1, a=2):
    if a == 2.0:  # L2
        dloss_dx = x / c**2
    elif a == 1.0:  # smoothed L1
        dloss_dx = x / c**2 / np.sqrt((x / c) ** 2 + 1)
    elif a == 0.0:  # Charbonnier loss
        dloss_dx = 2 * x / (x**2 + 2 * c**2)
    elif a == -2.0:  # Cauchy/Lorentzian loss
        dloss_dx = 16 * c**2 * x / (4 * c**2 + x**2) ** 2
    elif a <= loss_alpha_min:  # at -infinity, Welsch/Leclerc loss
        dloss_dx = x / c**2 * np.exp(-0.5 * (x / c) ** 2)
    else:
        dloss_dx = x / c**2 * ((x / c) ** 2 / np.abs(a - 2) + 1)

    return dloss_dx


@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def generalized_loss_weights(x: np.ndarray, a: float = 2, min_weight: float = 0.00):
    w = np.ones(len(x), dtype=numba.float64)
    for i, xi in enumerate(x):
        if a == 2 or xi <= 0:
            w[i] = 1
        elif a == 0:
            w[i] = 1 / (0.5 * xi**2 + 1)
        elif a <= loss_alpha_min:
            w[i] = np.exp(-0.5 * xi**2)
        else:
            w[i] = (xi**2 / np.abs(a - 2) + 1) ** (0.5 * a - 1)

    return w * (1 - min_weight) + min_weight


# approximate partition function for C=1, tau(alpha < 0)=1E5, tau(alpha >= 0)=inf 
# error < 4E-7
ln_Z_fit = BSpline.construct_fast(*tck)
ln_Z_inf = 11.206072645530174
def ln_Z(alpha, alpha_min=-1E6):
    if alpha <= alpha_min:
        return ln_Z_inf
    
    return ln_Z_fit(alpha)


# penalize the loss function using approximate partition function
# default to L2 loss
def penalized_loss_fcn(x, a=2, use_penalty=True):
    loss = generalized_loss_fcn(x, a)

    if use_penalty:
        penalty = ln_Z(a, loss_alpha_min)      # approximate partition function for C=1, tau=10
        loss += penalty

        if not np.isfinite(loss).all():
            # print("a: ", a)
            # print("x: ", x)
            # print("penalty: ", penalty)
            raise Exception("non-finite values in 'penalized_loss_fcn'")

    return loss


@numba.jit(nopython=True, error_model='numpy', cache=numba_cache) 
def alpha_scaled(s, a_max=2):
    if a_max == 2:
        a = 3
        b = 0.25

        if s < 0:
            s = 0

        if s > 1:
            s = 1

        s_max = (1 - 2/(1 + 10**a))
        s = (1 - 2/(1 + 10**(a*s**b)))/s_max

        alpha = loss_alpha_min + (2 - loss_alpha_min)*s
    
    else:
        x0 = 1
        k = 1.5 # 1 or 1.5, testing required

        if s >= 1:
            return 100
        elif s <= 0:
            return -100

        A = (np.exp((100 - x0)/k) + 1)/(1 - np.exp(200/k))
        K = (1 - A)*np.exp((x0 - 100)/k) + 1

        alpha = x0 - k*np.log((K - A)/(s - A) - 1)

    return alpha


def adaptive_loss_fcn(x, mu=0, c=1, alpha="adaptive", replace_nonfinite=True):
    if np.all(mu != 0) or np.all(c != 1):
        x = (x - mu) / c  # standardized residuals

    if replace_nonfinite:
        x[~np.isfinite(x)] = np.max(x)

    loss_alpha_fcn = lambda alpha: penalized_loss_fcn(
        x, a=alpha, use_penalty=True
    ).sum()

    if alpha == "adaptive":  #
        res = minimize_scalar(
            lambda s: loss_alpha_fcn(alpha_scaled(s)),
            bounds=[-1e-5, 1 + 1e-5],
            method="bounded",
            options={"xtol": 1e-5},
        )
        loss_alpha = alpha_scaled(res.x)
        # res = minimize(lambda s: loss_alpha_fcn(alpha_scaled(s[0])), x0=[0.7], bounds=[[0, 1]], method="L-BFGS-B")
        # loss_alpha = alpha_scaled(res.x[0])
        loss_fcn_val = res.fun

    else:
        loss_alpha = alpha
        loss_fcn_val = loss_alpha_fcn(alpha)

    return loss_fcn_val, loss_alpha


# Assumes that x has not been standardized
def adaptive_weights(
    x, weights=np.array([]), C_scalar=1, alpha="adaptive", 
    sigma=3, quantile=0.25, min_weight=0.00, replace_nonfinite=True
):
    x_no_outlier, _ = remove_outliers(x, sigma_threshold=sigma, quantile=0.25)

    # TODO: Should x be abs or not?
    # mu = np.median(np.abs(x_no_outlier))
    mu = np.median(x_no_outlier)

    C = get_C(x, mu, sigma, weights, C_scalar, quantile)
    x = (x - mu) / C

    if alpha == "adaptive":
        _, alpha = adaptive_loss_fcn(
            x, alpha=alpha, replace_nonfinite=replace_nonfinite
        )

    return generalized_loss_weights(x, a=alpha, min_weight=min_weight), C, alpha
