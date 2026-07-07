"""Per-shock preparation shared by the optimizer run and screening.

Trims each shock to its non-zero-weight window and attaches the
standardizing noise scales; both the residual objective and the pre-fit
screening consume the same prepared state.
"""
import numpy as np

from frhodo.common.scale import Scale
from frhodo.common.units import Bisymlog
from frhodo.experiment.uncertainty import correlation_length, sigma_bar



def trim_shocks(shocks2run: list, cost_settings) -> None:
    """Pre-mask each shock to the points where the weight profile is non-zero.

    Mutates each ``shock`` in place — populates ``weights_trim``,
    ``exp_data_trim``, ``bisymlog``, and ``sigma_bar`` (the per-shock
    noise scale standardizing its losses). Skipping zero-weight rows up
    front saves work inside every cost evaluation.
    """
    for shock in shocks2run:
        weights = shock.normalized_weights
        exp_bounds = np.nonzero(weights)[0]
        shock.weights_trim = weights[exp_bounds]
        shock.exp_data_trim = shock.exp_data[exp_bounds, :]

        if cost_settings.scale == "Bisymlog":
            bisymlog = Bisymlog(
                C=None, scaling_factor=cost_settings.bisymlog_scaling_factor,
            )
            bisymlog.set_C_heuristically(shock.exp_data_trim[:, 1])
            shock.bisymlog = bisymlog
        else:
            shock.bisymlog = None

        shock.sigma_bar = shock_sigma_bar(shock, cost_settings.scale)
        # Standardizing scale; calibrate_error_floor raises it to the
        # total expected error sqrt(sigma_bar^2 + E^2) before the run.
        shock.sigma_total = shock.sigma_bar
        shock.corr_length = shock_corr_length(shock, cost_settings.scale)


def shock_sigma_bar(shock, scale_mode: str) -> float:
    """σ̄ for one shock in the residual scale; 1.0 when unestimable.

    Bisymlog uses the shock's own calibrated transform so the noise
    scale and the residuals share identical units. Log-family scales
    use σ̄ = 1.0.
    """
    obs = shock.exp_data_trim[:, 1]
    if scale_mode == "Linear":
        z = np.asarray(obs, dtype=float)
    elif scale_mode == "Bisymlog":
        z = shock.bisymlog.transform(obs)
    else:
        return 1.0

    value = sigma_bar(z, scale=Scale("Linear"), window_weights=shock.weights_trim)
    if not np.isfinite(value) or value <= 0:
        return 1.0

    return float(value)


def shock_corr_length(shock, scale_mode: str) -> float:
    """Residual correlation length in samples; 1.0 when unestimable.

    Feeds the loss-scale floor's effective dof so correlated noise does
    not shrink the floor below the loss statistic's true sampling scale.
    """
    obs = shock.exp_data_trim[:, 1]
    if scale_mode == "Linear":
        z = np.asarray(obs, dtype=float)
    elif scale_mode == "Bisymlog":
        z = shock.bisymlog.transform(obs)
    else:
        return 1.0

    value = correlation_length(
        z, scale=Scale("Linear"), window_weights=shock.weights_trim,
    )

    return float(max(value, 1.0))
