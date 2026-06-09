"""Global elastic-net model of the per-shock time shift Δt(T, P, X).

Used when "Random t-uncertainty" is off: one shift surface is shared across
all shocks. The free-optimal per-shock shifts are regressed onto a quadratic
``(T, P)`` basis plus species mole fractions with an elastic-net penalty, and
the regularized (and clamped) predictions are used as the shifts. The penalty
is cross-validated over the shock set; features are standardized so the L1/L2
terms are scale-fair.

The model is fit and predicted on the same shock set each cost evaluation, so
there is no separate train/predict split — :func:`regularized_shifts` returns
the fitted shifts and the model diagnostics in one call.
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



_L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
_VARIANCE_FLOOR = 1e-12
_MIN_SHOCKS_FOR_MODEL = 3


def build_shift_features(conditions):
    """Raw feature matrix from per-shock ``(T, P, mole-fraction mapping)``.

    Columns are ``T, P, T^2, P^2, T*P`` followed by one column per species
    whose mole fraction varies across the set. Species that are constant
    (e.g. a fixed bath gas) carry no information and are dropped so they do
    not dilute the penalty.

    Args:
        conditions: sequence of ``(T, P, mix)`` where ``mix`` maps a species
            name to its mole fraction.

    Returns:
        ``(X, names)``: ``X`` is ``(n_shocks, n_features)`` and ``names``
        labels the columns.
    """
    T = np.array([c[0] for c in conditions], dtype=float)
    P = np.array([c[1] for c in conditions], dtype=float)
    species = sorted({s for _, _, mix in conditions for s in mix})
    comp = np.array(
        [[mix.get(s, 0.0) for s in species] for _, _, mix in conditions],
        dtype=float,
    )

    if comp.size:
        keep = comp.var(axis=0) > _VARIANCE_FLOOR
        comp = comp[:, keep]
        kept_species = [s for s, k in zip(species, keep) if k]
    else:
        kept_species = []

    poly = np.column_stack([T, P, T * T, P * P, T * P])
    if comp.size:
        X = np.column_stack([poly, comp])
    else:
        X = poly
    names = ["T", "P", "T^2", "P^2", "T*P", *kept_species]

    return X, names


def regularized_shifts(conditions, t_star, t_unc, *, l1_ratios=_L1_RATIOS, random_state=0):
    """Fit Δt(T,P,X) by standardized elastic-net CV on free-optimal shifts.

    Args:
        conditions: per-shock ``(T, P, mix)``.
        t_star: per-shock free-optimal time shifts (the regression target).
        t_unc: symmetric time-shift bound; predictions are clamped to
            ``[-t_unc, +t_unc]``.
        l1_ratios: elastic-net mixing grid searched by cross-validation.

    Returns:
        ``(shifts, info)``. ``shifts`` is the clamped per-shock prediction.
        ``info`` reports the fit: ``feature_names``, ``coefficients``,
        ``intercept``, selected ``penalty`` (sklearn ``alpha_``) and
        ``l1_ratio``, or ``model=None`` when there are too few shocks to fit.
    """
    t_star = np.asarray(t_star, dtype=float)
    n = len(conditions)

    if n < _MIN_SHOCKS_FOR_MODEL:
        shifts = np.clip(t_star, -t_unc, t_unc)
        info = {"model": None, "reason": "too few shocks for a parametric model"}

        return shifts, info

    X, names = build_shift_features(conditions)
    cv = max(2, min(5, n))
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=list(l1_ratios), cv=cv, random_state=random_state),
    )
    model.fit(X, t_star)
    enet = model.named_steps["elasticnetcv"]

    shifts = np.clip(model.predict(X), -t_unc, t_unc)
    info = {
        "model": model,
        "feature_names": names,
        "coefficients": enet.coef_,
        "intercept": float(enet.intercept_),
        "penalty": float(enet.alpha_),
        "l1_ratio": float(enet.l1_ratio_),
    }

    return shifts, info
