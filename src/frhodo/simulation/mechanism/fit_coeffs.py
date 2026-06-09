# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

"""Cantera rate-type dispatcher and Arrhenius fitter.

``fit_coeffs`` and ``fit_generic`` route by Cantera rate class:
ArrheniusRate goes to ``fit_arrhenius``; pressure-dependent types
(Plog/Falloff/Lindemann/Tsang/Troe/Sri) go to ``Troe`` from
:mod:`frhodo.simulation.mechanism.troe`.
"""
import warnings

import cantera as ct
import numpy as np
from scipy.optimize import lsq_linear, OptimizeWarning

from frhodo.simulation.mechanism.coef_helpers import (
    arrhenius_coefNames,
    set_arrhenius_bnds,
)
from frhodo.simulation.mechanism.troe_kernels import Ru



def _coef_index(coefNames, name):
    if isinstance(coefNames, np.ndarray):
        return np.argwhere(coefNames == name)[0]

    return coefNames.index(name)


def fit_arrhenius(rates, T, x0=None, coefNames=None, bnds=None):
    """Fit a modified Arrhenius rate by bounded linear least squares.

    The model ``ln k = ln A + n·ln T - Ea/(R_u·T)`` is linear in
    ``(-Ea/R_u, ln A, n)``; the fit reduces to one
    :func:`scipy.optimize.lsq_linear` solve.

    Args:
        rates: Forward rate constants at temperatures ``T``.
        T: Temperatures [K] parallel to ``rates``.
        x0: Optional ``[Ea, A, n]`` initial values. Coefs not named in
            ``coefNames`` stay at the ``x0`` values; the fit only varies
            the named subset. ``A`` is given in linear space.
        coefNames: Subset of :data:`arrhenius_coefNames` to fit; the
            rest are held. Defaults to fitting all three.
        bnds: Optional ``[lower, upper]`` matching ``coefNames`` order.
            Generated from :func:`set_arrhenius_bnds` when ``None``.

    Returns:
        Best-fit coefficient vector matching ``coefNames``, with
        ``pre_exponential_factor`` in linear space. Always returns a
        valid array — falls back to the bounds-clipped initial guess if
        the design is rank-deficient or the inputs are non-finite.
    """
    if coefNames is None:
        coefNames = arrhenius_coefNames
    coefNames = list(coefNames)

    T = np.asarray(T, dtype=float)
    ln_k = np.log(np.asarray(rates, dtype=float))

    # Design matrix columns in arrhenius_coefNames order [Ea, ln A, n]
    design = np.column_stack([
        -1.0 / (Ru * T),
        np.ones_like(T),
        np.log(T),
    ])

    held = np.zeros(3)
    if x0 is not None:
        held[:] = x0
        held[1] = np.log(held[1])

    fit_mask = np.array(
        [c in coefNames for c in arrhenius_coefNames], dtype=bool,
    )
    y = ln_k - design[:, ~fit_mask] @ held[~fit_mask]
    X = design[:, fit_mask]

    if bnds is None or len(bnds) == 0:
        bnds = set_arrhenius_bnds(held[fit_mask], coefNames)
    lb = np.asarray(bnds[0], dtype=float).copy()
    ub = np.asarray(bnds[1], dtype=float).copy()

    A_idx = None
    if "pre_exponential_factor" in coefNames:
        A_idx = coefNames.index("pre_exponential_factor")
        lb[A_idx] = np.log(lb[A_idx])
        ub[A_idx] = np.log(ub[A_idx])

    eps = np.maximum(np.abs(lb), 1.0) * 1e-12
    ub = np.maximum(ub, lb + eps)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt = lsq_linear(X, y, bounds=(lb, ub)).x
    except (ValueError, np.linalg.LinAlgError):
        popt = np.clip(held[fit_mask], lb, ub)

    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt


def fit_generic(
    rates,
    T,
    P,
    X,
    rxnIdx,
    coefKeys,
    coefNames,
    is_falloff_limit,
    mech,
    bnds,
):
    """Fit any supported rate type for one reaction.

    Routes by Cantera rate class: Arrhenius → :func:`fit_arrhenius`;
    pressure-dependent types (Plog/Falloff/Lindemann/Tsang/Troe/Sri) →
    :class:`~frhodo.simulation.mechanism.troe.Troe`.

    Returns:
        Coefficient array for the named subset, in the order matching
        ``coefNames``. For three-body Arrhenius reactions the
        pre-exponential is divided by ``M`` to recover the chemkin-style
        per-mole rate. ``None`` from :func:`fit_arrhenius` propagates.
    """
    # Keep inline: lifting Troe to module top forces torch (~1.4s) onto
    # every fit_coeffs import path, including arrhenius-only callers.
    from frhodo.simulation.mechanism.troe import Troe

    rxn = mech.gas.reaction(rxnIdx)
    rates = np.array(rates)
    T = np.array(T)
    P = np.array(P)
    coefNames = np.array(coefNames)
    bnds = np.array(bnds).copy()
    rms = None

    if type(rxn.rate) is ct.ArrheniusRate:
        x0 = [
            mech.coeffs_bnds[rxnIdx]["rate"][coefName]["resetVal"]
            for coefName in arrhenius_coefNames
        ]
        coeffs = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)
        if coeffs is None:
            return None, None

        if rxn.reaction_type.startswith("three-body") and (
            "pre_exponential_factor" in coefNames
        ):
            A_idx = _coef_index(coefNames, "pre_exponential_factor")
            coeffs[A_idx] = coeffs[A_idx] / mech.M(rxnIdx)

    elif type(rxn.rate) in [
        ct.PlogRate,
        ct.ChebyshevRate,
        ct.FalloffRate,
        ct.LindemannRate,
        ct.TsangRate,
        ct.TroeRate,
        ct.SriRate,
    ]:
        M = lambda T, P: mech.M(rxnIdx, [T, P, X])

        x0 = []
        for initial_parameters in mech.coeffs_bnds[rxnIdx].values():
            for coef in initial_parameters.values():
                x0.append(coef["resetVal"])

        falloff_coefNames = []
        for key, coefName in zip(coefKeys, coefNames):
            if key["coeffs_bnds"] == "low_rate":
                falloff_coefNames.append(f"{coefName}_0")
            elif key["coeffs_bnds"] == "high_rate":
                falloff_coefNames.append(f"{coefName}_inf")

        # Plog upgrade: the caller's coefNames are level-keyed, so the
        # loop above adds nothing. Fit the full Arrhenius limbs so the
        # orchestrator's upgrade machinery gets a 10-element coef_x0.
        if not falloff_coefNames:
            falloff_coefNames = [
                f"{cn}_{suffix}" for suffix in ["0", "inf"]
                for cn in arrhenius_coefNames
            ]

        falloff_coefNames.extend(["A", "T3", "T1", "T2"])
        Troe_parameters = Troe(
            rates,
            T,
            P,
            M,
            x0=x0,
            coefNames=falloff_coefNames,
            bnds=bnds,
            is_falloff_limit=is_falloff_limit,
        )
        coeffs, rms = Troe_parameters.fit()

    else:
        raise ValueError(
            f"Unsupported rate type for R{rxnIdx + 1}: "
            f"{type(rxn.rate).__name__}"
        )

    return coeffs, rms


def fit_coeffs(
    rates,
    T,
    P,
    X,
    rxnIdx,
    coefKeys,
    coefNames,
    is_falloff_limit,
    bnds,
    mech,
):
    """Public entry — forwards to :func:`fit_generic` after a fast-path skip.

    Empty ``coefNames`` returns ``None`` without invoking the fitter,
    so callers can pass through reactions that have nothing to fit.
    """
    if len(coefNames) == 0:
        return

    coeffs, _ = fit_generic(
        rates,
        T,
        P,
        X,
        rxnIdx,
        coefKeys,
        coefNames,
        is_falloff_limit,
        mech,
        bnds,
    )

    return coeffs


