"""Subset of opendsm.common.utils used by the vendored stats package.

Vendored from OpenDSM (Apache-2.0).
"""
import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def OoM_numba(x, method="round"):
    """Order-of-magnitude (rounded log10) per element.

    method: "round" | "floor" | "ceil" | "exact"
    """
    x_OoM = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM
