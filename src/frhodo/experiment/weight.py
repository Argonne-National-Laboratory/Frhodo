"""Weight / uncertainty function math.

``double_sigmoid`` is the parametric envelope GUI knobs feed into to
produce per-sample weights and uncertainties on each shock trace.
"""
import numpy as np


_min_pos_system_value = (np.finfo(float).tiny * (1e20)) ** (1 / 2)
_max_pos_system_value = (np.finfo(float).max * (1e-20)) ** (1 / 2)


def double_sigmoid(x, A, k, x0):
    """Two-sided sigmoid envelope. ``A`` = extrema, ``k`` = inverse growth rate, ``x0`` = shifts."""
    def sig(x):  # numerically stable sigmoid
        eval = np.empty_like(x)
        pos_val_f = np.exp(-x[x >= 0])
        eval[x >= 0] = 1 / (1 + pos_val_f)
        neg_val_f = np.exp(x[x < 0])
        eval[x < 0] = neg_val_f / (1 + neg_val_f)

        eval[eval > 0] = np.clip(
            eval[eval > 0], _min_pos_system_value, _max_pos_system_value
        )
        eval[eval < 0] = np.clip(
            eval[eval < 0], -_max_pos_system_value, -_min_pos_system_value
        )

        return eval

    def b_eval(x, k, x0):
        if k == 0:  # k = 0 means infinite growth rate
            b = np.ones_like(x) * np.inf
            if isinstance(x, (list, np.ndarray)):
                b[x < x0] *= -1
            elif x <= x0:
                b *= -1
        else:
            b = 1.5 / k * (x - x0)

        return b

    b = [[], []]
    for i in range(0, 2):
        b[i] = b_eval(x, k[i], x0[i])

    if not np.isfinite(b).any():
        a = (A[2] - A[0]) * sig(b[0]) + A[0]
    else:
        a = (A[2] - A[0]) * sig(np.mean(b, 0)) + A[0]

    res = (A[1] - a) * sig(b[0]) * sig(-b[1]) + a

    return res
