"""Single source of truth for the optimizer's residual scale.

The optimizer's cost function transforms ``y_exp`` and ``y_sim`` into
a chosen scale before computing residuals (Linear / Log / Bisymlog).
Anything else that needs to work in the same space — σ estimation,
bounds construction, the plot's data band — should go through a
:class:`Scale` instance so they're consistent.

Bisymlog requires a calibrated knee parameter ``C`` derived from the
data range. Pre-2026 code spread that calibration across several
call sites, each remembering to share the same ``Bisymlog`` instance;
this class owns the calibrated instance internally so callers never
have to.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from frhodo.common.units import Bisymlog


ScaleMode = Literal["Linear", "Log", "AbsoluteLog", "Bisymlog"]
_MODES = ("Linear", "Log", "AbsoluteLog", "Bisymlog")


class Scale:
    """Forward + inverse transform owner for the optimizer's residual scale.

    Construct once for a given trace; ``forward(y)`` and ``inverse(z)``
    are pure functions of the input thereafter. Bisymlog calibration
    happens inside ``__init__`` from ``calibration_data`` — the caller
    doesn't need to remember to set ``C``.

    Modes:
        ``"Linear"`` — identity.
        ``"Log"`` — strict ``log10(y)``; non-positive ``y`` produces NaN.
        ``"AbsoluteLog"`` — ``log10(|y|)`` clamped at machine tiny.
        ``"Bisymlog"`` — symmetric log with linear knee, handles any sign.

    Args:
        mode: One of the modes listed above.
        calibration_data: Data array used to set the Bisymlog knee
            heuristically. Ignored for non-Bisymlog modes.
        bisymlog: Pre-calibrated :class:`Bisymlog` instance — useful
            when sharing calibration across shocks. When given, its
            ``C`` overrides anything derived from ``calibration_data``.
    """

    def __init__(
        self,
        mode: ScaleMode,
        *,
        calibration_data: np.ndarray | None = None,
        bisymlog: Bisymlog | None = None,
    ):
        if mode not in _MODES:
            raise ValueError(f"unknown scale: {mode!r}")
        self.mode: ScaleMode = mode
        self._bisymlog: Bisymlog | None = None
        if mode == "Bisymlog":
            if bisymlog is not None and bisymlog.C is not None:
                self._bisymlog = bisymlog
            else:
                self._bisymlog = bisymlog or Bisymlog()
                arr = np.asarray(
                    calibration_data
                    if calibration_data is not None
                    else np.array([0.0, 1.0]),
                    dtype=float,
                ).ravel()
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    finite = np.array([0.0, 1.0])
                self._bisymlog.set_C_heuristically(finite)

    @property
    def bisymlog(self) -> Bisymlog | None:
        """The calibrated :class:`Bisymlog` (Bisymlog mode only)."""
        return self._bisymlog

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Transform ``y`` into the residual scale."""
        arr = np.asarray(y, dtype=float)
        if self.mode == "Linear":
            return arr
        if self.mode == "Log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(arr > 0.0, np.log10(arr), np.nan)
        if self.mode == "AbsoluteLog":
            return np.log10(np.maximum(np.abs(arr), np.finfo(float).tiny))

        return self._bisymlog.transform(arr)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`forward`. ``AbsoluteLog`` inverse loses sign."""
        arr = np.asarray(z, dtype=float)
        if self.mode == "Linear":
            return arr
        if self.mode in ("Log", "AbsoluteLog"):
            return 10.0**arr

        return self._bisymlog.invTransform(arr)
