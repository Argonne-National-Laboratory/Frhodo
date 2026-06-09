"""Typed settings for the optimization stack.

``CostSettings`` shapes the cost function (objective type, scale,
loss-function parameters, Bayesian distribution choice).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CostSettings(BaseModel):
    obj_fcn_type: Literal["Residual", "Bayesian"] = "Residual"
    scale: Literal["Linear", "Log", "AbsoluteLog", "Bisymlog"] = "Linear"
    bisymlog_scaling_factor: float = 1.0
    loss_alpha: float = 3.0  # 3.0 sentinel selects adaptive tuning
    loss_c: float = 1.0
    bayes_dist_type: str = "Automatic"
    bayes_unc_sigma: float = 3.0

    model_config = ConfigDict(extra="forbid", frozen=True)
