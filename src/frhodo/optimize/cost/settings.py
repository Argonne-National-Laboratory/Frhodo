"""Typed settings for the optimization stack.

``CostSettings`` shapes the cost function (scale, loss-function
parameters, condition-space coverage weighting).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict



class CostSettings(BaseModel):
    scale: Literal["Linear", "Log", "AbsoluteLog", "Bisymlog"] = "Bisymlog"
    bisymlog_scaling_factor: float = 1.0
    loss_alpha: float = 3.0  # 3.0 sentinel selects adaptive tuning
    loss_c: float = 1.0
    # Experiment balance: "uniqueness" = coverage × information weights
    # from sensitivity collinearity frozen at run start (falls back to
    # plain coverage if screening fails); "coverage" = inverse
    # condition-space density in (1000/T, log10 P) only; "none" = user
    # weights only.
    experiment_weighting: Literal["none", "coverage", "uniqueness"] = "uniqueness"

    model_config = ConfigDict(extra="forbid", frozen=True)
