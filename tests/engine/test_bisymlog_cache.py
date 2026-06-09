"""``_trim_shocks`` populates ``shock.bisymlog`` once per opt run.

Pinning the cache contract: with ``scale="Bisymlog"`` the trim pass
attaches a ``Bisymlog`` instance whose ``C`` is set heuristically from
``exp_data_trim``. With other scales it stays ``None``. The residual
path then transforms via the cached object and never rebuilds it.
"""
import numpy as np

from frhodo.common.units import Bisymlog
from frhodo.experiment import ExperimentalShock
from frhodo.optimize.residual import _trim_shocks
from frhodo.optimize.cost.settings import CostSettings


def _shock_with_data():
    t = np.linspace(0.0, 1e-4, 50)
    obs = np.sin(t * 2e5) * 1e3
    s = ExperimentalShock.from_dict({
        "exp_data": np.column_stack([t, obs]),
        "weights": np.ones_like(t),
        "normalized_weights": np.ones_like(t),
    })

    return s


class TestBisymlogCache:
    def test_attached_when_scale_is_bisymlog(self):
        s = _shock_with_data()
        cost = CostSettings(scale="Bisymlog", bisymlog_scaling_factor=2.0)
        _trim_shocks([s], cost)
        assert isinstance(s.bisymlog, Bisymlog)
        assert s.bisymlog.scaling_factor == 2.0
        assert s.bisymlog.C is not None and s.bisymlog.C > 0

    def test_none_for_linear(self):
        s = _shock_with_data()
        cost = CostSettings(scale="Linear")
        _trim_shocks([s], cost)
        assert s.bisymlog is None

    def test_none_for_log(self):
        s = _shock_with_data()
        cost = CostSettings(scale="Log")
        _trim_shocks([s], cost)
        assert s.bisymlog is None

    def test_C_derived_from_exp_data_trim(self):
        """``set_C_heuristically`` is sensitive to its input. We pin
        that the cached C reflects the trimmed observable, not the
        full ``exp_data``."""
        s = _shock_with_data()
        cost = CostSettings(scale="Bisymlog", bisymlog_scaling_factor=1.0)
        _trim_shocks([s], cost)

        reference = Bisymlog(C=None, scaling_factor=1.0)
        reference.set_C_heuristically(s.exp_data_trim[:, 1])
        assert s.bisymlog.C == reference.C
