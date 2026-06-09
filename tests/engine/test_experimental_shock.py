"""``ExperimentalShock`` model construction + dict adoption.

The model defines the shape that's been a free-form dict for years.
Tests pin defaults (NaN where the loader hasn't filled in), the
two-element invariant on weight/uncertainty knobs, and that
``from_dict`` round-trips a legacy dict without dropping fields.
"""
import math

import numpy as np
import pytest

from frhodo.experiment import ExperimentalShock


class TestDefaults:
    def test_empty_factory(self):
        s = ExperimentalShock.empty(
            num=1, path={"exp": "/path/to/Shock1.exp"}, series_name="series A",
        )
        assert s.num == 1
        assert s.path == {"exp": "/path/to/Shock1.exp"}
        assert s.series_name == "series A"
        assert s.include is False
        assert s.run_SIM is True

    @pytest.mark.parametrize("field", ["T1", "P1", "u1", "rho1", "P4"])
    def test_pre_shock_state_is_nan(self, field):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        assert math.isnan(getattr(s, field))

    def test_zone_default_is_2(self):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        assert s.zone == 2

    def test_observable_default(self):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        assert s.observable == {"main": "", "sub": None}

    def test_weight_max_is_single_element_list(self):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        assert len(s.weight_max) == 1

    @pytest.mark.parametrize(
        "field",
        ["weight_min", "weight_shift", "weight_k"],
    )
    def test_two_element_lists(self, field):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        value = getattr(s, field)
        assert len(value) == 2, (
            f"{field} must have two entries (start, end) — set_boxes loader "
            f"requires it; got {value}"
        )

    @pytest.mark.parametrize(
        "field",
        ["raw_data", "exp_data", "exp_data_trim",
         "weights", "weights_trim", "normalized_weights", "sigma_t",
         "abs_uncertainties", "abs_uncertainties_trim", "SIM"],
    )
    def test_array_fields_default_empty_ndarray(self, field):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        value = getattr(s, field)
        assert isinstance(value, np.ndarray)
        assert value.size == 0

    def test_err_default_empty_list(self):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        assert s.err == []


class TestFromDict:
    def test_round_trip_known_keys(self):
        d = {
            "num": 7,
            "path": {"exp": "/p/Shock7.exp"},
            "series_name": "set B",
            "include": True,
            "T1": 295.0,
            "P1": 1330.0,
            "exp_data": np.array([[0.0, 1.0], [1.0, 2.0]]),
            "weight_max": [85.0],
            "weight_min": [0.0, 0.0],
            "weight_shift": [4.5, 35.0],
            "weight_k": [0.0, 0.7],
        }
        s = ExperimentalShock.from_dict(d)

        assert s.num == 7
        assert s.include is True
        assert s.T1 == pytest.approx(295.0)
        assert s.weight_max == [85.0]
        np.testing.assert_array_equal(s.exp_data, d["exp_data"])

    def test_unknown_keys_preserved(self):
        d = {
            "num": 1,
            "path": {},
            "series_name": "",
            "future_field": [1, 2, 3],
        }
        s = ExperimentalShock.from_dict(d)
        assert getattr(s, "future_field") == [1, 2, 3]

    def test_attribute_assignment(self):
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        s.T_reactor = 1500.0
        assert s.T_reactor == pytest.approx(1500.0)

    def test_setattr_for_runtime_keys(self):
        """Runtime-keyed callers use ``setattr``/``getattr`` instead of
        the dropped dict-style bridge."""
        s = ExperimentalShock.empty(num=1, path={}, series_name="")
        for key in ("T1", "P1", "u1"):
            setattr(s, key, 1.0)
            assert getattr(s, key) == pytest.approx(1.0)
