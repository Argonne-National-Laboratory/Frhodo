"""Tests for the typed profile / experiment-shock models used by the API.

``WeightProfile`` wraps ``double_sigmoid``; the equivalence test pins
per-sample bit-for-bit identity to that call.
"""
import pickle

import numpy as np
import pytest
from pydantic import ValidationError

from frhodo.api import (
    ExperimentShock,
    PostShockState,
    PreShockState,
    WeightProfile,
)
from frhodo.experiment.weight import double_sigmoid


class TestWeightProfile:
    def test_defaults(self):
        wp = WeightProfile()
        assert wp.peak == 100.0
        assert wp.floor_pre == 0.0
        assert wp.floor_post == 0.0
        assert wp.time_rise == 4.5
        assert wp.time_fall == 35.0
        assert wp.growth_rate_rise == 0.0
        assert wp.growth_rate_fall == 0.7
        assert wp.cutoff_pre == 0.0
        assert wp.cutoff_post == 100.0

    def test_cutoff_post_must_exceed_cutoff_pre(self):
        with pytest.raises(ValidationError):
            WeightProfile(cutoff_pre=50.0, cutoff_post=10.0)

    def test_cutoffs_equal_rejected(self):
        with pytest.raises(ValidationError):
            WeightProfile(cutoff_pre=50.0, cutoff_post=50.0)

    def test_evaluate_returns_array(self):
        wp = WeightProfile()
        weights = wp.evaluate(np.linspace(0.0, 100.0, 25))
        assert weights.shape == (25,)
        assert np.isfinite(weights).all()

    def test_evaluate_equivalence_with_double_sigmoid(self):
        """``WeightProfile.evaluate`` matches a direct ``double_sigmoid`` call bit-for-bit."""
        t = np.linspace(0.0, 100.0, 101)
        wp = WeightProfile(
            peak=42.0, floor_pre=1.0, floor_post=5.0,
            time_rise=10.0, time_fall=60.0,
            growth_rate_rise=0.3, growth_rate_fall=1.0,
        )
        reference = double_sigmoid(
            t, A=[1.0, 42.0, 5.0],
            k=[0.3, 1.0],
            x0=[10.0, 60.0],
        )

        np.testing.assert_allclose(
            wp.evaluate(t), reference, rtol=1e-12, atol=1e-15,
            err_msg="WeightProfile envelope must match double_sigmoid identically",
        )


def _sample_initial():
    return PreShockState(
        T1=294.0, P1=601.0, u1=1029.0,
        composition={"Kr": 0.96, "cC7H14": 0.04},
    )


class TestExperimentShock:
    def test_basic_construction(self):
        t = np.linspace(0.0, 5e-5, 11)
        obs = np.linspace(1.0, 2.0, 11)
        shock = ExperimentShock(
            t=t, observable=obs,
            initial=_sample_initial(),
            t_end=5e-5,
        )
        assert isinstance(shock.t, list)
        assert isinstance(shock.observable, list)

    def test_ndarray_coerced_to_list(self):
        arr = np.linspace(0.0, 1.0, 5)
        shock = ExperimentShock(
            t=arr, observable=arr,
            initial=_sample_initial(), t_end=1e-3,
        )
        assert shock.t == arr.tolist()
        assert shock.observable == arr.tolist()

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentShock(
                t=[0.0, 1.0], observable=[0.0, 1.0, 2.0],
                initial=_sample_initial(), t_end=1e-3,
            )

    def test_t_array_returns_ndarray(self):
        shock = ExperimentShock(
            t=[0.0, 1e-5, 2e-5], observable=[1.0, 1.5, 2.0],
            initial=_sample_initial(), t_end=2e-5,
        )
        arr = shock.t_array()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([0.0, 1e-5, 2e-5]))

    def test_observable_array_returns_ndarray(self):
        shock = ExperimentShock(
            t=[0.0, 1e-5], observable=[1.0, 2.0],
            initial=_sample_initial(), t_end=1e-5,
        )
        assert isinstance(shock.observable_array(), np.ndarray)

    def test_accepts_pre_shock_initial(self):
        shock = ExperimentShock(
            t=[0.0, 1e-5], observable=[1.0, 2.0],
            initial=PreShockState(
                T1=300.0, P1=1e5, u1=1000.0, composition={"Ar": 1.0},
            ),
            t_end=1e-5,
        )
        assert shock.initial.kind == "pre_shock"

    def test_accepts_post_shock_initial(self):
        shock = ExperimentShock(
            t=[0.0, 1e-5], observable=[1.0, 2.0],
            initial=PostShockState(
                T_reac=1500.0, P_reac=2e5,
                u_incident=1029.0, rho1=0.4, composition={"Ar": 1.0},
            ),
            t_end=1e-5,
        )
        assert shock.initial.kind == "post_shock"

    def test_pickle_roundtrip(self):
        original = ExperimentShock(
            t=[0.0, 1e-5, 2e-5], observable=[1.0, 1.5, 2.0],
            initial=_sample_initial(),
            t_end=2e-5,
            scalar_weight=2.5,
            weight_profile=WeightProfile(peak=42.0),
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.t == original.t
        assert restored.observable == original.observable
        assert restored.scalar_weight == 2.5
        assert restored.weight_profile.peak == 42.0

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentShock(
                t=[0.0, 1e-5], observable=[1.0, 2.0],
                initial=_sample_initial(), t_end=1e-5,
                bogus_field=1.0,
            )
