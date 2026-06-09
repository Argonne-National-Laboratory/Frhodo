"""``CheKiPEUQ_Frhodo_interface`` construction and ``Bayesian_dict`` shape.

Validates that the interface populates the required keys with shapes
matching the input rate-constant / parameter counts. ``evaluate()`` is
exercised by the higher-level integration suite.
"""
import numpy as np
import pytest

from frhodo.optimize.cost.bayesian import CheKiPEUQ_Frhodo_interface


def _make_inputs(n_rates=2, n_params_per_rxn=3):
    return {
        "bayes_dist_type": "Automatic",
        "rxn_rate_opt": {
            "x0": np.linspace(1.0, float(n_rates), n_rates),
            "bnds": {
                "lower": -10.0 * np.ones(n_rates),
                "upper": 10.0 * np.ones(n_rates),
            },
        },
        "coef_opt": [],
        "rxn_coef_opt": [
            {
                "coef_x0": np.array([1.0, 0.0, 5000.0])[:n_params_per_rxn],
                "coef_bnds": {
                    "lower": np.array([0.5, -1.0, 4000.0])[:n_params_per_rxn],
                    "upper": np.array([2.0, 1.0, 6000.0])[:n_params_per_rxn],
                    "exist": np.ones((n_params_per_rxn, 2), dtype=bool),
                },
            },
        ],
    }


@pytest.fixture
def interface_default():
    return CheKiPEUQ_Frhodo_interface(**_make_inputs())


class TestBayesianDictKeys:
    def test_required_top_level_keys(self, interface_default):
        for key in (
            "pars_uncertainty_distribution",
            "rate_constants_initial_guess",
            "rate_constants_lower_bnds",
            "rate_constants_upper_bnds",
            "rate_constants_bnds_exist",
            "rate_constants_parameters_changing",
            "rate_constants_parameters_initial_guess",
            "rate_constants_parameters_lower_bnds",
            "rate_constants_parameters_upper_bnds",
            "rate_constants_parameters_bnds_exist",
            "pars_initial_guess",
            "pars_lower_bnds",
            "pars_upper_bnds",
            "pars_bnds_exist",
            "unbounded_indices",
            "pars_initial_guess_truncated",
            "pars_lower_bnds_truncated",
            "pars_upper_bnds_truncated",
            "pars_bnds_exist_truncated",
        ):
            assert key in interface_default.Bayesian_dict, (
                f"Bayesian_dict missing key {key!r}"
            )

    def test_distribution_passthrough(self):
        for dist in ("Automatic", "Gaussian", "Uniform"):
            inputs = _make_inputs()
            inputs["bayes_dist_type"] = dist
            iface = CheKiPEUQ_Frhodo_interface(**inputs)
            assert iface.Bayesian_dict["pars_uncertainty_distribution"] == dist


class TestBayesianDictShapes:
    @pytest.mark.parametrize("n_rates,n_params", [(1, 3), (2, 3), (5, 3)])
    def test_rate_arrays_match_n_rates(self, n_rates, n_params):
        iface = CheKiPEUQ_Frhodo_interface(
            **_make_inputs(n_rates=n_rates, n_params_per_rxn=n_params),
        )
        bd = iface.Bayesian_dict
        assert bd["rate_constants_initial_guess"].shape == (n_rates,)
        assert bd["rate_constants_lower_bnds"].shape == (n_rates,)
        assert bd["rate_constants_upper_bnds"].shape == (n_rates,)
        assert bd["rate_constants_bnds_exist"].shape == (n_rates, 2)

    def test_parameter_arrays_concatenate_per_rxn(self):
        inputs = _make_inputs()
        second_coef_x0 = np.array([2.0, 0.5, 4500.0])
        inputs["rxn_coef_opt"].append({
            "coef_x0": second_coef_x0,
            "coef_bnds": {
                "lower": np.array([1.0, -0.5, 3500.0]),
                "upper": np.array([4.0, 1.5, 5500.0]),
                "exist": np.ones((3, 2), dtype=bool),
            },
        })
        iface = CheKiPEUQ_Frhodo_interface(**inputs)
        bd = iface.Bayesian_dict
        assert bd["rate_constants_parameters_initial_guess"].shape == (6,)
        assert bd["rate_constants_parameters_lower_bnds"].shape == (6,)
        assert bd["rate_constants_parameters_upper_bnds"].shape == (6,)


class TestBayesianDictBoundsHandling:
    def test_unbounded_indices_empty_when_all_bounds_exist(self, interface_default):
        bd = interface_default.Bayesian_dict
        assert len(bd["unbounded_indices"]) == 0

    def test_truncated_arrays_match_full_when_no_unbounded(self, interface_default):
        bd = interface_default.Bayesian_dict
        np.testing.assert_array_equal(
            bd["pars_initial_guess_truncated"], bd["pars_initial_guess"],
        )
        np.testing.assert_array_equal(
            bd["pars_lower_bnds_truncated"], bd["pars_lower_bnds"],
        )

    def test_pars_initial_guess_concatenates_rates_then_params(self):
        n_rates = 2
        n_params = 3
        inputs = _make_inputs(n_rates=n_rates, n_params_per_rxn=n_params)
        iface = CheKiPEUQ_Frhodo_interface(**inputs)
        bd = iface.Bayesian_dict

        # First n_rates entries are the rate-constant initial guesses;
        # following entries are concatenated rxn-coef x0 values.
        np.testing.assert_array_equal(
            bd["pars_initial_guess"][:n_rates],
            bd["rate_constants_initial_guess"],
        )
        np.testing.assert_array_equal(
            bd["pars_initial_guess"][n_rates:],
            bd["rate_constants_parameters_initial_guess"],
        )


class TestSigmaMultiplePlumbing:
    """The user-facing ``bayes_unc_sigma`` knob must reach CheKiPEUQ's
    ``sigma_multiple`` parameter — otherwise the bounds → σ conversion
    silently uses the library default."""

    def test_default_sigma_multiple(self):
        iface = CheKiPEUQ_Frhodo_interface(**_make_inputs())
        assert iface.bayes_unc_sigma == pytest.approx(3.0)

    @pytest.mark.parametrize("k", [1.0, 1.96, 2.5, 4.0])
    def test_explicit_sigma_multiple_round_trip(self, k):
        iface = CheKiPEUQ_Frhodo_interface(**_make_inputs(), bayes_unc_sigma=k)
        assert iface.bayes_unc_sigma == pytest.approx(k)


class TestBayesianWeightsFlow:
    """The user's weighting profile (``shock.weight_*``) must reach
    ``Bayesian_dict['weights_data']`` so CheKiPEUQ uses it. Regression
    against earlier behavior where the Bayesian path overwrote
    ``shock.weights`` with a top-hat derived from ``unc_cutoff``."""

    def test_evaluate_writes_weights_data(self, monkeypatch):
        iface = CheKiPEUQ_Frhodo_interface(**_make_inputs())

        captured: dict = {}

        def fake_loader(**kwargs):
            captured["weights_data"] = kwargs.get("weights_data")
            captured["sigma_multiple"] = kwargs.get("sigma_multiple")

            class _Stub:
                def doMetropolisHastings(self, *args, **kwargs):
                    return None, None
                log_posteriors_un_normed_vec = [0.0]
                map_logP = 0.0
                map_parameter_set = np.zeros(2)
            return _Stub()

        monkeypatch.setattr(
            "frhodo.optimize.cost.bayesian.load_into_CheKiPUEQ", fake_loader,
        )

        user_weights = np.array([0.1, 0.4, 0.3, 0.2])
        output_dict = {
            "obs_sim_interp": [np.array([1.0, 1.1, 1.2, 1.3])],
            "obs_exp": [np.array([1.0, 1.1, 1.2, 1.3])],
            "obs_bounds": [np.array([[0.9, 1.1], [1.0, 1.2], [1.1, 1.3], [1.2, 1.4]])],
        }

        try:
            iface.evaluate(
                log_opt_rates=np.array([0.0, 0.0]),
                x=np.array([1.0, 0.0, 5000.0]),
                output_dict=output_dict,
                bayesian_weights=user_weights,
                iteration_num=0,
            )
        except Exception:
            pass  # the fake CheKiPEUQ stub returns minimal data; we only
            # care that load_into_CheKiPUEQ received the right kwargs.

        np.testing.assert_array_equal(captured["weights_data"], user_weights)
        assert captured["sigma_multiple"] == pytest.approx(iface.bayes_unc_sigma)
