"""Series-settings controller — owns experiment series state for the GUI."""
from copy import deepcopy

import numpy as np
from scipy import integrate

from frhodo.common.scale import Scale
from frhodo.experiment import ExperimentLoader, ExperimentalShock, double_sigmoid
from frhodo.experiment.uncertainty import (
    bounds_from_sigma,
    estimate_pointwise_sigma,
)


class series:
    """Per-series experiment state: parsed shocks, weights, uncertainties.

    The main window holds one instance of this class. Each series
    holds a list of :class:`ExperimentalShock` records pulled from
    disk via :class:`ExperimentLoader`, plus the per-series weight
    and uncertainty profiles that override the global defaults.
    """

    def __init__(self, parent):
        self.parent = parent
        self.exp = ExperimentLoader(parent.convert_units)

        self.idx = 0  # series index number
        self.shock_idx = 0  # shock index number

        self.path = []
        self.name = []
        self.shock_num = []
        self.shock = []
        self.species_alias = []
        self.in_table = [False]

        self.initialize_shock()
        self.parent.display_shock = self.shock[self.idx][self.shock_idx]

    def initialize_shock(self):
        parent = self.parent

        self.path.append([])
        self.name.append("")
        self.shock_num.append([])
        self.species_alias.append({})
        self.in_table.append(False)

        self.update_current()

        shock = []
        shock.append(self._create_shock(1, {}))

        # Shock Parameters: in case no experimental series is loaded
        for var in ["T1", "P1", "u1"]:
            units = eval("str(parent." + var + "_units_box.currentText())")
            value = float(eval("str(parent." + var + "_value_box.value())"))
            setattr(shock[-1], var, parent.convert_units(value, units, unit_dir="in"))

        self.shock.append(shock)

    def _create_shock(self, num, shock_path):
        parent = self.parent
        offset = parent.time_offset_box.value()
        shock = ExperimentalShock.empty(
            num=num, path=shock_path, series_name=self.name[-1],
        )
        shock.time_offset = offset
        shock.opt_time_offset = offset
        shock.species_alias = self.species_alias[-1]

        return shock

    def add_series(self):
        parent = self.parent

        if (
            parent.path["exp_main"] in self.path
        ):  # check if series already exists before adding
            self.change_shock()
            return

        parent.path["shock"] = parent.path_set.shock_paths(prefix="Shock", ext="exp")
        if len(parent.path["shock"]) == 0:  # if no shocks in listed directory
            parent.directory.update_icons(invalid=["exp_main"])
            return

        if (
            self.in_table and not self.in_table[-1]
        ):  # if list exists and last item not in table, clear it
            self.clear_series(-1)

        self.path.append(deepcopy(parent.path["exp_main"]))
        self.name.append(parent.exp_series_name_box.text())
        self.shock_num.append(list(parent.path["shock"][:, 0].astype(int)))
        self.species_alias.append({})
        self.in_table.append(False)

        shock = []
        for shock_num, shock_path in parent.path["shock"]:
            shock.append(self._create_shock(shock_num, shock_path))

        self.shock.append(shock)
        self.change_shock()

    def change_series(self):
        self.change_shock()

    def added_to_table(self, n):  # update if in table
        self.in_table[n] = True

    def clear_series(self, n):
        del self.path[n], self.name[n], self.shock_num[n], self.species_alias[n]
        del self.shock[n], self.in_table[n]

    def clear_shocks(self):
        if self.parent.load_state.load_full_series:
            return

        self.update_idx()
        for shock in self.shock[self.idx]:
            shock.exp_data = np.array([])
            shock.raw_data = np.array([])
            shock.SIM = np.array([])

    def update_current(self):
        self.current = {
            "path": self.path[self.idx],
            "name": self.name[self.idx],
            "shock_num": self.shock_num[self.idx],
            "species_alias": self.species_alias[self.idx],
        }

    def update_idx(self):
        parent = self.parent

        try:
            self.idx = self.path.index(parent.path["exp_main"])
        except ValueError:
            # ``exp_main`` does not match any loaded series — leave
            # ``idx`` on the last-known-good series so ``change_shock``
            # can rewrite ``exp_main`` to match it.
            return
        self.shock_idx = parent.path_set.shock(
            self.shock_num[self.idx]
        )  # correct shock num to valid
        self.update_current()

    def weights(self, time, shock=[], calcIntegral=True):
        if not shock:
            shock = self.shock[self.idx][
                self.shock_idx
            ]  # sets parameters based on selected shock

        if len(shock.exp_data) == 0:
            return np.array([])

        parameters = [
            shock.weight_max, shock.weight_min,
            shock.weight_shift, shock.weight_k,
        ]
        if np.isnan(
            np.hstack(parameters)
        ).any():  # if weight parameters aren't set, default to gui
            self.parent.weight.update()

        t_conv = self.parent.reactor_state.t_unit_conv
        t0 = shock.exp_data[0, 0]
        tf = shock.exp_data[-1, 0]

        shift = np.array(shock.weight_shift) / 100 * (tf - t0) + t0
        k = np.array(shock.weight_k) * t_conv
        w_min = np.array(shock.weight_min) / 100
        w_max = shock.weight_max[0] / 100
        A = np.insert(w_min, 1, w_max)

        weights = double_sigmoid(time, A, k, shift)

        if (
            calcIntegral
        ):  # using trapazoidal method for efficiency, no simple analytical integral
            integral = integrate.cumulative_trapezoid(weights, time)[
                -1
            ]  # based on weights at data points

            if integral == 0.0:
                shock.normalized_weights = np.zeros_like(weights)
            else:
                weights_norm = weights.copy() / (integral / t_conv)
                shock.normalized_weights = weights_norm

        return weights

    def uncertainties(self, shock=None):
        """Estimate measurement-noise σ(t) and write linear-space bounds.

        Builds (or refreshes) a :class:`Scale` for the shock against
        the optimizer's current ``cost.scale``, stashes it on
        ``shock._scale``, and uses it for both σ estimation and the
        linear-unit bounds CheKiPEUQ ingests. Same ``Scale`` is read
        by the plot, so band visualization and likelihood always
        agree.
        """
        if shock is None:
            shock = self.shock[self.idx][self.shock_idx]
        scale = self.scale_for(shock)
        if shock.exp_data.size == 0:
            shock.sigma_t = np.array([])
            shock.abs_uncertainties = np.zeros((0, 2))

            return shock.sigma_t

        obs = shock.exp_data[:, 1]
        sigma_t = estimate_pointwise_sigma(obs, scale=scale)
        shock.sigma_t = sigma_t
        sigma_multiple = self._bayes_sigma_multiple()
        shock.abs_uncertainties = bounds_from_sigma(
            obs, sigma_t,
            sigma_multiple=sigma_multiple,
            scale=scale,
        )

        return sigma_t

    def scale_for(self, shock) -> Scale:
        """Return the ``Scale`` to use for ``shock``, building if needed.

        Caches on ``shock._scale`` so repeated calls don't re-calibrate
        Bisymlog. Rebuilds when the optimizer's ``cost.scale`` setting
        no longer matches the cached mode.
        """
        mode = self._cost_scale_mode()
        cached: Scale | None = getattr(shock, "_scale", None)
        if cached is not None and cached.mode == mode:
            return cached
        data = shock.exp_data[:, 1] if shock.exp_data.size else None
        scale = Scale(mode, calibration_data=data)
        shock._scale = scale

        return scale

    def _bayes_sigma_multiple(self) -> float:
        """Current ``bayes_unc_sigma`` knob, defaulting to 3 when unset."""
        try:
            val = self.parent.optimization_settings.get("obj_fcn", "bayes_unc_sigma")
        except (AttributeError, KeyError):

            return 3.0

        return float(val) if val is not None else 3.0

    def _cost_scale_mode(self) -> str:
        """Current optimizer residual scale (Linear / Log / AbsoluteLog / Bisymlog)."""
        try:
            val = self.parent.optimization_settings.get("obj_fcn", "scale")
        except (AttributeError, KeyError):

            return "Linear"

        return str(val) if val else "Linear"

    def set(self, key, val=[], **kwargs):
        parent = self.parent
        if key == "exp_data":
            if parent.load_state.load_full_series:
                shocks = self.shock[self.idx]
            else:
                self.clear_shocks()
                shocks = [self.shock[self.idx][self.shock_idx]]

            for shock in shocks:
                parameters, exp_data, raw_signal = self.exp.load_data(
                    shock.num, shock.path
                )
                for pname, pval in parameters.items():
                    setattr(shock, pname, pval)
                shock.exp_data = exp_data
                shock.raw_data = raw_signal

                for trace in ("exp_data", "raw_data"):
                    if getattr(shock, trace).size == 0:
                        shock.err.append(trace)

        elif (
            key == "series_name"
        ):
            self.name[self.idx] = val
            for shock in self.shock[self.idx]:
                shock.series_name = self.name[self.idx]

        elif key == "observable":
            for shock in self.shock[self.idx]:
                shock.observable["main"] = val[0]
                shock.observable["sub"] = val[1]

        elif key == "time_offset":
            for shock in self.shock[self.idx]:
                shock.time_offset = val

                if not parent.run_control.optimize_running:
                    shock.opt_time_offset = val

        elif key == "zone":
            for shock in self.shock[self.idx]:
                shock.zone = val
                shock.T_reactor = getattr(shock, f"T{val:d}")
                shock.P_reactor = getattr(shock, f"P{val:d}")

    def thermo_mix(self, shock=None):
        parent = self.parent
        alias = self.current["species_alias"]

        if shock is None:
            shock = parent.display_shock

        exp_mix = shock.exp_mix
        shock.thermo_mix = mix = deepcopy(shock.exp_mix)

        for species in exp_mix:
            if species in alias:
                mix[alias[species]] = mix.pop(species)

    def rates(self, shock, rxnIdxs=None):
        """Forward rate constants at the shock's reactor state.

        Args:
            shock: Carries the reactor T/P/composition. When
                ``rxnIdxs`` is ``None`` the full rate vector is
                also cached onto ``shock.rate_val`` for downstream
                consumers (Bayesian cost, plotters).
            rxnIdxs: Restrict computation to these reactions.
                ``None`` recomputes every reaction. Accepts an int
                for a single reaction.

        Returns:
            ``np.ndarray`` of length ``n_reactions`` when
            ``rxnIdxs`` is ``None``. Otherwise a ``dict`` keyed by
            reaction index. ``None`` is returned if the mechanism is
            not loaded or ``set_TPX`` rejects the state.
        """
        if not self.parent.mech.isLoaded:
            return None
        mech = self.parent.mech

        mech_out = mech.set_TPX(
            shock.T_reactor, shock.P_reactor, shock.thermo_mix
        )
        if not mech_out["success"]:
            self.parent.log.append(mech_out["message"])
            return None

        if rxnIdxs is None:
            shock.rate_val = np.asarray(mech.gas.forward_rate_constants)

            return shock.rate_val

        if isinstance(rxnIdxs, (int, np.integer)):
            indices = [int(rxnIdxs)]
        else:
            indices = list(rxnIdxs)
        forward = mech.gas.forward_rate_constants

        return {idx: forward[idx] for idx in indices}

    def rate_bnds(self, shock):
        if not self.parent.mech.isLoaded:
            return
        mech = self.parent.mech

        mech.set_TPX(shock.T_reactor, shock.P_reactor, shock.thermo_mix)

        prior_mech = mech.reset()

        shock.rate_reset_val = []
        shock.rate_bnds = []
        for rxnIdx in range(mech.gas.n_reactions):
            if self.parent.mech_tree.rxn[rxnIdx]["rxnType"] in [
                "Arrhenius",
                "Plog Reaction",
                "Falloff Reaction",
            ]:
                resetVal = mech.gas.forward_rate_constants[rxnIdx]
                shock.rate_reset_val.append(resetVal)
                if "limits" not in mech.rate_bnds[rxnIdx]:
                    print(rxnIdx)
                rate_bnds = mech.rate_bnds[rxnIdx]["limits"](resetVal)
                shock.rate_bnds.append(rate_bnds)

            else:  # skip if not Arrhenius type
                shock.rate_reset_val.append(np.nan)
                shock.rate_bnds.append([np.nan, np.nan])

        mech.coeffs = prior_mech
        mech.modify_reactions(mech.coeffs)

    def set_coef_reset(self, rxnIdx, coefName):
        mech = self.parent.mech
        reset_val = self.parent.mech.coeffs[rxnIdx][coefName]
        mech.coeffs_bnds[rxnIdx][coefName]["resetVal"] = reset_val

    def load_full_series(self):
        self.set("exp_data")
        self.parent.series_viewer.update()

    def change_shock(self):
        parent = self.parent

        self.update_idx()
        if parent.path["exp_main"] != self.path[self.idx]:
            parent.exp_main_box.setText(self.path[self.idx])
            self.update_idx()

        shock = self.shock[self.idx][self.shock_idx]
        parent.display_shock = shock

        if "exp_data" not in shock.err:
            if shock.exp_data.size == 0:
                self.set("exp_data")

        if np.isnan(shock.T_reactor):
            self.set("zone", shock.zone)

        # if weights not set, set them otherwise load
        parameters = [
            shock.weight_max, shock.weight_min,
            shock.weight_shift, shock.weight_k,
        ]
        if np.isnan(
            np.hstack(parameters)
        ).any():
            parent.weight.update()
        else:
            parent.weight.set_boxes()

        if not shock.rate_bnds:
            shock.rate_bnds = deepcopy(parent.display_shock.rate_bnds)

        if shock.observable == {"main": "", "sub": None}:
            parent.plot.observable_widget.update_observable()
        else:
            parent.plot.observable_widget.set_observable(shock.observable)

        for var_type in ["T1", "P1", "u1"]:
            parent.shock_widgets.set_shock_value_box(var_type)

        parent.plot.signal.clear_sim()
        parent.mix.update_species()

        self.rate_bnds(shock)

        parent.series_viewer.update(self.shock_idx)

        if parent.display_shock.exp_data.size > 0:
            parent.plot.signal.update(update_lim=True)
        else:
            parent.plot.signal.clear_plot()

        if parent.display_shock.raw_data.size > 0:
            parent.plot.raw_sig.update(update_lim=True)
        else:
            parent.plot.raw_sig.clear_plot()
