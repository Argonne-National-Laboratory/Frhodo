# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import sys, os, io, stat, contextlib, pathlib, time
from copy import deepcopy
import cantera as ct
from cantera import interrupts, cti2yaml  # , ck2yaml, ctml2yaml
import numpy as np
from calculate import shock_fcns, integrate
import ck2yaml
from timeit import default_timer as timer


# list of all possible variables
all_var = {
    "Laboratory Time": {"SIM_name": "t_lab", "sub_type": None},
    "Shockwave Time": {"SIM_name": "t_shock", "sub_type": None},
    "Gas Velocity": {"SIM_name": "vel", "sub_type": None},
    "Temperature": {"SIM_name": "T", "sub_type": None},
    "Pressure": {"SIM_name": "P", "sub_type": None},
    "Enthalpy": {"SIM_name": "h", "sub_type": ["total", "species"]},
    "Entropy": {"SIM_name": "s", "sub_type": ["total", "species"]},
    "Density": {"SIM_name": "rho", "sub_type": None},
    "Density Gradient": {"SIM_name": "drhodz", "sub_type": ["total", "rxn"]},
    "% Density Gradient": {"SIM_name": "perc_drhodz", "sub_type": ["rxn"]},
    "\u00B1 % |Density Gradient|": {"SIM_name": "perc_abs_drhodz", "sub_type": ["rxn"]},
    "Mole Fraction": {"SIM_name": "X", "sub_type": ["species"]},
    "Mass Fraction": {"SIM_name": "Y", "sub_type": ["species"]},
    "Concentration": {"SIM_name": "conc", "sub_type": ["species"]},
    "Net Production Rate": {"SIM_name": "wdot", "sub_type": ["species"]},
    "Creation Rate": {"SIM_name": "wdotfor", "sub_type": ["species"]},
    "Destruction Rate": {"SIM_name": "wdotrev", "sub_type": ["species"]},
    "Heat Release Rate": {"SIM_name": "HRR", "sub_type": ["total", "rxn"]},
    "Delta Enthalpy (Heat of Reaction)": {"SIM_name": "delta_h", "sub_type": ["rxn"]},
    "Delta Entropy": {"SIM_name": "delta_s", "sub_type": ["rxn"]},
    "Equilibrium Constant": {"SIM_name": "eq_con", "sub_type": ["rxn"]},
    "Forward Rate Constant": {"SIM_name": "rate_con", "sub_type": ["rxn"]},
    "Reverse Rate Constant": {"SIM_name": "rate_con_rev", "sub_type": ["rxn"]},
    "Net Rate of Progress": {"SIM_name": "net_ROP", "sub_type": ["rxn"]},
    "Forward Rate of Progress": {"SIM_name": "for_ROP", "sub_type": ["rxn"]},
    "Reverse Rate of Progress": {"SIM_name": "rev_ROP", "sub_type": ["rxn"]},
}

rev_all_var = {
    all_var[key]["SIM_name"]: {"name": key, "sub_type": all_var[key]["sub_type"]}
    for key in all_var.keys()
}

# translation dictionary between SIM name and ct.SolutionArray name
SIM_Dict = {
    "t_lab": "t",
    "t_shock": "t_shock",
    "z": "z",
    "A": "A",
    "vel": "vel",
    "T": "T",
    "P": "P",
    "h_tot": "enthalpy_mole",
    "h": "partial_molar_enthalpies",
    "s_tot": "entropy_mole",
    "s": "partial_molar_entropies",
    "rho": "density",
    "drhodz_tot": "drhodz_tot",
    "drhodz": "drhodz",
    "perc_drhodz": "perc_drhodz",
    "Y": "Y",
    "X": "X",
    "conc": "concentrations",
    "wdot": "net_production_rates",
    "wdotfor": "creation_rates",
    "wdotrev": "destruction_rates",
    "HRR_tot": "heat_release_rate",
    "HRR": "heat_production_rates",
    "delta_h": "delta_enthalpy",
    "delta_s": "delta_entropy",
    "eq_con": "equilibrium_constants",
    "rate_con": "forward_rate_constants",
    "rate_con_rev": "reverse_rate_constants",
    "net_ROP": "net_rates_of_progress",
    "for_ROP": "forward_rates_of_progress",
    "rev_ROP": "reverse_rates_of_progress",
}


class SIM_Property:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.conversion = None  # this needs to be assigned per property
        self.value = {"SI": np.array([]), "CGS": np.array([])}
        self.ndim = self.value["SI"].ndim

    def clear(self):
        self.value = {"SI": np.array([]), "CGS": np.array([])}
        self.ndim = self.value["SI"].ndim

    def __call__(self, idx=None, units="CGS"):  # units must be 'CGS' or 'SI'
        # assumes Sim data comes in as SI and is converted to CGS
        # values to be calculated post-simulation
        if len(self.value["SI"]) == 0 or np.isnan(self.value["SI"]).all():
            parent = self.parent
            if self.name == "drhodz_tot":
                self.value["SI"] = shock_fcns.drhodz(parent.states)

            elif self.name == "drhodz":
                self.value["SI"] = shock_fcns.drhodz_per_rxn(parent.states)

            elif self.name == "perc_drhodz":
                drhodz_tot = parent.drhodz_tot(units="SI")[:, None]
                drhodz = parent.drhodz(units="SI").T
                if not np.any(drhodz_tot):
                    self.value["SI"] = np.zeros_like(drhodz)
                else:
                    self.value["SI"] = drhodz / np.abs(drhodz_tot) * 100

            elif self.name == "perc_abs_drhodz":
                drhodz_tot = parent.drhodz_tot(units="SI")[:, None]
                drhodz = parent.drhodz(units="SI").T
                if not np.any(drhodz_tot):
                    self.value["SI"] = np.zeros_like(drhodz)
                else:
                    self.value["SI"] = (
                        drhodz / np.abs(drhodz).sum(axis=1)[:, None] * 100
                    )

            else:
                self.value["SI"] = getattr(parent.states, SIM_Dict[self.name])

            if self.value["SI"].ndim > 1:  # Transpose if matrix
                self.value["SI"] = self.value["SI"].T

            self.ndim = self.value["SI"].ndim

        # currently converts entire list of properties rather than by index
        if units == "CGS" and len(self.value["CGS"]) == 0:
            if self.conversion is None:
                self.value["CGS"] = self.value["SI"]
            else:
                self.value["CGS"] = self.conversion(self.value["SI"])

        return self.value[units]


class Simulation_Result:
    def __init__(self, num=None, states=None, reactor_vars=[]):
        self.states = states
        self.all_var = all_var
        self.rev_all_var = rev_all_var
        self.reactor_var = {}
        for var in reactor_vars:
            if var in self.rev_all_var:
                self.reactor_var[self.rev_all_var[var]["name"]] = var

        if num is None:  # if no simulation stop here
            self.reactor_var = {}
            return

        self.conv = {
            "conc": 1e-3,
            "wdot": 1e-3,
            "P": 760 / 101325,
            "vel": 1e2,
            "rho": 1e-3,
            "drhodz_tot": 1e-5,
            "drhodz": 1e-5,
            "delta_h": 1e-3 / 4184,
            "h_tot": 1e-3 / 4184,
            "h": 1e-3 / 4184,  # to kcal
            "delta_s": 1 / 4184,
            "s_tot": 1 / 4184,
            "s": 1 / 4184,
            "eq_con": 1e3 ** np.array(num["reac"] - num["prod"])[:, None],
            "rate_con": np.power(1e3, num["reac"] - 1)[:, None],
            "rate_con_rev": np.power(1e3, num["prod"] - 1)[:, None],
            "net_ROP": 1e-3 / 3.8,  # Don't understand 3.8 value
            "for_ROP": 1e-3 / 3.8,  # Don't understand 3.8 value
            "rev_ROP": 1e-3 / 3.8,
        }  # Don't understand 3.8 value

        for name in reactor_vars:
            property = SIM_Property(name, parent=self)
            if name in self.conv:
                property.conversion = lambda x, s=self.conv[name]: x * s
            setattr(self, name, property)

    def set_independent_var(self, ind_var, units="CGS"):
        self.independent_var = getattr(self, ind_var)(units=units)

    def set_observable(self, observable, units="CGS"):
        k = observable["sub"]
        if observable["main"] == "Temperature":
            self.observable = self.T(units=units)
        elif observable["main"] == "Pressure":
            self.observable = self.P(units=units)
        elif observable["main"] == "Density Gradient":
            self.observable = self.drhodz_tot(units=units)
        elif observable["main"] == "Heat Release Rate":
            self.observable = self.HRR_tot(units=units)
        elif observable["main"] == "Mole Fraction":
            self.observable = self.X(units=units)
        elif observable["main"] == "Mass Fraction":
            self.observable = self.Y(units=units)
        elif observable["main"] == "Concentration":
            self.observable = self.conc(units=units)

        if (
            self.observable.ndim > 1
        ):  # reduce observable down to only plotted information
            self.observable = self.observable[k]

    def finalize(self, success, ind_var, observable, units="CGS"):
        self.set_independent_var(ind_var, units)
        self.set_observable(observable, units)

        self.success = success


class Reactor:
    def __init__(self, mech):
        self.mech = mech
        self.ODE_success = False

    def run(self, reactor_choice, t_end, T_reac, P_reac, mix, **kwargs):
        def list2ct_mixture(
            mix,
        ):  # list in the form of [[species, mol_frac], [species, mol_frac],...]
            return ", ".join(
                "{!s}:{!r}".format(species, mol_frac) for (species, mol_frac) in mix
            )

        details = {"success": False, "message": []}

        if isinstance(mix, list):
            mix = list2ct_mixture(mix)

        mech_out = self.mech.set_TPX(T_reac, P_reac, mix)
        if not mech_out["success"]:
            details["success"] = False
            details["message"] = mech_out["message"]
            return None, mech_out

        # start = timer()
        if reactor_choice == "Incident Shock Reactor":
            SIM, details = self.incident_shock_reactor(
                self.mech.gas, details, t_end, **kwargs
            )
        elif "0d Reactor" in reactor_choice:
            if reactor_choice == "0d Reactor - Constant Volume":
                reactor = ct.IdealGasReactor(self.mech.gas)
            elif reactor_choice == "0d Reactor - Constant Pressure":
                reactor = ct.IdealGasConstPressureReactor(self.mech.gas)

            SIM, details = self.zero_d_ideal_gas_reactor(
                self.mech.gas, reactor, details, t_end, **kwargs
            )

        # print('{:0.1f} us'.format((timer() - start)*1E3))
        return SIM, details

    def checkRxnRates(self, gas):
        limit = [
            1e9,
            1e15,
            1e21,
        ]  # reaction limit [first order, second order, third order]
        checkRxn = []
        for rxnIdx in range(gas.n_reactions):
            coef_sum = int(sum(gas.reaction(rxnIdx).reactants.values()))
            if type(gas.reactions()[rxnIdx]) is ct.ThreeBodyReaction:
                coef_sum += 1
            if coef_sum > 0 and coef_sum - 1 <= len(
                limit
            ):  # check that the limit is specified
                rate = [
                    gas.forward_rate_constants[rxnIdx],
                    gas.reverse_rate_constants[rxnIdx],
                ]
                if (
                    np.array(rate) > limit[coef_sum - 1]
                ).any():  # if forward or reverse rate exceeds limit
                    checkRxn.append(rxnIdx + 1)

        return checkRxn

    def incident_shock_reactor(self, gas, details, t_end, **kwargs):
        if "u_reac" not in kwargs or "rho1" not in kwargs:
            details["success"] = False
            details["message"] = "velocity and rho1 not specified\n"
            return None, details

        # set default values
        var = {
            "sim_int_f": 1,
            "observable": {"main": "Density Gradient", "sub": 0},
            "A1": 0.2,
            "As": 0.2,
            "L": 0.1,
            "t_lab_save": None,
            "ODE_solver": "BDF",
            "rtol": 1e-4,
            "atol": 1e-7,
        }
        var.update(kwargs)

        y0 = np.hstack(
            (0.0, var["A1"], gas.density, var["u_reac"], gas.T, 0.0, gas.Y)
        )  # Initial condition
        ode = shock_fcns.ReactorOde(
            gas, t_end, var["rho1"], var["L"], var["As"], var["A1"], False
        )

        with np.errstate(over="raise", divide="raise"):
            try:
                sol = integrate.solve_ivp(
                    ode,
                    [0, t_end],
                    y0,
                    method=var["ODE_solver"],
                    dense_output=True,
                    rtol=var["rtol"],
                    atol=var["atol"],
                )
                sol_success = True
                sol_message = sol.message
                sol_t = sol.t

            except:
                sol_success = False
                sol_message = sys.exc_info()[0]
                sol_t = sol.t

        if sol_success:
            self.ODE_success = (
                True  # this is passed to SIM to inform saving output function
            )
            details["success"] = True
        else:
            self.ODE_success = (
                False  # this is passed to SIM to inform saving output function
            )
            details["success"] = False

            # Generate log output
            explanation = "\nCheck for: Fast rates or bad thermo data"
            checkRxns = self.checkRxnRates(gas)
            if len(checkRxns) > 0:
                explanation += "\nSuggested Reactions: " + ", ".join(
                    [str(x) for x in checkRxns]
                )
            details["message"] = "\nODE Error: {:s}\n{:s}\n".format(
                sol_message, explanation
            )

        if var["sim_int_f"] > np.shape(sol_t)[0]:  # in case of integration failure
            var["sim_int_f"] = np.shape(sol_t)[0]

        if var["sim_int_f"] == 1:
            t_sim = sol_t
        else:  # perform interpolation if integrator sample factor > 1
            j = 0
            t_sim = np.zeros(
                var["sim_int_f"] * (np.shape(sol_t)[0] - 1) + 1
            )  # preallocate array
            for i in range(np.shape(sol_t)[0] - 1):
                t_interp = np.interp(
                    np.linspace(i, i + 1, var["sim_int_f"] + 1),
                    [i, i + 1],
                    sol_t[i : i + 2],
                )
                t_sim[j : j + len(t_interp)] = t_interp
                j += len(t_interp) - 1

        ind_var = "t_lab"  # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        if (
            var["t_lab_save"] is None
        ):  # if t_save is not being sent, only plotting variables are needed
            t_all = t_sim
        else:
            t_all = np.sort(
                np.unique(np.concatenate((t_sim, var["t_lab_save"])))
            )  # combine t_all and t_save, sort, only unique values

        states = ct.SolutionArray(
            gas,
            extra=[
                "t",
                "t_shock",
                "z",
                "A",
                "vel",
                "drhodz_tot",
                "drhodz",
                "perc_drhodz",
            ],
        )
        if self.ODE_success:
            for i, t in enumerate(t_all):  # calculate from solution
                y = sol.sol(t)
                z, A, rho, v, T, t_shock = y[0:6]
                Y = y[6:]

                states.append(
                    TDY=(T, rho, Y),
                    t=t,
                    t_shock=t_shock,
                    z=z,
                    A=A,
                    vel=v,
                    drhodz_tot=np.nan,
                    drhodz=np.nan,
                    perc_drhodz=np.nan,
                )
        else:
            states.append(
                TDY=(gas.T, gas.density, gas.Y),
                t=0.0,
                t_shock=0.0,
                z=0.0,
                A=var["A1"],
                vel=var["u_reac"],
                drhodz_tot=np.nan,
                drhodz=np.nan,
                perc_drhodz=np.nan,
            )

        reactor_vars = [
            "t_lab",
            "t_shock",
            "z",
            "A",
            "vel",
            "T",
            "P",
            "h_tot",
            "h",
            "s_tot",
            "s",
            "rho",
            "drhodz_tot",
            "drhodz",
            "perc_drhodz",
            "perc_abs_drhodz",
            "Y",
            "X",
            "conc",
            "wdot",
            "wdotfor",
            "wdotrev",
            "HRR_tot",
            "HRR",
            "delta_h",
            "delta_s",
            "eq_con",
            "rate_con",
            "rate_con_rev",
            "net_ROP",
            "for_ROP",
            "rev_ROP",
        ]

        num = {
            "reac": np.sum(gas.reactant_stoich_coeffs(), axis=0),
            "prod": np.sum(gas.product_stoich_coeffs(), axis=0),
            "rxns": gas.n_reactions,
        }

        SIM = Simulation_Result(num, states, reactor_vars)
        SIM.finalize(self.ODE_success, ind_var, var["observable"], units="CGS")

        return SIM, details

    def zero_d_ideal_gas_reactor(self, gas, reactor, details, t_end, **kwargs):
        # set default values
        var = {
            "observable": {"main": "Concentration", "sub": 0},
            "t_lab_save": None,
            "rtol": 1e-4,
            "atol": 1e-7,
        }

        var.update(kwargs)

        # Modify reactor if necessary for frozen composition and isothermal
        reactor.energy_enabled = var["solve_energy"]
        reactor.chemistry_enabled = not var["frozen_comp"]

        # Create Sim
        sim = ct.ReactorNet([reactor])
        sim.atol = var["atol"]
        sim.rtol = var["rtol"]

        # set up times and observables
        ind_var = "t_lab"  # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        if var["t_lab_save"] is None:
            t_all = [t_end]
        else:
            t_all = np.sort(
                np.unique(np.concatenate(([t_end], var["t_lab_save"])))
            )  # combine t_end and t_save, sort, only unique values

        self.ODE_success = True
        details["success"] = True

        states = ct.SolutionArray(gas, extra=["t"])
        states.append(reactor.thermo.state, t=0.0)
        for t in t_all:
            if not self.ODE_success:
                break
            while sim.time < t:  # integrator step until time > target time
                try:
                    sim.step()
                    if sim.time > t:  # force interpolation to target time
                        sim.advance(t)
                    states.append(reactor.thermo.state, t=sim.time)
                except:
                    self.ODE_success = False
                    details["success"] = False
                    explanation = "\nCheck for: Fast rates or bad thermo data"
                    checkRxns = self.checkRxnRates(gas)
                    if len(checkRxns) > 0:
                        explanation += "\nSuggested Reactions: " + ", ".join(
                            [str(x) for x in checkRxns]
                        )
                    details["message"] = "\nODE Error: {:s}\n{:s}\n".format(
                        str(sys.exc_info()[1]), explanation
                    )
                    break

        reactor_vars = [
            "t_lab",
            "T",
            "P",
            "h_tot",
            "h",
            "s_tot",
            "s",
            "rho",
            "Y",
            "X",
            "conc",
            "wdot",
            "wdotfor",
            "wdotrev",
            "HRR_tot",
            "HRR",
            "delta_h",
            "delta_s",
            "eq_con",
            "rate_con",
            "rate_con_rev",
            "net_ROP",
            "for_ROP",
            "rev_ROP",
        ]

        num = {
            "reac": np.sum(gas.reactant_stoich_coeffs(), axis=0),
            "prod": np.sum(gas.product_stoich_coeffs(), axis=0),
            "rxns": gas.n_reactions,
        }

        SIM = Simulation_Result(num, states, reactor_vars)
        SIM.finalize(self.ODE_success, ind_var, var["observable"], units="CGS")

        return SIM, details

    def plug_flow_reactor(self, gas, details, length, area, u_0, **kwargs):
        # set default values
        var = {
            "observable": {"main": "Concentration", "sub": 0},
            "t_lab_save": None,
            "rtol": 1e-4,
            "atol": 1e-7,
        }

        var.update(kwargs)

        # Modify reactor if necessary for frozen composition and isothermal
        reactor.energy_enabled = var["solve_energy"]
        reactor.chemistry_enabled = not var["frozen_comp"]

        # Create Sim
        sim = ct.ReactorNet([reactor])
        sim.atol = var["atol"]
        sim.rtol = var["rtol"]

        # set up times and observables
        ind_var = "t_lab"  # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        if var["t_lab_save"] is None:
            t_all = [t_end]
        else:
            t_all = np.sort(
                np.unique(np.concatenate(([t_end], var["t_lab_save"])))
            )  # combine t_end and t_save, sort, only unique values

        self.ODE_success = True
        details["success"] = True

        states = ct.SolutionArray(gas, extra=["t"])
        states.append(reactor.thermo.state, t=0.0)
        for t in t_all:
            if not self.ODE_success:
                break
            while sim.time < t:  # integrator step until time > target time
                try:
                    sim.step()
                    if sim.time > t:  # force interpolation to target time
                        sim.advance(t)
                    states.append(reactor.thermo.state, t=sim.time)
                except:
                    self.ODE_success = False
                    details["success"] = False
                    explanation = "\nCheck for: Fast rates or bad thermo data"
                    checkRxns = self.checkRxnRates(gas)
                    if len(checkRxns) > 0:
                        explanation += "\nSuggested Reactions: " + ", ".join(
                            [str(x) for x in checkRxns]
                        )
                    details["message"] = "\nODE Error: {:s}\n{:s}\n".format(
                        str(sys.exc_info()[1]), explanation
                    )
                    break

        reactor_vars = [
            "t_lab",
            "T",
            "P",
            "h_tot",
            "h",
            "s_tot",
            "s",
            "rho",
            "Y",
            "X",
            "conc",
            "wdot",
            "wdotfor",
            "wdotrev",
            "HRR_tot",
            "HRR",
            "delta_h",
            "delta_s",
            "eq_con",
            "rate_con",
            "rate_con_rev",
            "net_ROP",
            "for_ROP",
            "rev_ROP",
        ]

        num = {
            "reac": np.sum(gas.reactant_stoich_coeffs(), axis=0),
            "prod": np.sum(gas.product_stoich_coeffs(), axis=0),
            "rxns": gas.n_reactions,
        }

        SIM = Simulation_Result(num, states, reactor_vars)
        SIM.finalize(self.ODE_success, ind_var, var["observable"], units="CGS")

        return SIM, details
