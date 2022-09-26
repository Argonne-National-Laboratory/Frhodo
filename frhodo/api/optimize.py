"""Utilities useful for using Frhodo from external optimizers"""
import multiprocessing
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Sequence, Tuple

import numpy as np
from scipy import stats as ss
from scipy.interpolate import interp1d

from frhodo.api.driver import CoefIndex, FrhodoDriver, set_coefficients, get_coefficients, run_simulation
from frhodo.calculate.mech_fcns import Chemical_Mechanism


class BaseObjectiveFunction:
    """Base class for a Frhodo-based objective function

    Launches its own Frhodo GUI instance the first time you invoke a prediction function.
    """

    def __init__(
            self,
            exp_directory: Path,
            mech_directory: Path,
            parameters: List[CoefIndex],
            aliases: Optional[Dict[str, str]] = None,
    ):
        """

        Args:
            exp_directory: Path to the experimental data
            mech_directory: Path to the mechanism file(s)
            parameters: Parameters to be adjusted
            aliases: Aliases that map species in the experiment to the mechanism description
        """
        self.parameters = parameters.copy()

        # Make a placeholder for the Frhodo instance
        self._exp_directory = exp_directory
        self._mech_directory = mech_directory
        self._aliases = aliases
        self._frhodo: Optional[FrhodoDriver] = None

        # Make a placeholder for experimental data
        self.mech: Optional[Chemical_Mechanism] = None
        self._observations: Optional[List[np.ndarray]] = None
        self._weights: Optional[List[np.ndarray]] = None
        self._sim_kwargs: Optional[List[dict]] = None
        self._rxn_conditions: Optional[List[Tuple[float, float, dict]]] = None

    def __getstate__(self):
        if multiprocessing.get_start_method() != "spawn":
            warnings.warn('You must set the multiprocessing start method to spawn so we can run >1 multiprocessing instance')
        state = self.__dict__.copy()
        return state

    @property
    def observations(self) -> List[np.ndarray]:
        """Experimental data less any regions that are masked out"""
        # Load the data in if needed
        if self._observations is None:
            self.load_experiments()
        return self._observations

    @property
    def weights(self) -> List[np.ndarray]:
        """Weights for each observation"""
        if self._observations is None:
            self.load_experiments()
        return self._weights

    @property
    def x(self) -> np.ndarray:
        """Get the current state of the coefficient being optimized

        This is either the initial values from loading the mechanism or the last values ran.
        """
        return np.array(get_coefficients(self.mech, self.parameters))

    def load_experiments(self, frhodo: Optional[FrhodoDriver] = None):
        """Load observations and weights from disk

        Sets the values in `self._observations` and `self._weights`
        """

        if frhodo is None:
            frhodo = FrhodoDriver.create_driver()

        # Use Frhodo to load the data
        frhodo.load_files(
            self._exp_directory,
            self._mech_directory,
            self._mech_directory / 'outputs',
            aliases=self._aliases
        )

        # Extract the data
        self._observations = []
        self._weights = []
        for obs, weights in zip(*frhodo.get_observables()):
            # Find portions that are weighted sufficiently
            mask = weights > 1e-6

            # Store the masked weights and observations
            self._observations.append(obs[mask, :])
            self._weights.append(weights[mask])

        self._sim_kwargs, self._rxn_conditions = frhodo.get_simulator_inputs()
        self.mech = frhodo.window.mech

    def run_simulations(self, x: Sequence[float]) -> List[np.ndarray]:
        """Run the simulations with a new set of parameters

        Args:
            x: New set of reaction coefficients
        Returns:
            Simulated for each experiment interpolated at the same time increments
            as :meth:`observations`.
        """

        # Update the parameters
        assert len(x) == len(self.parameters), f"Expected {len(self.parameters)} parameters but got {len(x)}"
        set_coefficients(self.mech, dict(zip(self.parameters, x)))

        # Run each simulation to each set of experimental data
        sims = []
        for sim_kwargs, rxn_cond in zip(self._sim_kwargs, self._rxn_conditions):
            sims.append(run_simulation(self.mech, rxn_cond, sim_kwargs))

        # Interpolate simulation data over the same steps as the experiments
        output = []
        for sim, obs in zip(sims, self.observations):
            sim_func = interp1d(sim[:, 0], sim[:, 1], kind='cubic', fill_value='extrapolate')
            output.append(sim_func(obs[:, 0]))
        return output

    def compute_residuals(self, x: Sequence[float]) -> List[np.ndarray]:
        """Compute the residual between simulation and experiment

        Args:
            x: New set of reaction coefficients
        Returns:
            Residuals for each shock experiment at each point
        """

        # Run the simulations with the new parameters
        sims = self.run_simulations(x)

        # Compute residuals
        output = []
        for sim, obs in zip(sims, self.observations):
            output.append(np.subtract(sim, obs[:, 1]))
        return output

    def __call__(self, x: np.ndarray, **kwargs):
        """Invoke the objective function"""
        raise NotImplementedError()


class BayesianObjectiveFunction(BaseObjectiveFunction):
    """Computes the log-probability of observing experimental data given the simulated results

    Users must also pass an estimated "uncertainty" of experimental measurements along with the
    other input parameters to :meth:`__call__`. We place it as the first argument in the list.

    Uses a t-Distribution error model for the data to be robust against noise in the data and
    weighs data differently by scaling the width of the t-distribution so that it is wider
    for data with smaller weights.
    These approaches are modelled after the techniques demonstrated by
    `Paulson et al. 2019 <https://www.sciencedirect.com/science/article/abs/pii/S0020722518314721>`_.
    """

    def __init__(
            self,
            exp_directory: Path,
            mech_directory: Path,
            parameters: List[CoefIndex],
            priors: Optional[List[ss.rv_continuous]] = None,
            aliases: Optional[Dict[str, str]] = None,
    ):
        """

        Args:
            exp_directory: Path to the experimental data
            mech_directory: Path to the mechanism file(s)
            parameters: Parameters to be adjusted
            aliases: Aliases that map species in the experiment to the mechanism description
        """
        super().__init__(exp_directory, mech_directory, parameters, aliases)
        self.priors = priors

    def compute_log_probs(self, x: np.ndarray, uncertainty: float) -> np.ndarray:
        """Compute the log probability of observing each shock experiment

        Args:
            x: New values for each coefficient
            uncertainty: Size of the uncertainty for the observables
        """

        resids = self.compute_residuals(x)
        output = []
        for resid, weights in zip(resids, self.weights):
            # Adjust uncertainty so that the error tolerance of less-important data is larger
            #  See doi:10.1016/j.ijengsci.2019.05.011
            std = uncertainty / weights
            output.append(ss.t(loc=0, scale=std, df=2.1).logpdf(resid).sum())
        return np.array(output)

    def load_experiments(self, frhodo: Optional[FrhodoDriver] = None):
        super().load_experiments(frhodo)

        # Adjust the weights such that the largest is 1
        for i, weight in enumerate(self.weights):
            self.weights[i] /= weight.max()

    def __call__(self, x: np.ndarray, **kwargs):
        """Invoke the objective function

        Args:
            x: Uncertainty of the observations followed by the new coefficients
        """
        assert len(kwargs) == 0, "This function does not take keyword arguments"

        # Compute the log probability of the data
        uncertainty, coeffs = x[0], x[1:]
        logprob = self.compute_log_probs(coeffs, uncertainty).sum()

        # Compute the log probability from the priors
        if self.priors is not None:
            logprob += sum(p.logpdf(c) for p, c in zip(self.priors, x))

        return logprob
