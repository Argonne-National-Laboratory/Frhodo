"""Utilities useful for using Frhodo from external optimizers"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import List, Optional, Dict, Sequence, Tuple

import numpy as np
from scipy import stats as ss
from scipy.interpolate import interp1d

from frhodo.api.driver import CoefIndex, FrhodoDriver, set_coefficients, get_coefficients, run_simulation
from frhodo.calculate.mech_fcns import Chemical_Mechanism


def _run_simulation(x, mech, parameters, observations, rxn_conditions, sim_kwargs):
    """Private method to run the simulations. Intended to be run in a Process as the simulator can throw seg faults"""

    # Update the parameters
    set_coefficients(mech, dict(zip(parameters, x)))

    # Run each simulation to each set of experimental data
    sims = []
    for sim_kwargs, rxn_cond in zip(sim_kwargs, rxn_conditions):
        res = run_simulation(mech, rxn_cond, sim_kwargs)
        sims.append(res)

    # Interpolate simulation data over the same steps as the experiments
    output = []
    for sim, obs in zip(sims, observations):
        sim_func = interp1d(sim[:, 0], sim[:, 1], kind='cubic', fill_value='extrapolate')
        output.append(sim_func(obs[:, 0]))
    return output


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
            num_workers: Optional[int] = None
    ):
        """

        Args:
            exp_directory: Path to the experimental data
            mech_directory: Path to the mechanism file(s)
            parameters: Parameters to be adjusted
            aliases: Aliases that map species in the experiment to the mechanism description
            num_workers: Maximum number of parallel workers to allow
        """
        self.parameters = parameters.copy()

        # Store where to find the experimental data
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

        # Holder for the process pool, which is not serialized
        self._exec: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(num_workers)
        self._num_workers = num_workers

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_exec']
        return state

    def __setstate__(self, state: dict):
        state = state.copy()
        state['_exec'] = ProcessPoolExecutor(state['_num_workers'])
        self.__dict__.update(state)

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
    def sim_kwargs(self) -> List[dict]:
        """Keyword arguments used to simulate each experiment"""
        if self._sim_kwargs is None:
            self.load_experiments()
        return self._sim_kwargs.copy()

    def rxn_conditions(self) -> List[Tuple[float, float, dict]]:
        """Starting conditions for each experiment"""
        if self._rxn_conditions is None:
            self.load_experiments()
        return self._rxn_conditions.copy()

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

        # Submit the simulation as a subprocess
        future = self._exec.submit(_run_simulation,
                                   x, self.mech, self.parameters, self.observations,
                                   self._rxn_conditions, self._sim_kwargs)
        try:
            return future.result()
        except BrokenProcessPool:
            raise ValueError(f'Process pool failed for: {x}')

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

    def _call(self, x: np.ndarray, **kwargs):
        """Invoke the objective function"""
        raise NotImplementedError()

    def __call__(self, x: np.ndarray, **kwargs):
        """Invoke the objective function"""
        try:
            return self._call(x)
        except ValueError:
            return np.inf


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
            **kwargs
    ):
        """

        Args:
            exp_directory: Path to the experimental data
            mech_directory: Path to the mechanism file(s)
            parameters: Parameters to be adjusted
            aliases: Aliases that map species in the experiment to the mechanism description
        """
        super().__init__(exp_directory, mech_directory, parameters, aliases, **kwargs)
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

    def _call(self, x: np.ndarray, **kwargs):
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
