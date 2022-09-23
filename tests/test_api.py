"""Testing the API components of Frhodo"""
import numpy as np
from pytest import fixture, mark
from multiprocessing import Pool, set_start_method

from frhodo.api.driver import FrhodoDriver
from frhodo.api.optimize import BayesianObjectiveFunction
from frhodo.api import optimize


@fixture
def loaded_frhodo(frhodo_driver, example_dir, tmp_path):
    """Set up the driver with a specific problem case"""
    frhodo_driver.load_files(
        example_dir / 'Experiment',
        example_dir / 'Mechanism',
        tmp_path,
        {'A': 'B'}  # A fake alias
    )
    return frhodo_driver


def test_launch():
    """Make sure we can launch Frhodo"""
    driver = FrhodoDriver.create_driver()
    assert driver.window.isVisible()


def test_load(loaded_frhodo):
    """Load loading in desired data"""
    assert loaded_frhodo.n_shocks == 1


def test_observables(loaded_frhodo):
    runs, weights = loaded_frhodo.get_observables()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2

    assert len(weights) == 1
    assert weights[0].size == runs[0].shape[0]


def test_simulate(loaded_frhodo):
    runs = loaded_frhodo.run_simulations()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2

    # Make sure the aliases propagated through
    #  They are reloaded from configuration variables when we re-run a simulation
    assert loaded_frhodo.window.display_shock['species_alias']['A'] == 'B'

    # Test running the simulation from the keyword arguments
    kwargs, rxn_cond = loaded_frhodo.get_simulator_inputs()
    manual_sim = loaded_frhodo.run_simulation_from_kwargs(kwargs[0], rxn_cond[0])
    assert np.isclose(runs[0], manual_sim).all()

@mark.parametrize(
    'rxn_id, prs_id', [(0, 0),  # PLog TODO (wardlt): Reaction 0 fails because we don't yet support PLog reactions
                       (3, 0),  # Elementary
                       (36, 'low_rate')]  # Three-body
)
def test_update(loaded_frhodo, rxn_id, prs_id):
    """Test that we can update individual parameters"""
    # Get the initial reaction rates
    rates = loaded_frhodo.get_reaction_rates()
    assert rates.shape == (66, 1)

    # Get the simulation before
    sim = loaded_frhodo.run_simulations()[0]

    # Get the original value
    coef_ind = (rxn_id, prs_id, 'pre_exponential_factor')
    orig_value = loaded_frhodo.get_coefficients([coef_ind])[0]

    # Update one of the rates
    loaded_frhodo.set_coefficients({coef_ind: 200})
    assert loaded_frhodo.rxn_coeffs[rxn_id][prs_id]['pre_exponential_factor'] == 200
    assert loaded_frhodo.get_coefficients([coef_ind]) == [200]

    # Make sure that only one rate changes
    new_rates = loaded_frhodo.get_reaction_rates()
    changes = np.abs(rates - new_rates)
    assert changes[rxn_id, 0] > 0
    mask = np.ones_like(rates, bool)
    mask[rxn_id, 0] = False
    assert np.isclose(changes[mask], 0).all()

    # Make sure that this changes the simulation
    new_sim = loaded_frhodo.run_simulations()[0]
    try:
        assert not np.isclose(sim[-1, 1], new_sim[-1, 1], rtol=1e-4)  # Look just at the last point
    finally:
        # Set it back to not mess-up our other tasks
        loaded_frhodo.set_coefficients({coef_ind: orig_value})


def test_fittable_parameters(loaded_frhodo):
    """Test getting lists of parameters to update"""

    # Make sure we can get all parameters
    total = loaded_frhodo.get_fittable_parameters()
    assert len(total) == 360

    # Make sure it works with each reaction type
    assert len(loaded_frhodo.get_fittable_parameters([0])) == 24  # plog
    assert len(loaded_frhodo.get_fittable_parameters([3])) == 3  # elementary
    assert len(loaded_frhodo.get_fittable_parameters([36])) == 6  # Troe

    # Test getting a parameter
    assert np.isclose(loaded_frhodo.get_coefficients([(1, 0, 'pre_exponential_factor')]), 5.9102033e+93)


def test_optimizer(loaded_frhodo, example_dir, tmp_path):
    set_start_method("spawn")  # Allows us to run >1 Frhodo instance
    opt = BayesianObjectiveFunction(
        exp_directory=example_dir / 'Experiment',
        mech_directory=example_dir / 'Mechanism',
        parameters=[(3, 0, 'pre_exponential_factor')]
    )

    # Set the frhodo executable for that module (we can have only 1 per process)
    optimize._frhodo = loaded_frhodo

    # Test the state
    assert opt.weights[0].max() == 1
    assert len(opt.x) == 1

    # Make sure the optimizer produces different results with different inputs
    x0 = opt.x.tolist()
    x0.insert(0, 1e-4)
    y0 = opt(x0)

    x1 = list(x0)
    x1[1] = x0[1] * 100
    y1 = opt(x1)
    assert y0 != y1

    # Re-running the initial guess should produce the same result
    y0_repeat = opt(x0)
    assert y0 == y0_repeat

