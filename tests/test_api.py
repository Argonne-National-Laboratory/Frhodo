"""Testing the API components of Frhodo"""
import numpy as np
from pytest import fixture, mark

from frhodo.api import FrhodoDriver


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

    # Update one of the rates
    loaded_frhodo.change_coefficient({(rxn_id, prs_id, 'pre_exponential_factor'): 200})
    assert loaded_frhodo.rxn_coeffs[rxn_id][prs_id]['pre_exponential_factor'] == 200

    # Make sure that only one rate changes
    new_rates = loaded_frhodo.get_reaction_rates()
    changes = np.abs(rates - new_rates)
    assert changes[rxn_id, 0] > 0
    mask = np.ones_like(rates, bool)
    mask[rxn_id, 0] = False
    assert np.isclose(changes[mask], 0).all()

    # Make sure that this changes the simulation
    new_sim = loaded_frhodo.run_simulations()[0]
    assert not np.isclose(sim[-1, 1], new_sim[-1, 1], rtol=1e-4)  # Look just at the last point


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
