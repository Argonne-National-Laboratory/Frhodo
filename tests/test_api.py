"""Testing the API components of Frhodo"""
import numpy as np
from pytest import fixture


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


def test_launch(frhodo_driver):
    """Make sure we can launch Frhodo"""
    assert frhodo_driver.window.isVisible()


def test_load(loaded_frhodo):
    """Load loading in desired data"""
    assert loaded_frhodo.n_shocks == 1


def test_observables(loaded_frhodo):
    runs = loaded_frhodo.get_observables()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2


def test_simulate(loaded_frhodo):
    runs = loaded_frhodo.run_simulations()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2

    # Make sure the aliases propagated through
    #  They are reloaded from configuration variables when we re-run a simulation
    assert loaded_frhodo.window.display_shock['species_alias']['A'] == 'B'


def test_update(loaded_frhodo):
    # Get the initial reaction rates
    rates = loaded_frhodo.get_reaction_rates()
    assert rates.shape == (66, 1)

    # Get the simulation before
    sim = loaded_frhodo.run_simulations()[0]

    # Update one of the rates
    rxn_id = 3  # A simple reaction
    loaded_frhodo.change_coefficient({(rxn_id, 0, 'pre_exponential_factor'): 200})
    assert loaded_frhodo.rxn_coeffs[rxn_id][0]['pre_exponential_factor'] == 200

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
