"""Shared fixtures for the Frhodo test suite.

Qt is forced offscreen for headless runs. The ``frhodo`` package is
resolved via the installed (editable) distribution.
"""
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

import pytest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_MECH_DIR = ROOT / "example" / "mechanism"


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def example_mech_dir():
    return EXAMPLE_MECH_DIR


@pytest.fixture
def cycloheptane_paths(tmp_path):
    """Path dict accepted by ``ChemicalMechanism.load_mechanism``.

    Output YAML lands in a fresh tmp dir per test so the loader's
    write step has no side effects across tests.
    """
    return {
        "mech": EXAMPLE_MECH_DIR / "cycloheptane.mech",
        "thermo": EXAMPLE_MECH_DIR / "cycloheptane.therm",
        "Cantera_Mech": tmp_path / "cyc7.yaml",
    }


@pytest.fixture(scope="module")
def loaded_cycloheptane(tmp_path_factory):
    """Module-scoped loaded mechanism so we don't re-parse Chemkin per test."""
    from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader

    paths = {
        "mech": EXAMPLE_MECH_DIR / "cycloheptane.mech",
        "thermo": EXAMPLE_MECH_DIR / "cycloheptane.therm",
        "Cantera_Mech": tmp_path_factory.mktemp("loaded") / "cyc7.yaml",
    }
    return MechanismLoader().load(paths)


# Synthetic mechanism carrying one reaction per Cantera rate type
# Frhodo supports. Used as the fixture mech for recast / rate-type
# routing tests where the bundled Cycloheptane fixture has gaps.
ALL_RATE_TYPES_TEST_YAML = """
generator: test_all_rate_types
cantera-version: 3.2.0
units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

phases:
  - name: gas
    thermo: ideal-gas
    elements: [H, O, Ar]
    species: [H, O, OH, H2O, H2, AR]
    kinetics: gas

species:
  - name: H
    composition: {H: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[2.5, 0, 0, 0, 0, 25473.66, -0.4466829],
                    [2.5, 0, 0, 0, 0, 25473.66, -0.4466829]]}
  - name: O
    composition: {O: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[3.168, -0.00328, 6.65e-06, -6.13e-09, 2.11e-12, 29122.6, 2.05193],
                    [2.569, -8.6e-05, 4.19e-08, -1.0e-11, 1.22e-15, 29217.6, 4.78434]]}
  - name: OH
    composition: {O: 1, H: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[3.991, -0.00240, 4.617e-06, -3.881e-09, 1.364e-12, 3368.9, -0.103],
                    [2.838, 0.00110, -2.94e-07, 4.21e-11, -2.42e-15, 3704.0, 5.844]]}
  - name: H2O
    composition: {O: 1, H: 2}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[4.198, -0.00203, 6.52e-06, -5.49e-09, 1.77e-12, -30293.7, -0.849],
                    [2.677, 0.00297, -7.74e-07, 9.44e-11, -4.27e-15, -29885.8, 6.881]]}
  - name: H2
    composition: {H: 2}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[2.344, 0.00798, -1.95e-05, 2.01e-08, -7.38e-12, -917.9, 0.683],
                    [3.337, -4.94e-05, 4.99e-07, -1.80e-10, 2.00e-14, -950.2, -3.205]]}
  - name: AR
    composition: {Ar: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[2.5, 0, 0, 0, 0, -745.375, 4.366001],
                    [2.5, 0, 0, 0, 0, -745.375, 4.366001]]}

reactions:
  - equation: H + O <=> OH
    rate-constant: {A: 1.0e+13, b: 0.0, Ea: 0.0}
  - equation: OH + H (+M) <=> H2O (+M)
    type: falloff
    low-P-rate-constant: {A: 1.0e+19, b: -1.5, Ea: 0.0}
    high-P-rate-constant: {A: 1.0e+13, b: 0.0, Ea: 0.0}
  - equation: H2 + O (+M) <=> H + OH (+M)
    type: falloff
    low-P-rate-constant: {A: 1.0e+18, b: -1.0, Ea: 0.0}
    high-P-rate-constant: {A: 5.0e+12, b: 0.0, Ea: 0.0}
    Troe: {A: 0.7, T3: 100.0, T1: 1000.0, T2: 5000.0}
  - equation: OH + OH <=> O + H2O
    type: pressure-dependent-Arrhenius
    rate-constants:
      - {P: 0.01 atm, A: 1.0e+10, b: 0.0, Ea: 1000.0}
      - {P: 1.0 atm, A: 1.0e+12, b: 0.0, Ea: 1000.0}
      - {P: 100.0 atm, A: 1.0e+13, b: 0.0, Ea: 1000.0}
  - equation: H2 + OH <=> H + H2O
    type: Chebyshev
    temperature-range: [300.0, 2000.0]
    pressure-range: [0.001 atm, 100.0 atm]
    data:
      - [8.2883, -1.1397, -0.12059, 0.016034]
      - [1.9764, 1.0037, 0.0072865, -0.030432]
      - [0.31840, 0.36133, 0.14543, -0.026403]
  - equation: H + OH (+M) <=> H2O (+M)
    type: falloff
    low-P-rate-constant: {A: 1.0e+18, b: -0.5, Ea: 0.0}
    high-P-rate-constant: {A: 1.0e+12, b: 0.5, Ea: 0.0}
    SRI: {A: 0.5, B: 1000.0, C: 5000.0}
"""


@pytest.fixture
def loaded_all_rate_types():
    """Per-test in-memory mech carrying one reaction per Cantera rate type.

    Reaction layout (matches index):
      0 — Arrhenius
      1 — Lindemann falloff
      2 — Troe falloff
      3 — Plog
      4 — Chebyshev
      5 — SRI falloff
    """
    import cantera as ct

    from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism

    mech = ChemicalMechanism()
    mech.gas = ct.Solution(yaml=ALL_RATE_TYPES_TEST_YAML)
    mech.set_rate_expression_coeffs()
    mech.set_thermo_expression_coeffs()
    mech.isLoaded = True

    return mech


# Synthetic Chemkin mech with one Arrhenius and one Chebyshev reaction.
# Used to test code paths where the bundled Cycloheptane fixture has
# zero coverage (Chebyshev reactions are not present in Cycloheptane).
CHEBYSHEV_TEST_YAML = """
generator: test_chebyshev
cantera-version: 3.2.0
units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

phases:
  - name: gas
    thermo: ideal-gas
    elements: [H, O, Ar]
    species: [H, O, OH, H2O, AR]
    kinetics: gas

species:
  - name: H
    composition: {H: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[2.5, 0, 0, 0, 0, 25473.66, -0.4466829],
                    [2.5, 0, 0, 0, 0, 25473.66, -0.4466829]]}
  - name: O
    composition: {O: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[3.168, -0.00328, 6.65e-06, -6.13e-09, 2.11e-12, 29122.6, 2.05193],
                    [2.569, -8.6e-05, 4.19e-08, -1.0e-11, 1.22e-15, 29217.6, 4.78434]]}
  - name: OH
    composition: {O: 1, H: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[3.991, -0.00240, 4.617e-06, -3.881e-09, 1.364e-12, 3368.9, -0.103],
                    [2.838, 0.00110, -2.94e-07, 4.21e-11, -2.42e-15, 3704.0, 5.844]]}
  - name: H2O
    composition: {O: 1, H: 2}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[4.198, -0.00203, 6.52e-06, -5.49e-09, 1.77e-12, -30293.7, -0.849],
                    [2.677, 0.00297, -7.74e-07, 9.44e-11, -4.27e-15, -29885.8, 6.881]]}
  - name: AR
    composition: {Ar: 1}
    thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
             data: [[2.5, 0, 0, 0, 0, -745.375, 4.366001],
                    [2.5, 0, 0, 0, 0, -745.375, 4.366001]]}

reactions:
  - equation: H + O <=> OH
    rate-constant: {A: 1.0e+13, b: 0.0, Ea: 0.0}
  - equation: OH + H <=> H2O
    type: Chebyshev
    temperature-range: [300.0, 2000.0]
    pressure-range: [0.001 atm, 100.0 atm]
    data:
      - [8.2883e+00, -1.1397e+00, -1.2059e-01, 1.6034e-02]
      - [1.9764e+00, 1.0037e+00, 7.2865e-03, -3.0432e-02]
      - [3.1840e-01, 3.6133e-01, 1.4543e-01, -2.6403e-02]
"""


@pytest.fixture
def loaded_chebyshev():
    """In-memory ``ChemicalMechanism`` from CHEBYSHEV_TEST_YAML."""
    import cantera as ct

    from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism

    mech = ChemicalMechanism()
    mech.gas = ct.Solution(yaml=CHEBYSHEV_TEST_YAML)
    mech.set_rate_expression_coeffs()
    mech.set_thermo_expression_coeffs()
    mech.isLoaded = True
    return mech


