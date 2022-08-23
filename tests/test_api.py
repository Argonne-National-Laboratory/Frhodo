"""Testing the API components of Frhodo"""

def _load_example(frhodo_driver, example_dir, tmp_path):
    """Set up the driver with a specific problem case"""
    frhodo_driver.load_files(
        example_dir / 'Experiment',
        example_dir / 'Mechanism',
        tmp_path
    )


def test_launch(frhodo_driver):
    """Make sure we can launch Frhodo"""
    assert frhodo_driver.window.isVisible()


def test_load(frhodo_driver, example_dir, tmp_path):
    """Load loading in desired data"""
    _load_example(frhodo_driver, example_dir, tmp_path)
    assert frhodo_driver.n_shocks == 1


def test_observables(frhodo_driver, example_dir, tmp_path):
    _load_example(frhodo_driver, example_dir, tmp_path)
    runs = frhodo_driver.get_observables()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2


def test_simulate(frhodo_driver, example_dir, tmp_path):
    _load_example(frhodo_driver, example_dir, tmp_path)
    runs = frhodo_driver.run_simulations()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2
