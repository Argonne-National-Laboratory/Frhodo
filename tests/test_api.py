"""Testing the API components of Frhodo"""


def test_launch(frhodo_driver):
    """Make sure we can launch Frhodo"""
    assert frhodo_driver.window.isVisible()


def test_load(frhodo_driver, example_dir, tmp_path):
    """Load loading in desired data"""
    frhodo_driver.load_files(
        example_dir / 'Experiment',
        example_dir / 'Mechanism',
        tmp_path
    )
    assert frhodo_driver.n_shocks == 1


def test_observables(frhodo_driver, example_dir, tmp_path):
    frhodo_driver.load_files(
        example_dir / 'Experiment',
        example_dir / 'Mechanism',
        tmp_path
    )
    runs = frhodo_driver.get_observables()
    assert len(runs) == 1
    assert runs[0].ndim == 2
    assert runs[0].shape[1] == 2
