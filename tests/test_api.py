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
    assert len(frhodo_driver.window.series.shock) == 1
