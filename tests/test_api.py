"""Testing the API components of Frhodo"""
from frhodo.api import load_files


def test_launch(frhodo_app):
    """Make sure we can launch Frhodo"""
    app, window = frhodo_app
    assert window.isVisible()


def test_load(frhodo_app, example_dir, tmp_path):
    """Load loading in desired data"""
    app, window = frhodo_app
    load_files(window, app,
               example_dir / 'Experiment',
               example_dir / 'Mechanism',
               tmp_path)
    assert len(window.series.shock) == 1
