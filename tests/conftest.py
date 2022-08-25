from pathlib import Path

from pytest import fixture

from frhodo.api.driver import FrhodoDriver
from frhodo.main import launch_gui

# Launch the application headless and without reading the configuration from a past run
app, window = launch_gui(['frhodo', '-platform', 'offscreen'], fresh_gui=True)
driver = FrhodoDriver(window, app)


@fixture()
def example_dir() -> Path:
    """Directory containing example input files"""
    return Path(__file__).parent.parent / 'Example'


@fixture()
def frhodo_driver() -> FrhodoDriver:
    return driver
