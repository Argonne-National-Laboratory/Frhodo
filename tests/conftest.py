from pathlib import Path
from typing import Tuple

from PyQt5.QtWidgets import QApplication
from pytest import fixture

from frhodo.api import FrhodoDriver
from frhodo.main import launch_gui, Main


@fixture()
def example_dir() -> Path:
    """Directory containing example input files"""
    return Path(__file__).parent.parent / 'Example'


@fixture()
def frhodo_driver() -> FrhodoDriver:
    # Launch the application headless
    app, window = launch_gui(['frhodo', '-platform', 'offscreen'], fresh_gui=True)
    yield FrhodoDriver(window, app)
    app.quit()
