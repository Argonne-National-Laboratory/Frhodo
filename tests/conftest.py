from pathlib import Path
from typing import Tuple

from PyQt5.QtWidgets import QApplication
from pytest import fixture

from frhodo.main import launch_gui, Main


@fixture()
def example_dir() -> Path:
    """Directory containing example input files"""
    return Path(__file__).parent.parent / 'Example'


@fixture()
def frhodo_app() -> Tuple[QApplication, Main]:
    # Launch the application headless
    app, window = launch_gui(['frhodo', '-platform', 'offscreen'], fresh_gui=True)
    yield app, window
    app.quit()
