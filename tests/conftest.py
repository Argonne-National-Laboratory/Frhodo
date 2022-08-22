import sys
from typing import Tuple

from PyQt5.QtWidgets import QApplication
from pytest import fixture

from frhodo.main import launch_gui, Main


@fixture()
def frhodo_app() -> Tuple[QApplication, Main]:
    # Launch the application headless
    app, window = launch_gui(['frhodo', '-platform', 'offscreen'])
    yield app, window
    app.quit()
