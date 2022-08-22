from typing import Tuple

from PyQt5.QtWidgets import QApplication
from pytest import fixture

from frhodo.main import launch_gui, Main


@fixture()
def frhodo_app() -> Tuple[QApplication, Main]:
    app, window = launch_gui()
    yield app, window
    app.quit()
