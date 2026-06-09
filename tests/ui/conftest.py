"""UI-layer fixtures: Qt platform, isolated config paths, Main() boot."""
import pytest

from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader


@pytest.fixture
def main_with_loaded_mech(main_window, cycloheptane_paths):
    """``Main`` with the bundled Cycloheptane mech loaded and tree set.

    Also seeds ``parent.path["mech"]`` and ``["mech_main"]`` so callers
    that exercise post-load workflows (path-based naming, optimizer
    orchestration) don't ``KeyError`` on missing path entries.
    """
    main_window.mech = MechanismLoader().load(cycloheptane_paths)
    main_window.tree.set_trees(main_window.mech)
    main_window.path["mech"] = cycloheptane_paths["mech"]
    main_window.path["mech_main"] = cycloheptane_paths["mech"].parent

    return main_window


@pytest.fixture
def isolated_path(tmp_path, repo_root):
    """A ``path`` dict suitable for ``Main(app, path)``.

    ``appdata`` (and therefore ``default_config.yaml``) is routed into a
    tmpdir so tests do not read or clobber the user's real config.
    """
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    return {
        "package": repo_root / "src" / "frhodo" / "gui",
        "main": repo_root / "src" / "frhodo",
        "appdata": appdata,
    }


@pytest.fixture
def main_window(qtbot, isolated_path):
    """Boot a fresh ``Main`` window. The fixture handles cleanup via qtbot."""
    from qtpy.QtWidgets import QApplication

    from frhodo import app as main_module

    qapp = QApplication.instance() or QApplication([])
    win = main_module.Main(qapp, isolated_path)
    qtbot.addWidget(win)
    return win
