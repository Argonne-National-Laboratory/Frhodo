"""``RuntimePaths`` construction + Main.runtime_paths wiring."""
from pathlib import Path

import pytest

from frhodo.gui.runtime_paths import RuntimePaths


pytestmark = pytest.mark.gui


class TestRuntimePathsModel:
    def test_from_package_derives_fields(self, tmp_path):
        package = tmp_path / "frhodo" / "gui"
        package.mkdir(parents=True)
        appdata = tmp_path / "appdata"
        appdata.mkdir()

        rp = RuntimePaths.from_package(package=package, appdata=appdata)

        assert rp.package == package
        assert rp.main == package.parent
        assert rp.appdata == appdata
        assert rp.default_config == appdata / "default_config.yaml"
        assert rp.cantera_mech == appdata / "generated_mech.yaml"
        assert rp.graphics == package / "ui" / "graphics"

    def test_frozen(self, tmp_path):
        rp = RuntimePaths.from_package(package=tmp_path, appdata=tmp_path)
        with pytest.raises(Exception):
            rp.package = Path("/elsewhere")


class TestMainWiring:
    def test_runtime_paths_attached(self, main_window, isolated_path):
        rp = main_window.runtime_paths
        assert isinstance(rp, RuntimePaths)
        assert rp.package == isolated_path["package"]
        assert rp.appdata == isolated_path["appdata"]
        assert rp.default_config == isolated_path["appdata"] / "default_config.yaml"
