"""Tests for ``frhodo.gui.runtime_paths``.

``RuntimePaths`` is the source-of-truth for every file-system location
the GUI uses; downstream code (capture hook, mechanism cache, default
config persistence) reads its fields. A drifted derivation here would
silently route data to the wrong place.
"""
import pytest
from pydantic import ValidationError

from frhodo.gui.runtime_paths import RuntimePaths


class TestRuntimePathsFromPackage:
    """``from_package`` derives every path from ``package`` and ``appdata``
    so callers cannot independently set the derived fields and let them
    drift apart."""

    def test_appdata_passed_through_unchanged(self, tmp_path):
        appdata = tmp_path / "appdata"
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=appdata,
        )
        assert paths.appdata == appdata

    def test_package_passed_through_unchanged(self, tmp_path):
        package = tmp_path / "gui"
        paths = RuntimePaths.from_package(package=package, appdata=tmp_path)
        assert paths.package == package

    def test_main_is_parent_of_package(self, tmp_path):
        """``main`` is the frhodo package root (parent of ``frhodo/gui``)."""
        package = tmp_path / "frhodo" / "gui"
        paths = RuntimePaths.from_package(package=package, appdata=tmp_path)
        assert paths.main == tmp_path / "frhodo"

    def test_default_config_under_appdata(self, tmp_path):
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )
        assert paths.default_config == tmp_path / "appdata" / "default_config.yaml"

    def test_cantera_mech_cache_under_appdata(self, tmp_path):
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )
        assert paths.cantera_mech == tmp_path / "appdata" / "generated_mech.yaml"

    def test_graphics_under_package_ui(self, tmp_path):
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path,
        )
        assert paths.graphics == tmp_path / "gui" / "ui" / "graphics"

    def test_troe_captures_under_appdata(self, tmp_path):
        """Capture-hook default destination — moving this field would
        silently re-route NN training data collection."""
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )
        assert paths.troe_captures == tmp_path / "appdata" / "troe_captures"


class TestRuntimePathsImmutability:
    """``RuntimePaths`` is ``frozen=True`` — once built, callers can't
    re-point any field. Prevents accidental mid-session drift between
    related paths."""

    def test_assignment_to_appdata_raises(self, tmp_path):
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )
        with pytest.raises(ValidationError):
            paths.appdata = tmp_path / "other"

    def test_assignment_to_troe_captures_raises(self, tmp_path):
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )
        with pytest.raises(ValidationError):
            paths.troe_captures = tmp_path / "alt_captures"


class TestRuntimePathsFieldShape:
    """Every required field must be present so a typo doesn't surface
    only when a downstream feature finally reads it."""

    @pytest.mark.parametrize("field", [
        "package", "main", "appdata", "default_config",
        "cantera_mech", "graphics", "troe_captures",
    ])
    def test_field_populated_after_from_package(self, tmp_path, field):
        paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )
        value = getattr(paths, field)
        assert value is not None, f"{field} should be set"
