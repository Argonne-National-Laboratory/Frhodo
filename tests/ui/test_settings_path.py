"""``settings.Path``: the mechanism listings, and the directory files that set the folders."""
import configparser
import locale
import shutil

import pytest

from frhodo.gui.widgets.settings.path import _read_directory_file
from frhodo.simulation.mechanism.mechanism_loader import supported_mech_suffixes



pytestmark = pytest.mark.gui

NON_ASCII_FOLDER = "C:\\Users\\José Ω\\Frhodo\\mechanism"


def _items(combobox):
    return {combobox.itemText(i) for i in range(combobox.count())}


def _write_directory_file(path, directories):
    """Write a directory file at ``path`` whose [Directories] section is ``directories``."""
    config = configparser.RawConfigParser()
    config["Experiment Set Name"] = {"name": "Test"}
    config["Species Default Aliases"] = {"aliases": ""}
    config["Directories"] = directories
    with open(path, "w", encoding="utf-8") as f:
        config.write(f)

    return path


@pytest.fixture
def windows_encoding(monkeypatch):
    """Report the system encoding of a default Windows install, where it is not UTF-8."""
    monkeypatch.setattr(locale, "getpreferredencoding", lambda do_setlocale=True: "cp1252")


class TestPathMechListing:
    def test_lists_only_supported_suffixes_and_skips_generated_mech(
        self, main_window, tmp_path
    ):
        mech_dir = tmp_path / "mechs"
        mech_dir.mkdir()

        expected_mech_files = set()
        for suffix in supported_mech_suffixes():
            name = f"mech{suffix}"
            (mech_dir / name).write_text("")
            expected_mech_files.add(name)

        for name in ("generated_mech.yaml", "generated_mech.cti"):
            (mech_dir / name).write_text("")

        (mech_dir / "therm.therm").write_text("")
        (mech_dir / "tran.tran").write_text("")

        main_window.path["mech_main"] = mech_dir
        main_window.path_set.mech()

        listed_mech_files = _items(main_window.mech_select_comboBox)
        assert listed_mech_files == expected_mech_files, (
            f"expected {expected_mech_files}, got {listed_mech_files}"
        )

    def test_skips_mechanisms_written_by_a_conversion(self, main_window, tmp_path):
        mech_dir = tmp_path / "mechs"
        mech_dir.mkdir()
        (mech_dir / "cyc7.mech").write_text("")

        artifacts = (
            "generated_mech.yaml",
            "generated_mech.cti",
            "cyc7.converted.yaml",
            "CYC7.CONVERTED.YAML",
        )
        for name in artifacts:
            (mech_dir / name).write_text("")

        main_window.path["mech_main"] = mech_dir
        main_window.path_set.mech()

        listed = _items(main_window.mech_select_comboBox)
        assert listed == {"cyc7.mech"}, (
            f"conversion output must stay out of the list, got {listed}"
        )

    def test_lists_therm_file_in_thermo_combobox(self, main_window, tmp_path):
        mech_dir = tmp_path / "mechs"
        mech_dir.mkdir()
        (mech_dir / "therm.therm").write_text("")

        main_window.path["mech_main"] = mech_dir
        main_window.path_set.mech()

        assert _items(main_window.thermo_select_comboBox) == {"therm.therm"}

    def test_lists_tran_file_in_transport_combobox_only(self, main_window, tmp_path):
        mech_dir = tmp_path / "mechs"
        mech_dir.mkdir()
        for name in ("mech.mech", "therm.therm", "tran.tran"):
            (mech_dir / name).write_text("")

        main_window.path["mech_main"] = mech_dir
        main_window.path_set.mech()

        assert _items(main_window.transport_select_comboBox) == {"tran.tran"}
        assert _items(main_window.mech_select_comboBox) == {"mech.mech"}, (
            "a .tran file is not a mechanism"
        )
        assert _items(main_window.thermo_select_comboBox) == {"therm.therm"}, (
            "a .tran file is not a thermodynamics file"
        )


class TestReadDirectoryFile:
    def test_utf8_file_reads_intact(self, tmp_path, windows_encoding):
        path = tmp_path / "directories.ini"
        path.write_bytes(f"[Directories]\nmech_main = {NON_ASCII_FOLDER}\n".encode("utf-8"))
        parser = configparser.RawConfigParser()

        _read_directory_file(parser, path)

        read = parser["Directories"]["mech_main"]
        assert read == NON_ASCII_FOLDER, f"got {read!r}"

    def test_file_in_the_system_encoding_still_reads(self, tmp_path, windows_encoding):
        folder = "C:\\Users\\José\\Frhodo\\mechanism"
        path = tmp_path / "directories.ini"
        path.write_bytes(f"[Directories]\nmech_main = {folder}\n".encode("cp1252"))
        parser = configparser.RawConfigParser()

        _read_directory_file(parser, path)

        read = parser["Directories"]["mech_main"]
        assert read == folder, f"got {read!r}"

    def test_byte_order_mark_is_ignored(self, tmp_path):
        path = tmp_path / "directories.ini"
        path.write_bytes("\ufeff[Directories]\nmech_main = mechanism\n".encode("utf-8"))
        parser = configparser.RawConfigParser()

        _read_directory_file(parser, path)

        read = parser["Directories"]["mech_main"]
        assert read == "mechanism", f"got {read!r}"


class TestDirectoryFile:
    def test_non_ascii_folders_from_a_session_restore_load_intact(
        self, main_window, example_mech_dir, tmp_path,
    ):
        project = tmp_path / "José Ω" / "project"
        shutil.copytree(example_mech_dir.parent, project)
        session_file = _write_directory_file(tmp_path / "session.ini", {
            "exp_main": "",
            "mech_main": str(project / "mechanism"),
            "sim_main": str(project / "simulation"),
        })

        main_window.path_set.load_dir_file(session_file)

        loaded = main_window.path["mech_main"]
        assert loaded == project / "mechanism", f"got {loaded}"

    def test_saved_directory_file_is_utf8(self, main_window, example_mech_dir, tmp_path):
        project = tmp_path / "José Ω" / "project"
        shutil.copytree(example_mech_dir.parent, project)
        directory_file = _write_directory_file(tmp_path / "project.ini", {
            "exp_main": str(project / "experiment"),
            "mech_main": str(project / "mechanism"),
            "sim_main": str(project / "simulation"),
        })
        main_window.path_set.load_dir_file(directory_file)
        saved = tmp_path / "saved.ini"

        main_window.path_set.save_dir_file(saved)

        text = saved.read_bytes().decode("utf-8")
        assert "José Ω" in text, f"the saved folders lost their characters: {text!r}"


class TestOptimizedMechNumbering:
    @pytest.mark.parametrize(("current", "existing", "expected"), [
        ("cycloheptane", ["cycloheptane - Opt 1", "cycloheptane - Opt 2"], "cycloheptane - Opt 3"),
        ("H2 (Li) mech", ["H2 (Li) mech - Opt 1", "H2 (Li) mech - Opt 2"], "H2 (Li) mech - Opt 3"),
        ("C7H14+Ar", ["C7H14+Ar - Opt 1"], "C7H14+Ar - Opt 2"),
        ("USC [II]", ["USC [II] - Opt 1"], "USC [II] - Opt 2"),
        ("AramcoMech (v3", ["AramcoMech (v3 - Opt 1"], "AramcoMech (v3 - Opt 2"),
        ("cyc7", ["cyc7 - Opt 2.0 backup"], "cyc7 - Opt 3"),
        ("H2", ["C2H2 - Opt 5"], "H2 - Opt 1"),
        ("cyc7 - Opt 2", ["cyc7 - Opt 1"], "cyc7 - Opt 3"),
    ], ids=[
        "plain", "parentheses", "plus", "brackets", "unbalanced-parenthesis",
        "hand-renamed-copy", "longer-name-ending-in-this-one", "optimizing-an-opt-file",
    ])
    def test_next_file_follows_the_highest_existing_number(
        self, main_window, tmp_path, current, existing, expected,
    ):
        for name in (current, *existing):
            (tmp_path / f"{name}.mech").write_text("")
        main_window.path["mech"] = tmp_path / f"{current}.mech"
        main_window.path["mech_main"] = tmp_path

        chosen = main_window.path_set.optimized_mech().name

        assert chosen == f"{expected}.mech", f"got {chosen!r}"

    @pytest.mark.parametrize(("current", "existing", "expected"), [
        ("cycloheptane", ["cycloheptane - Opt 1"], "cycloheptane - PreOpt 2"),
        ("Optimized H2", [], "Optimized H2 - PreOpt 1"),
        ("H2 Opt-A", ["H2 Opt-A - Opt 2"], "H2 Opt-A - PreOpt 3"),
    ], ids=["plain", "name-starting-with-opt", "name-containing-opt"])
    def test_recast_file_keeps_the_name_and_pairs_with_the_next_number(
        self, main_window, tmp_path, current, existing, expected,
    ):
        for name in (current, *existing):
            (tmp_path / f"{name}.mech").write_text("")
        main_window.path["mech"] = tmp_path / f"{current}.mech"
        main_window.path["mech_main"] = tmp_path

        chosen = main_window.path_set.optimized_mech(file_out="recast_mech").name

        assert chosen == f"{expected}.mech", f"got {chosen!r}"
