"""``settings.Path``: the mechanism listings, and the directory files that set the folders."""
import configparser
import locale
import os
import pathlib
import shutil

import pytest

from frhodo.gui.runtime_paths import RuntimePaths
from frhodo.gui.widgets.settings.path import (
    EXAMPLE_CONFIG,
    _read_directory_file,
    _resolve_dir_entry,
    copy_example,
)
from frhodo.simulation.mechanism.mechanism_loader import supported_mech_suffixes



pytestmark = pytest.mark.gui

EXAMPLE_FOLDERS = {"exp_main": "experiment", "mech_main": "mechanism", "sim_main": "simulation"}
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


def _files_under(folder):
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def _bundle(root):
    """A minimal bundled example: the three named folders plus a walkthrough nothing names."""
    for folder, name in (("experiment", "shock1.exp"), ("mechanism", "mech.mech"),
                         ("simulation", "readme.txt")):
        (root / folder).mkdir(parents=True)
        (root / folder / name).write_text(f"{folder}\n")
    (root / "walkthrough.md").write_text("read me\n")
    _write_directory_file(root / EXAMPLE_CONFIG, EXAMPLE_FOLDERS)

    return root


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


class TestResolveDirEntry:
    def test_relative_entry_reads_against_the_file_folder(self, tmp_path):
        resolved = _resolve_dir_entry(tmp_path, "experiment")

        assert resolved == str(tmp_path / "experiment"), f"got {resolved}"

    def test_absolute_entry_is_kept(self, tmp_path):
        absolute = str(tmp_path / "elsewhere" / "experiment")

        resolved = _resolve_dir_entry(tmp_path / "ini_folder", absolute)

        assert resolved == absolute, f"got {resolved}"

    def test_empty_entry_stays_empty(self, tmp_path):
        resolved = _resolve_dir_entry(tmp_path, "")

        assert resolved == "", f"an unset directory became {resolved!r}"


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


class TestCopyExample:
    def test_copies_only_the_folders_the_directory_file_names(self, tmp_path):
        bundled = _bundle(tmp_path / "bundled")
        destination = tmp_path / "appdata" / "example"

        copied = copy_example(bundled, destination)

        assert copied == destination / EXAMPLE_CONFIG, f"got {copied}"
        expected = {EXAMPLE_CONFIG, "experiment/shock1.exp", "mechanism/mech.mech",
                    "simulation/readme.txt"}
        assert _files_under(destination) == expected, f"got {_files_under(destination)}"

    def test_existing_copy_is_returned_without_overwriting_it(self, tmp_path):
        bundled = _bundle(tmp_path / "bundled")
        destination = tmp_path / "example"
        copy_example(bundled, destination)
        edited = destination / "mechanism" / "mech.mech"
        edited.write_text("edited by the user\n")

        again = copy_example(bundled, destination)

        assert again == destination / EXAMPLE_CONFIG, f"got {again}"
        assert edited.read_text() == "edited by the user\n", "a repeat copy overwrote an edit"

    def test_interrupted_copy_is_completed_without_overwriting(self, tmp_path):
        bundled = _bundle(tmp_path / "bundled")
        destination = tmp_path / "example"
        (destination / "mechanism").mkdir(parents=True)
        already_there = destination / "mechanism" / "mech.mech"
        already_there.write_text("already here\n")

        copied = copy_example(bundled, destination)

        assert copied.is_file(), "the directory file was not written"
        assert (destination / "experiment" / "shock1.exp").is_file(), "a folder was not copied"
        assert already_there.read_text() == "already here\n", "an existing file was overwritten"

    def test_empty_entries_copy_no_folder(self, tmp_path):
        # An empty entry must not read as the bundle itself, which would copy all of it.
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "walkthrough.md").write_text("read me\n")
        _write_directory_file(
            bundled / EXAMPLE_CONFIG, {"exp_main": "", "mech_main": "", "sim_main": ""},
        )
        destination = tmp_path / "example"

        copy_example(bundled, destination)

        assert _files_under(destination) == {EXAMPLE_CONFIG}, f"got {_files_under(destination)}"

    def test_absolute_entries_are_not_copied(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "shock1.exp").write_text("x\n")
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        _write_directory_file(
            bundled / EXAMPLE_CONFIG,
            {"exp_main": str(outside), "mech_main": "", "sim_main": ""},
        )
        destination = tmp_path / "example"

        copy_example(bundled, destination)

        assert _files_under(destination) == {EXAMPLE_CONFIG}, f"got {_files_under(destination)}"

    def test_copy_of_a_read_only_install_is_writable(self, tmp_path):
        bundled = _bundle(tmp_path / "bundled")
        for file in bundled.rglob("*"):
            if file.is_file():
                file.chmod(0o444)
        destination = tmp_path / "example"

        copy_example(bundled, destination)

        unwritable = sorted(name for name in _files_under(destination)
                            if not os.access(destination / name, os.W_OK))
        assert unwritable == [], f"copied without write permission: {unwritable}"

    def test_copy_killed_partway_leaves_no_file_that_looks_finished(self, tmp_path, monkeypatch):
        bundled = _bundle(tmp_path / "bundled")
        destination = tmp_path / "example"

        def killed_partway(source, target):
            pathlib.Path(target).write_text("half")
            raise OSError("disk full")

        monkeypatch.setattr(shutil, "copyfile", killed_partway)

        with pytest.raises(OSError):
            copy_example(bundled, destination)

        finished = sorted(name for name in _files_under(destination)
                          if not name.endswith(".part"))
        assert finished == [], f"an interrupted copy left finished-looking files: {finished}"

    def test_leftover_part_file_is_replaced_by_the_full_copy(self, tmp_path):
        bundled = _bundle(tmp_path / "bundled")
        destination = tmp_path / "example"
        (destination / "mechanism").mkdir(parents=True)
        (destination / "mechanism" / "mech.mech.part").write_text("half")

        copy_example(bundled, destination)

        copied = (destination / "mechanism" / "mech.mech").read_text()
        assert copied == "mechanism\n", f"got {copied!r}"

    def test_install_without_an_example_returns_none(self, tmp_path):
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        destination = tmp_path / "example"

        copied = copy_example(bundled, destination)

        assert copied is None, f"got {copied}"
        assert not destination.exists(), "a folder was created with nothing to copy"


class TestShippedExample:
    @pytest.mark.parametrize("key", sorted(EXAMPLE_FOLDERS))
    def test_directory_file_names_a_relative_folder_that_ships(self, example_dir, key):
        config = configparser.RawConfigParser()
        config.read(example_dir / EXAMPLE_CONFIG)
        entry = config["Directories"][key]

        assert entry and not pathlib.Path(entry).is_absolute(), f"{key} = {entry!r} is not relative"
        assert (example_dir / entry).is_dir(), f"{key} names {entry!r}, which does not ship"


class TestDirectoryFile:
    def test_relative_entries_resolve_against_the_file_not_the_working_directory(
        self, main_window, example_dir, tmp_path, monkeypatch,
    ):
        directory_file = copy_example(example_dir, tmp_path / "project")
        # Read against the working directory instead, the entries would land in tmp_path.
        monkeypatch.chdir(tmp_path)

        main_window.path_set.load_dir_file(directory_file)

        project = directory_file.parent.resolve()
        loaded = {key: main_window.path[key] for key in EXAMPLE_FOLDERS}
        expected = {key: project / folder for key, folder in EXAMPLE_FOLDERS.items()}
        assert loaded == expected, f"got {loaded}"

    def test_alias_save_after_loading_another_file_keeps_this_files_settings(
        self, main_window, example_dir, tmp_path,
    ):
        directory_file = copy_example(example_dir, tmp_path / "project")
        main_window.path_set.load_dir_file(directory_file)
        # A session restore reads a file of its own through load_dir_file.
        session = tmp_path / "session"
        session.mkdir()
        session_file = _write_directory_file(session / EXAMPLE_CONFIG, {
            "exp_main": "",
            "mech_main": str(directory_file.parent / "mechanism"),
            "sim_main": str(tmp_path / "elsewhere"),
        })
        main_window.path_set.load_dir_file(session_file)

        main_window.path_set.save_aliases(directory_file)

        config = configparser.RawConfigParser()
        config.read(directory_file)
        saved = dict(config["Directories"])
        assert saved == EXAMPLE_FOLDERS, f"the file took the session's folders: {saved}"
        name = config["Experiment Set Name"]["name"]
        assert name == "Example Input Files", f"the file took the session's set name: {name!r}"

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


    def test_alias_save_keeps_the_entries_relative(self, main_window, example_dir, tmp_path):
        directory_file = copy_example(example_dir, tmp_path / "project")
        main_window.path_set.load_dir_file(directory_file)

        main_window.path_set.save_aliases(directory_file)

        config = configparser.RawConfigParser()
        config.read(directory_file)
        saved = dict(config["Directories"])
        assert saved == EXAMPLE_FOLDERS, f"a save made the copy location-bound: {saved}"


class TestOpenExample:
    def test_install_without_the_example_is_logged(self, main_window, tmp_path):
        main_window.runtime_paths = RuntimePaths.from_package(
            package=tmp_path / "gui", appdata=tmp_path / "appdata",
        )

        main_window.path_set.open_example()

        log_text = main_window.log.log.toPlainText()
        assert "No example project was found" in log_text, f"log was: {log_text!r}"
        assert main_window.path_file_box.toPlainText() == "", "a directory file was adopted"


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
