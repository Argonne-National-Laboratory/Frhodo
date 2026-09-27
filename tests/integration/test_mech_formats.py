"""``FORMAT_TABLE`` / ``supported_mech_suffixes`` -- the single source of
truth for mechanism-format dispatch shared by the loader, the GUI listing,
and ``api.py``.
"""
import logging
import os
import shutil
import sys
from pathlib import Path

import cantera as ct
import pytest
import yaml

from frhodo.common.errors import MechanismLoadError
from frhodo.simulation.mechanism.mechanism_loader import (
    FORMAT_TABLE,
    MechanismLoader,
    supported_mech_suffixes,
)



CANTERA_DATA_DIR = Path(ct.__path__[0]) / "data"


class TestSupportedMechSuffixes:
    def test_matches_format_table_keys(self):
        assert set(supported_mech_suffixes()) == set(FORMAT_TABLE)

    def test_every_suffix_is_lowercase_and_dot_prefixed(self):
        for suffix in supported_mech_suffixes():
            assert suffix.startswith("."), f"{suffix!r} is not dot-prefixed"
            assert suffix == suffix.lower(), f"{suffix!r} is not lowercase"

    def test_dat_and_chem_are_absent(self):
        assert ".dat" not in supported_mech_suffixes()
        assert ".chem" not in supported_mech_suffixes()

    def test_lxcat2yaml_kind_is_absent(self):
        assert "lxcat2yaml" not in FORMAT_TABLE.values()


class TestFormatTableKinds:
    def test_native_suffixes(self):
        assert FORMAT_TABLE[".yaml"] == "native"
        assert FORMAT_TABLE[".yml"] == "native"

    def test_cti_suffix(self):
        assert FORMAT_TABLE[".cti"] == "cti2yaml"

    def test_ctml_and_xml_suffixes(self):
        assert FORMAT_TABLE[".ctml"] == "ctml2yaml"
        assert FORMAT_TABLE[".xml"] == "ctml2yaml"

    def test_chemkin_suffixes(self):
        assert FORMAT_TABLE[".inp"] == "chemkin"
        assert FORMAT_TABLE[".ck"] == "chemkin"
        assert FORMAT_TABLE[".mech"] == "chemkin"


class TestYamlSourceNotAliased:
    """A native YAML source must load itself, not whatever is already
    sitting at ``Cantera_Mech`` from a prior conversion.
    """

    @staticmethod
    def _stale_cantera_mech(tmp_path):
        """A ``Cantera_Mech`` target pre-seeded with gri30 (53 species),
        standing in for leftover state from a previous session's load.
        """
        stale = tmp_path / "generated_mech.yaml"
        shutil.copy(CANTERA_DATA_DIR / "gri30.yaml", stale)

        return stale

    def test_yaml_source_loads_its_own_species_not_the_stale_target(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "h2o2.yaml"
        shutil.copy(CANTERA_DATA_DIR / "h2o2.yaml", source)

        stale = self._stale_cantera_mech(tmp_path)
        paths = {"mech": source, "thermo": None, "Cantera_Mech": stale}

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == 10

    def test_yml_source_loads_its_own_species_not_the_stale_target(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "h2o2.yml"
        shutil.copy(CANTERA_DATA_DIR / "h2o2.yaml", source)

        stale = self._stale_cantera_mech(tmp_path)
        paths = {"mech": source, "thermo": None, "Cantera_Mech": stale}

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == 10

    def test_yaml_source_does_not_write_the_stale_target(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "h2o2.yaml"
        shutil.copy(CANTERA_DATA_DIR / "h2o2.yaml", source)

        stale = self._stale_cantera_mech(tmp_path)
        original_content = stale.read_text()
        paths = {"mech": source, "thermo": None, "Cantera_Mech": stale}

        MechanismLoader().load(paths)

        assert stale.read_text() == original_content

    def test_yaml_source_omits_wrote_message(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "h2o2.yaml"
        shutil.copy(CANTERA_DATA_DIR / "h2o2.yaml", source)

        stale = self._stale_cantera_mech(tmp_path)
        paths = {"mech": source, "thermo": None, "Cantera_Mech": stale}

        loader = MechanismLoader()
        loader.load(paths)

        assert not any("Wrote YAML mechanism file" in m for m in loader.messages)

    def test_chemkin_source_includes_wrote_message(self, h2o2_chemkin_paths):
        loader = MechanismLoader()
        loader.load(h2o2_chemkin_paths)

        assert any("Wrote YAML mechanism file" in m for m in loader.messages)

    def test_yaml_source_with_cross_file_species_reference_loads(self, tmp_path):
        """A native YAML source that includes another file's species (a
        relative reference resolved against the source's own directory,
        not the current working directory) must still load.
        """
        source_doc = yaml.safe_load((CANTERA_DATA_DIR / "h2o2.yaml").read_text())
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "species_db.yaml").write_text(
            yaml.safe_dump({"species": source_doc["species"]}, sort_keys=False)
        )

        phase = dict(source_doc["phases"][0])
        phase["species"] = [{"species_db.yaml/species": "all"}]
        phase.pop("transport", None)
        main_doc = {
            "units": source_doc["units"],
            "phases": [phase],
            "reactions": source_doc["reactions"],
        }
        source = source_dir / "main.yaml"
        source.write_text(yaml.safe_dump(main_doc, sort_keys=False))

        paths = {"mech": source, "thermo": None, "Cantera_Mech": tmp_path / "target.yaml"}

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == 10


class TestSuffixCaseInsensitivity:
    def test_uppercase_yaml_suffix_loads(self, tmp_path):
        source = tmp_path / "UPPER.YAML"
        shutil.copy(CANTERA_DATA_DIR / "h2o2.yaml", source)
        paths = {"mech": source, "thermo": None, "Cantera_Mech": tmp_path / "target.yaml"}

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == 10

    def test_mixed_case_chemkin_suffix_loads_same_species_count_as_lowercase(
        self, tmp_path, mech_fixture_dir,
    ):
        reference_paths = {
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "reference.yaml",
        }
        reference = MechanismLoader().load(reference_paths)

        mixed_case_source = tmp_path / "h2o2.MECH"
        shutil.copy(mech_fixture_dir / "h2o2.mech", mixed_case_source)
        mixed_case_paths = {
            "mech": mixed_case_source,
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "target.yaml",
        }

        mech = MechanismLoader().load(mixed_case_paths)

        assert mech.gas.n_species == reference.gas.n_species


class TestUnsupportedSuffixRaises:
    def test_unsupported_suffix_names_itself_in_the_error(self, tmp_path, mech_fixture_dir):
        source = tmp_path / "mech.txt"
        shutil.copy(mech_fixture_dir / "h2o2.mech", source)
        paths = {
            "mech": source,
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "target.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        message = str(excinfo.value)
        assert ".txt" in message
        assert "not a supported mechanism format" in message

    def test_unsupported_suffix_does_not_leak_a_cantera_parse_error(
        self, tmp_path, mech_fixture_dir,
    ):
        source = tmp_path / "mech.txt"
        shutil.copy(mech_fixture_dir / "h2o2.mech", source)
        paths = {
            "mech": source,
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "target.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        message = str(excinfo.value)
        assert "index" not in message
        assert "out of bounds" not in message


class TestConverterDiagnostics:
    """Converter output reaches the caller, a converter that calls
    ``sys.exit`` surfaces as a load error, and neither outcome leaves
    scratch files or logging handlers behind.
    """

    @staticmethod
    def _garbage_cti(tmp_path):
        """A ``.cti`` file whose first line is a syntax error, which
        ``cti2yaml`` reports before calling ``sys.exit``.
        """
        source = tmp_path / "garbage.cti"
        source.write_text("total garbage !!! (((\n")

        return source

    @staticmethod
    def _undeclared_species_chemkin(tmp_path):
        """A Chemkin mech whose one reaction names a species absent from
        its ``SPECIES`` block.
        """
        source = tmp_path / "undeclared.mech"
        source.write_text(
            "ELEMENTS\nH O\nEND\n"
            "SPECIES\nH2 O2 H2O\nEND\n"
            "REACTIONS\n"
            "H2 + O2 <=> NOTASPECIES + H2O   1.0E13  0.0  1000.0\n"
            "END\n"
        )

        return source

    @staticmethod
    def _thermo_with_redundant_entry(tmp_path, mech_fixture_dir):
        """The H2/O2 thermo file with its argon block repeated, which
        ``ck2yaml`` converts successfully while warning about the repeat.
        """
        lines = (mech_fixture_dir / "h2o2.therm").read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("AR "))
        end = next(i for i, line in enumerate(lines) if line.strip().upper() == "END")
        repeated = lines[:end] + lines[start:start + 4] + lines[end:]

        target = tmp_path / "redundant.therm"
        target.write_text("\n".join(repeated) + "\n")

        return target

    @staticmethod
    def _unflagged_duplicate_reaction_chemkin(tmp_path, mech_fixture_dir):
        """The H2/O2 Chemkin mech with one reaction repeated verbatim and
        no ``DUPLICATE`` marker. ``ck2yaml`` converts it without complaint
        (duplicate-reaction checking happens later, in ``ct.Solution``),
        so it is a case where the converter has diagnostics to report but
        raises nothing itself.
        """
        lines = (mech_fixture_dir / "h2o2.mech").read_text().splitlines()
        repeat_at = next(i for i, line in enumerate(lines) if line.startswith("H2 + O <=>"))
        lines.insert(repeat_at + 1, lines[repeat_at])

        source = tmp_path / "duplicate.mech"
        source.write_text("\n".join(lines) + "\n")

        return source

    def test_ct_solution_failure_after_conversion_carries_converter_diagnostics(
        self, tmp_path, mech_fixture_dir,
    ):
        paths = {
            "mech": self._unflagged_duplicate_reaction_chemkin(tmp_path, mech_fixture_dir),
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        message = str(excinfo.value)
        assert "Undeclared duplicate reactions" in message
        assert "Wrote YAML mechanism file" in message

    def test_malformed_cti_raises_load_error_not_system_exit(self, tmp_path):
        paths = {
            "mech": self._garbage_cti(tmp_path),
            "thermo": None,
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        assert "SyntaxError" in str(excinfo.value)

    def test_malformed_chemkin_error_carries_the_converter_diagnostic(self, tmp_path):
        paths = {
            "mech": self._undeclared_species_chemkin(tmp_path),
            "thermo": None,
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        message = str(excinfo.value)
        assert "Unexpected token" in message
        assert "NOTASPECIES" in message

    def test_ck2yaml_warning_reaches_loader_messages(self, tmp_path, mech_fixture_dir):
        paths = {
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": self._thermo_with_redundant_entry(tmp_path, mech_fixture_dir),
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }

        loader = MechanismLoader()
        loader.load(paths)

        assert any("Ignoring redundant thermo data" in m for m in loader.messages), (
            f"converter warning missing from {loader.messages}"
        )

    def test_silent_loader_keeps_converter_warnings_out_of_messages(
        self, tmp_path, mech_fixture_dir,
    ):
        paths = {
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": self._thermo_with_redundant_entry(tmp_path, mech_fixture_dir),
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }

        loader = MechanismLoader(silent=True)
        loader.load(paths)

        assert loader.messages == []

    def test_failed_conversion_leaves_no_tmp_file(self, tmp_path):
        target_dir = tmp_path / "appdata"
        paths = {
            "mech": self._garbage_cti(tmp_path),
            "thermo": None,
            "Cantera_Mech": target_dir / "generated_mech.yaml",
        }

        with pytest.raises(MechanismLoadError):
            MechanismLoader().load(paths)

        assert list(target_dir.glob("*.tmp")) == []

    def test_successful_load_leaves_no_handler_on_the_cantera_logger(
        self, tmp_path, mech_fixture_dir,
    ):
        paths = {
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }
        cantera_logger = logging.getLogger("cantera")
        before = list(cantera_logger.handlers)

        MechanismLoader().load(paths)

        assert cantera_logger.handlers == before

    def test_failed_load_leaves_no_handler_on_the_cantera_logger(self, tmp_path):
        paths = {
            "mech": self._garbage_cti(tmp_path),
            "thermo": None,
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }
        cantera_logger = logging.getLogger("cantera")
        before = list(cantera_logger.handlers)

        with pytest.raises(MechanismLoadError):
            MechanismLoader().load(paths)

        assert cantera_logger.handlers == before


class TestLoadPerFormatMatchesSourceCounts:
    """Renaming a source file to any other suffix of the same dispatch kind
    must not change the parsed species/reaction counts.
    """

    @staticmethod
    def _reference_and_source(suffix, tmp_path, mech_fixture_dir):
        kind = FORMAT_TABLE[suffix]
        if kind == "native":
            source = CANTERA_DATA_DIR / "h2o2.yaml"
            thermo = None
        elif kind == "cti2yaml":
            source = mech_fixture_dir / "mini.cti"
            thermo = None
        elif kind == "ctml2yaml":
            source = mech_fixture_dir / "mini.ctml"
            thermo = None
        else:  # chemkin
            source = mech_fixture_dir / "h2o2.mech"
            thermo = mech_fixture_dir / "h2o2.therm"

        reference = MechanismLoader().load({
            "mech": source, "thermo": thermo, "Cantera_Mech": tmp_path / "reference.yaml",
        })

        return reference, source, thermo

    @pytest.mark.parametrize("suffix", sorted(supported_mech_suffixes()))
    def test_renamed_source_matches_reference_counts(self, suffix, tmp_path, mech_fixture_dir):
        reference, source, thermo = self._reference_and_source(suffix, tmp_path, mech_fixture_dir)

        renamed_source = tmp_path / f"copy{suffix}"
        shutil.copy(source, renamed_source)
        paths = {
            "mech": renamed_source, "thermo": thermo,
            "Cantera_Mech": tmp_path / f"target{suffix}.yaml",
        }

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == reference.gas.n_species
        assert mech.gas.n_reactions == reference.gas.n_reactions


class TestTransportPassThrough:
    """``paths["transport"]`` reaches ``ck2yaml`` and survives round-trip export."""

    def test_transport_file_sets_a_transport_model(self, h2o2_transport_paths):
        mech = MechanismLoader().load(h2o2_transport_paths)

        assert mech.gas.transport_model != "none"

    def test_no_transport_file_leaves_transport_model_absent(self, h2o2_chemkin_paths):
        mech = MechanismLoader().load(h2o2_chemkin_paths)

        assert mech.gas.transport_model in (None, "none")

    def test_transport_survives_export_to_chemkin(self, h2o2_transport_paths, tmp_path):
        mech = MechanismLoader().load(h2o2_transport_paths)
        out = tmp_path / "out.mech"

        mech.to_chemkin(out)

        assert "TRANSPORT" in out.read_text()

    def test_transport_file_missing_a_species_raises_with_converter_diagnostic(
        self, tmp_path, mech_fixture_dir,
    ):
        lines = (mech_fixture_dir / "h2o2.tran").read_text().splitlines()
        missing_species = [line for line in lines if not line.strip().upper().startswith("H2 ")]
        bad_transport = tmp_path / "missing_species.tran"
        bad_transport.write_text("\n".join(missing_species) + "\n")

        paths = {
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": mech_fixture_dir / "h2o2.therm",
            "transport": bad_transport,
            "Cantera_Mech": tmp_path / "out" / "generated_mech.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        message = str(excinfo.value)
        assert "No transport data for species 'H2'" in message

    def test_chemkin_source_with_no_transport_key_still_loads(self, tmp_path, mech_fixture_dir):
        paths = {
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": tmp_path / "target.yaml",
        }

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species > 0


class TestCtmlFormat:
    """CTML/XML sources dispatch through ``ctml2yaml``."""

    def test_ctml_source_loads_two_species_zero_reactions(self, tmp_path, mech_fixture_dir):
        paths = {
            "mech": mech_fixture_dir / "mini.ctml",
            "thermo": None,
            "Cantera_Mech": tmp_path / "target.yaml",
        }

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == 2
        assert mech.gas.n_reactions == 0

    def test_xml_suffix_alias_loads_same_counts_as_ctml(self, tmp_path, mech_fixture_dir):
        source = tmp_path / "mini.xml"
        shutil.copy(mech_fixture_dir / "mini.ctml", source)
        paths = {
            "mech": source,
            "thermo": None,
            "Cantera_Mech": tmp_path / "target.yaml",
        }

        mech = MechanismLoader().load(paths)

        assert mech.gas.n_species == 2
        assert mech.gas.n_reactions == 0

    def test_malformed_ctml_raises_load_error_not_a_parse_error(self, tmp_path):
        source = tmp_path / "garbage.ctml"
        source.write_text("total garbage !!! (((\n")
        target_dir = tmp_path / "out"
        paths = {
            "mech": source,
            "thermo": None,
            "Cantera_Mech": target_dir / "generated_mech.yaml",
        }

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader().load(paths)

        assert "syntax error" in str(excinfo.value)
        assert list(target_dir.glob("*.tmp")) == []


class TestSolutionSource:
    """Converted mechanisms parse from text in memory; a native YAML source opens by path."""

    def test_reconverting_into_the_same_target_loads_the_new_mechanism(
        self, tmp_path, mech_fixture_dir, example_mech_dir, monkeypatch,
    ):
        # Pin every conversion to one timestamp, as a coarse filesystem clock would for
        # two quick loads. Cantera caches the files it opens by name and modification time.
        real_replace = os.replace

        def replace_on_one_tick(source, target):
            real_replace(source, target)
            os.utime(target, ns=(0, 0))

        monkeypatch.setattr(os, "replace", replace_on_one_tick)
        target = tmp_path / "generated_mech.yaml"
        MechanismLoader(silent=True).load({
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": target,
        })

        second = MechanismLoader(silent=True).load({
            "mech": example_mech_dir / "cycloheptane.mech",
            "thermo": example_mech_dir / "cycloheptane.therm",
            "Cantera_Mech": target,
        })

        assert second.gas.n_species == 36, (
            f"got the earlier conversion back: {second.gas.n_species} species"
        )

    def test_converted_mechanism_in_a_non_ascii_folder_loads(self, tmp_path, mech_fixture_dir):
        target = tmp_path / "José Ω" / "generated_mech.yaml"

        mech = MechanismLoader(silent=True).load({
            "mech": mech_fixture_dir / "h2o2.mech",
            "thermo": mech_fixture_dir / "h2o2.therm",
            "Cantera_Mech": target,
        })

        assert mech.gas.n_species == 10, f"got {mech.gas.n_species} species"

    @pytest.mark.xfail(
        sys.platform == "win32", strict=True,
        reason="Cantera opens files through narrow paths, which Windows reads in its ANSI "
               "code page",
    )
    def test_native_yaml_in_a_non_ascii_folder_loads(self, tmp_path):
        folder = tmp_path / "José Ω"
        folder.mkdir()
        shutil.copy(CANTERA_DATA_DIR / "h2o2.yaml", folder / "h2o2.yaml")

        mech = MechanismLoader(silent=True).load({
            "mech": folder / "h2o2.yaml", "thermo": None, "Cantera_Mech": tmp_path / "t.yaml",
        })

        assert mech.gas.n_species == 10, f"got {mech.gas.n_species} species"

    @pytest.mark.parametrize(("folder_name", "noted"), [("José Ω", True), ("plain", False)])
    def test_windows_failure_notes_a_non_ascii_native_path(
        self, tmp_path, monkeypatch, folder_name, noted,
    ):
        monkeypatch.setattr("frhodo.simulation.mechanism.mechanism_loader._ON_WINDOWS", True)
        folder = tmp_path / folder_name
        folder.mkdir()
        (folder / "broken.yaml").write_text("phases: [\n")

        with pytest.raises(MechanismLoadError) as excinfo:
            MechanismLoader(silent=True).load({
                "mech": folder / "broken.yaml", "thermo": None,
                "Cantera_Mech": tmp_path / "t.yaml",
            })

        message = str(excinfo.value)
        assert ("non-ASCII characters" in message) is noted, f"message was: {message!r}"
