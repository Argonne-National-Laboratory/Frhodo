"""``Main.load_mech`` gating of the separate thermodynamics/transport file controls."""
import shutil

import pytest



pytestmark = pytest.mark.gui


CHEMKIN_WITH_THERMO = """\
ELEMENTS H O N END
SPECIES H2 O2 OH END
THERMO ALL
   300.000  1000.000  5000.000
END
REACTIONS
H2 + O2 = OH + OH   1.700E+13   0.000   47780.0
END
"""

# No THERMO block, but a species name carrying the substring "ther": a
# scan that is not anchored to the start of a line reads this as embedded
# thermodynamics.
CHEMKIN_WITHOUT_THERMO = """\
ELEMENTS H O C END
SPECIES H2 O2 OH DIETHYLETHER END
REACTIONS
H2 + O2 = OH + OH   1.700E+13   0.000   47780.0
END
"""

YAML_MECH = """\
phases:
- name: gas
  thermo: ideal-gas
  elements: [H, O]
  species: [H2]
  kinetics: gas
  reactions: none
"""

CTI_MECH = """\
ideal_gas(name='gas', elements='H O', species='H2', reactions='none')
"""


def _write_mech(
    tmp_path, name, text, thermo_names=("therm.therm",), transport_names=()
):
    """A mech directory holding one mech file and the given auxiliary files."""
    mech_dir = tmp_path / "mechs"
    mech_dir.mkdir(exist_ok=True)
    (mech_dir / name).write_text(text)
    for thermo_name in thermo_names:
        (mech_dir / thermo_name).write_text("THERMO ALL\nEND\n")
    for transport_name in transport_names:
        (mech_dir / transport_name).write_text("H2   1   38.000   2.920\n")

    return mech_dir


def _select_mech(main_window, mech_dir, name):
    """Populate the comboboxes from ``mech_dir`` and select ``name``."""
    main_window.path["mech_main"] = mech_dir
    main_window.path_set.mech()

    idx = main_window.mech_select_comboBox.findText(name)
    assert idx >= 0, f"{name} is missing from the mech combobox"
    main_window.mech_select_comboBox.setCurrentIndex(idx)


class TestThermoControlGate:
    def test_yaml_source_disables_the_thermo_control(self, main_window, tmp_path):
        mech_dir = _write_mech(tmp_path, "mech.yaml", YAML_MECH)
        _select_mech(main_window, mech_dir, "mech.yaml")

        main_window.load_mech()

        assert not main_window.use_thermo_file_box.isEnabled(), (
            "YAML carries its own thermo data, the control must be disabled"
        )
        assert main_window.path["thermo"] is None, (
            f'expected thermo path None, got {main_window.path["thermo"]}'
        )

    def test_cti_source_disables_the_thermo_control(self, main_window, tmp_path):
        mech_dir = _write_mech(tmp_path, "mech.cti", CTI_MECH)
        _select_mech(main_window, mech_dir, "mech.cti")

        main_window.load_mech()

        assert not main_window.use_thermo_file_box.isEnabled(), (
            "cti2yaml sources take no separate thermo file"
        )
        assert main_window.path["thermo"] is None, (
            f'expected thermo path None, got {main_window.path["thermo"]}'
        )

    def test_chemkin_with_embedded_thermo_offers_an_optional_file(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(tmp_path, "mech.mech", CHEMKIN_WITH_THERMO)
        _select_mech(main_window, mech_dir, "mech.mech")

        main_window.load_mech()

        assert main_window.use_thermo_file_box.isEnabled(), (
            "a thermo file is optional here, so the control stays available"
        )
        assert not main_window.use_thermo_file_box.isChecked(), (
            "the embedded THERMO block is used unless the user opts out of it"
        )

    def test_checking_the_box_sets_the_thermo_path(self, main_window, tmp_path):
        mech_dir = _write_mech(tmp_path, "mech.mech", CHEMKIN_WITH_THERMO)
        _select_mech(main_window, mech_dir, "mech.mech")
        main_window.load_mech()

        main_window.use_thermo_file_box.setChecked(True)

        assert main_window.use_thermo_file_box.isChecked(), (
            "the reload the click triggers must not undo the opt-in"
        )
        assert main_window.path["thermo"] == mech_dir / "therm.therm", (
            f'expected therm.therm, got {main_window.path["thermo"]}'
        )
        assert main_window.thermo_select_comboBox.isEnabled(), (
            "the file choice must be reachable once the box is checked"
        )

    def test_forced_thermo_selection_is_dropped_by_an_embedded_thermo_mech(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(tmp_path, "bare.mech", CHEMKIN_WITHOUT_THERMO)
        (mech_dir / "embeds.mech").write_text(CHEMKIN_WITH_THERMO)
        _select_mech(main_window, mech_dir, "bare.mech")
        main_window.load_mech()
        assert main_window.use_thermo_file_box.isChecked(), "precondition: forced on"

        _select_mech(main_window, mech_dir, "embeds.mech")
        main_window.load_mech()

        assert not main_window.use_thermo_file_box.isChecked(), (
            "a forced selection is not an opt-in and must not carry over"
        )
        assert main_window.path["thermo"] is None, (
            f'expected thermo path None, got {main_window.path["thermo"]}'
        )

    def test_chemkin_without_embedded_thermo_forces_the_file(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(tmp_path, "mech.mech", CHEMKIN_WITHOUT_THERMO)
        _select_mech(main_window, mech_dir, "mech.mech")

        main_window.load_mech()

        assert main_window.use_thermo_file_box.isChecked(), (
            "without embedded thermodynamics a separate file is mandatory"
        )
        assert not main_window.use_thermo_file_box.isEnabled(), (
            "the mandatory file must not be switchable off"
        )
        assert main_window.path["thermo"] == mech_dir / "therm.therm", (
            f'expected therm.therm, got {main_window.path["thermo"]}'
        )

    def test_chemkin_with_no_thermo_files_disables_an_empty_combobox(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(
            tmp_path, "mech.mech", CHEMKIN_WITH_THERMO, thermo_names=()
        )
        _select_mech(main_window, mech_dir, "mech.mech")

        main_window.load_mech()

        assert not main_window.use_thermo_file_box.isEnabled(), (
            "there is no thermo file to point the control at"
        )

    def test_programmatic_reload_keeps_the_thermo_selection(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(
            tmp_path,
            "mech.mech",
            CHEMKIN_WITH_THERMO,
            thermo_names=("first.therm", "second.therm"),
        )
        _select_mech(main_window, mech_dir, "mech.mech")
        thermo_combo = main_window.thermo_select_comboBox
        thermo_combo.setCurrentIndex(thermo_combo.findText("second.therm"))
        main_window.use_thermo_file_box.setChecked(True)

        main_window.load_mech()

        assert main_window.use_thermo_file_box.isChecked(), (
            "a reload with no sender must not drop the user's opt-in choice"
        )
        assert main_window.path["thermo"] == mech_dir / "second.therm", (
            f'expected second.therm, got {main_window.path["thermo"]}'
        )


class TestTransportControlGate:
    def test_yaml_source_disables_the_transport_control(self, main_window, tmp_path):
        mech_dir = _write_mech(
            tmp_path, "mech.yaml", YAML_MECH, transport_names=("tran.tran",)
        )
        _select_mech(main_window, mech_dir, "mech.yaml")

        main_window.load_mech()

        assert not main_window.use_transport_file_box.isEnabled(), (
            "only the Chemkin converter takes a separate transport file"
        )
        assert main_window.path["transport"] is None, (
            f'expected transport path None, got {main_window.path["transport"]}'
        )

    def test_chemkin_source_offers_an_unchecked_transport_control(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(
            tmp_path,
            "mech.mech",
            CHEMKIN_WITH_THERMO,
            transport_names=("tran.tran",),
        )
        _select_mech(main_window, mech_dir, "mech.mech")

        main_window.load_mech()

        assert main_window.use_transport_file_box.isEnabled(), (
            "a transport file is available, so the control must be offered"
        )
        assert not main_window.use_transport_file_box.isChecked(), (
            "a separate transport file is the uncommon case and must be opted into"
        )
        assert main_window.path["transport"] is None, (
            f'expected transport path None, got {main_window.path["transport"]}'
        )

    def test_checking_the_box_sets_the_transport_path(self, main_window, tmp_path):
        mech_dir = _write_mech(
            tmp_path,
            "mech.mech",
            CHEMKIN_WITH_THERMO,
            transport_names=("tran.tran",),
        )
        _select_mech(main_window, mech_dir, "mech.mech")
        main_window.load_mech()

        main_window.use_transport_file_box.setChecked(True)

        assert main_window.path["transport"] == mech_dir / "tran.tran", (
            f'expected tran.tran, got {main_window.path["transport"]}'
        )
        assert main_window.transport_select_comboBox.isEnabled(), (
            "the file choice must be reachable once the box is checked"
        )

    def test_unchecking_the_box_clears_the_transport_path(self, main_window, tmp_path):
        mech_dir = _write_mech(
            tmp_path,
            "mech.mech",
            CHEMKIN_WITH_THERMO,
            transport_names=("tran.tran",),
        )
        _select_mech(main_window, mech_dir, "mech.mech")
        main_window.use_transport_file_box.setChecked(True)

        main_window.use_transport_file_box.setChecked(False)

        assert main_window.path["transport"] is None, (
            f'expected transport path None, got {main_window.path["transport"]}'
        )
        assert not main_window.transport_select_comboBox.isEnabled()

    def test_chemkin_with_no_transport_files_disables_a_checked_box(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(tmp_path, "mech.mech", CHEMKIN_WITH_THERMO)
        _select_mech(main_window, mech_dir, "mech.mech")
        main_window.use_transport_file_box.setChecked(True)

        main_window.load_mech()

        assert not main_window.use_transport_file_box.isEnabled(), (
            "there is no transport file to point the control at"
        )
        assert main_window.path["transport"] is None, (
            f'expected transport path None, got {main_window.path["transport"]}'
        )

    def test_programmatic_reload_keeps_the_transport_selection(
        self, main_window, tmp_path
    ):
        mech_dir = _write_mech(
            tmp_path,
            "mech.mech",
            CHEMKIN_WITH_THERMO,
            transport_names=("first.tran", "second.tran"),
        )
        _select_mech(main_window, mech_dir, "mech.mech")
        transport_combo = main_window.transport_select_comboBox
        transport_combo.setCurrentIndex(transport_combo.findText("second.tran"))
        main_window.use_transport_file_box.setChecked(True)

        main_window.load_mech()

        assert main_window.use_transport_file_box.isChecked(), (
            "a reload with no sender must not drop the user's choice"
        )
        assert main_window.path["transport"] == mech_dir / "second.tran", (
            f'expected second.tran, got {main_window.path["transport"]}'
        )

    def test_missing_thermo_aborts_the_load_without_a_stale_transport_path(
        self, main_window, tmp_path
    ):
        first_dir = _write_mech(
            tmp_path,
            "mech.mech",
            CHEMKIN_WITH_THERMO,
            transport_names=("tran.tran",),
        )
        _select_mech(main_window, first_dir, "mech.mech")
        main_window.use_transport_file_box.setChecked(True)
        assert main_window.path["transport"] == first_dir / "tran.tran", (
            "precondition: the transport opt-in points at the first directory"
        )

        second_root = tmp_path / "second"
        second_root.mkdir()
        second_dir = _write_mech(
            second_root, "mech.mech", CHEMKIN_WITHOUT_THERMO, thermo_names=()
        )
        _select_mech(main_window, second_dir, "mech.mech")
        main_window.load_mech()

        assert main_window.path["transport"] is None, (
            "the aborted load must leave no transport file from the previous "
            f'directory, got {main_window.path["transport"]}'
        )

    def test_transport_file_reaches_the_loaded_gas(
        self, main_window, tmp_path, mech_fixture_dir
    ):
        mech_dir = tmp_path / "mechs"
        mech_dir.mkdir()
        for name in ("h2o2.mech", "h2o2.therm", "h2o2.tran"):
            shutil.copy(mech_fixture_dir / name, mech_dir / name)
        _select_mech(main_window, mech_dir, "h2o2.mech")

        main_window.use_transport_file_box.setChecked(True)
        main_window.load_mech()

        assert main_window.path["transport"] == mech_dir / "h2o2.tran", (
            f'expected h2o2.tran, got {main_window.path["transport"]}'
        )
        assert main_window.mech.gas.transport_model != "none", (
            "the transport file must give the gas a transport model"
        )
