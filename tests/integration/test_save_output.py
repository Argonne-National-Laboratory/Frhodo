"""``save_output.Save.make_table`` and ``write_table``."""
import types

import numpy as np
import pytest

from frhodo.gui.widgets.save_output import Save


@pytest.fixture
def save_instance(tmp_path):
    """``Save.__init__`` only reads ``parent.path``; a stub is sufficient."""
    parent = types.SimpleNamespace(path={"appdata": tmp_path})
    return Save(parent)


class TestMakeTable:
    LABELS = ["Time (μs)", "Density"]
    DATA = np.array([
        [1.0, 2.0, 3.0],          # time column
        [0.5, 0.55, 0.6],         # density column
    ])

    def test_returns_list_of_strings(self, save_instance):
        result = save_instance.make_table(self.LABELS, self.DATA)
        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result), (
            f"all rows should be strings; got types {set(type(r).__name__ for r in result)}"
        )

    def test_has_one_row_per_data_point_plus_header_and_separator(self, save_instance):
        """Output is: [header, separator, *data_rows]."""
        result = save_instance.make_table(self.LABELS, self.DATA)
        assert len(result) == 2 + self.DATA.shape[1], (
            f"expected {2 + self.DATA.shape[1]} rows (header + sep + {self.DATA.shape[1]} data), "
            f"got {len(result)}"
        )

    def test_sig_fig_controls_decimal_places(self, save_instance):
        """sig_fig=1 should yield "x.ye+nn" (one digit after decimal)."""
        result = save_instance.make_table(self.LABELS, self.DATA, sig_fig=1)
        # Skip header + separator → first data line.
        first_data = result[2]
        assert "1.0e+00" in first_data, (
            f"sig_fig=1 should format 1.0 as '1.0e+00'; got line: {first_data!r}"
        )

    def test_name_decoration_adds_three_extra_lines(self, save_instance):
        """Passing a name prepends [equals, name, equals] (3 extra lines)."""
        without_name = save_instance.make_table(self.LABELS, self.DATA)
        with_name = save_instance.make_table(self.LABELS, self.DATA, name="My Run")
        assert len(with_name) == len(without_name) + 3, (
            f"name decoration should add 3 lines, got {len(with_name) - len(without_name)}"
        )
        # Layout: [equals_top, name, equals_bottom, *original_rows]
        assert "My Run" in with_name[1], (
            f"name should appear at index 1 (between equals lines); "
            f"got: {with_name[:3]!r}"
        )


class TestWriteTable:
    """``write_table`` round-trips: write strings to a file, read them back."""

    def test_round_trip_writes_each_row_on_its_own_line(self, save_instance, tmp_path):
        out = tmp_path / "table.txt"
        rows = ["header_a  header_b", "1.0e+00   2.0e+00", "3.0e+00   4.0e+00"]

        save_instance.write_table(out, rows)

        recovered = out.read_text().splitlines()
        assert recovered == rows, (
            f"round-trip mismatch:\n  wrote: {rows}\n  read:  {recovered}"
        )

    def test_line_start_skips_initial_lines(self, save_instance, tmp_path):
        out = tmp_path / "skip.txt"
        rows = ["AAA", "BBB", "CCC"]

        save_instance.write_table(out, rows, line_start=1)

        recovered = out.read_text().splitlines()
        assert recovered == ["BBB", "CCC"], (
            f"line_start=1 should skip first row; got {recovered}"
        )


class TestSimSensitivity:
    """``sim_sensitivity`` table layout, reaction columns, and file split."""

    T = np.array([0.0, 1.0, 2.0])
    SENS = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])

    @pytest.fixture
    def writer(self, save_instance, monkeypatch):
        monkeypatch.setattr(
            save_instance, "_sensitivity_array",
            lambda SIM, observable, species_idx: (self.T, self.SENS),
        )
        save_instance.save_indices = range(len(self.T))

        return save_instance

    @staticmethod
    def _path_set(tmp_path):
        return lambda name: tmp_path / "{:s}.txt".format(name)

    def test_writes_one_column_per_selected_reaction(self, writer, tmp_path):
        save_var = {
            "species": {}, "reactions": {0: "A=B", 2: "C=D"},
            "output_time_offset": 0.0,
        }
        writer.sim_sensitivity(
            None, save_var, "Temperature Sensitivity Analysis",
            self._path_set(tmp_path),
        )

        out = (tmp_path / "Temperature Sensitivity Analysis.txt").read_text()
        header = out.splitlines()[0]
        assert "R1" in header and "R3" in header, (
            f"header should carry the selected reactions R1 and R3; got {header!r}"
        )
        assert "R2" not in header, (
            f"unselected reaction R2 must not be written; got {header!r}"
        )

    def test_species_observable_writes_one_file_per_species(self, writer, tmp_path):
        save_var = {
            "species": {0: "H2", 1: "O2"}, "reactions": {0: "A=B"},
            "output_time_offset": 0.0,
        }
        writer.sim_sensitivity(
            None, save_var, "Mole Fraction Sensitivity Analysis",
            self._path_set(tmp_path),
        )

        written = sorted(p.name for p in tmp_path.glob("*.txt"))
        assert written == [
            "Mole Fraction Sensitivity Analysis - H2.txt",
            "Mole Fraction Sensitivity Analysis - O2.txt",
        ], f"expected one file per selected species; got {written}"

    def test_time_column_carries_the_output_offset(self, writer, tmp_path):
        save_var = {
            "species": {}, "reactions": {0: "A=B"},
            "output_time_offset": 5.0,
        }
        writer.sim_sensitivity(
            None, save_var, "Pressure Sensitivity Analysis",
            self._path_set(tmp_path),
        )

        out = (tmp_path / "Pressure Sensitivity Analysis.txt").read_text()
        first_data = out.splitlines()[2]
        assert "5.000e+00" in first_data, (
            f"t=0 with a 5 s offset should be written as 5.000e+00; "
            f"got line {first_data!r}"
        )
