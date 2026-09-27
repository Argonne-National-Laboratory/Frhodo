"""``settings.Path``: the mechanism listings, and the directory files that set the folders."""
import pytest



pytestmark = pytest.mark.gui


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
