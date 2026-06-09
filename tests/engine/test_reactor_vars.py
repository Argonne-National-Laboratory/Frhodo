"""``frhodo.simulation.shock.reactor_output`` registry — sanity checks on the canonical
ReactorVar set. ``ReactorOutput`` consumers (``SimProperty``,
``set_observable``) drive off this registry, so the entries' shape
matters for every downstream simulation read.
"""
import pytest

from frhodo.simulation.shock.reactor_output import (
    BY_DISPLAY_OBSERVABLE,
    BY_SIM_NAME,
    REACTOR_VARS,
    VARIANTS_BY_DISPLAY,
    base_sim_name_for_display,
    sub_types_for_display,
)


class TestRegistryStructure:
    def test_sim_names_are_unique(self):
        names = [v.sim_name for v in REACTOR_VARS]
        assert len(names) == len(set(names)), (
            f"duplicate sim_name: {[n for n in names if names.count(n) > 1]}"
        )

    def test_every_entry_has_compute_xor_cantera_attr(self):
        """Each entry either reads a Cantera attribute or computes its
        value. Both ``None`` is invalid; both set is also invalid."""
        for v in REACTOR_VARS:
            has_compute = v.compute is not None
            has_attr = v.cantera_attr is not None
            assert has_compute != has_attr, (
                f"{v.sim_name}: compute={has_compute}, "
                f"cantera_attr={has_attr}; exactly one must be set"
            )

    def test_displays_are_consistent(self):
        """If multiple variants share a display, they share it for a
        reason — they are sub_types of the same conceptual quantity."""
        for display, variants in VARIANTS_BY_DISPLAY.items():
            if len(variants) == 1:
                continue
            sub_types = {v.sub_type for v in variants}
            assert None not in sub_types, (
                f"display {display!r} has multiple variants "
                f"but one has sub_type=None: {variants}"
            )


class TestObservableLookup:
    def test_known_observables_present(self):
        for display in (
            "Temperature", "Pressure", "Density Gradient",
            "Heat Release Rate", "Mole Fraction", "Mass Fraction",
            "Concentration",
        ):
            assert display in BY_DISPLAY_OBSERVABLE, (
                f"{display!r} must be in BY_DISPLAY_OBSERVABLE"
            )

    def test_observable_default_resolves_to_total_variant(self):
        """``Density Gradient`` → ``drhodz_tot`` (the total, not the per-rxn)."""
        assert BY_DISPLAY_OBSERVABLE["Density Gradient"] == "drhodz_tot"
        assert BY_DISPLAY_OBSERVABLE["Heat Release Rate"] == "HRR_tot"


class TestLegacyHelpers:
    @pytest.mark.parametrize("display,expected_sub_types", [
        ("Temperature", None),
        ("Density Gradient", ["total", "rxn"]),
        ("Mole Fraction", ["species"]),
        ("Heat Release Rate", ["total", "rxn"]),
    ])
    def test_sub_types_for_display(self, display, expected_sub_types):
        assert sub_types_for_display(display) == expected_sub_types

    @pytest.mark.parametrize("display,expected_base", [
        ("Temperature", "T"),
        ("Density Gradient", "drhodz"),  # rxn variant, not _tot
        ("Mole Fraction", "X"),
        ("Heat Release Rate", "HRR"),    # rxn variant, not _tot
        ("Enthalpy", "h"),               # species variant, not _tot
    ])
    def test_base_sim_name_for_display(self, display, expected_base):
        assert base_sim_name_for_display(display) == expected_base


class TestRegistryIndices:
    def test_by_sim_name_covers_registry(self):
        assert set(BY_SIM_NAME) == {v.sim_name for v in REACTOR_VARS}

    def test_variants_by_display_covers_registry(self):
        all_sim_names = {v.sim_name for v in REACTOR_VARS}
        assembled = {
            v.sim_name
            for variants in VARIANTS_BY_DISPLAY.values()
            for v in variants
        }
        assert assembled == all_sim_names
