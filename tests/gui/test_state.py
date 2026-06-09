"""Tests for the Pydantic state models in ``frhodo.gui.state``.

These are typed slices of widget state that the rest of the GUI reads
and writes. Defaults, type validation, and ``extra="forbid"`` matter
because they're the schema contract for every consumer.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from frhodo.gui.state import (
    SaveDialogState,
    ShockSelectionState,
    TimeUncertaintyState,
)


class TestShockSelectionStateDefaults:
    def test_current_defaults_to_one(self):
        assert ShockSelectionState().current == 1

    def test_previous_defaults_to_one(self):
        assert ShockSelectionState().previous == 1

    def test_step_direction_via_subtraction(self):
        """``current - previous`` is how the path-pulldown sync reads
        step direction; positive = forward, negative = backward."""
        s = ShockSelectionState(current=4, previous=2)
        assert s.current - s.previous == 2


class TestShockSelectionStateValidation:
    def test_extra_kwargs_rejected(self):
        with pytest.raises(ValidationError):
            ShockSelectionState(current=1, undocumented_field=42)

    def test_string_for_current_rejected_on_construction(self):
        with pytest.raises(ValidationError):
            ShockSelectionState(current="one")

    def test_string_for_current_rejected_on_assignment(self):
        """``validate_assignment=True`` — set-after-construct is also
        type-checked, not just __init__."""
        s = ShockSelectionState()
        with pytest.raises(ValidationError):
            s.current = "one"


class TestTimeUncertaintyStateDefaults:
    def test_value_defaults_to_zero(self):
        assert TimeUncertaintyState().value == 0.0

    def test_offset_defaults_to_zero(self):
        assert TimeUncertaintyState().offset == 0.0


class TestTimeUncertaintyStateRoundTrip:
    @pytest.mark.parametrize("value,offset", [
        (1e-6, 0.0),       # 1 μs uncertainty, no offset
        (1e-3, 1e-4),      # 1 ms uncertainty, 100 μs offset
        (0.0, 0.0),        # both zero (defaults)
        (1e-5, -1e-6),     # asymmetric: positive uncertainty, negative offset
    ])
    def test_construction_preserves_values(self, value, offset):
        t = TimeUncertaintyState(value=value, offset=offset)
        assert t.value == value
        assert t.offset == offset


class TestTimeUncertaintyStateValidation:
    def test_extra_kwargs_rejected(self):
        with pytest.raises(ValidationError):
            TimeUncertaintyState(value=1.0, surprise=True)

    def test_non_numeric_value_rejected(self):
        with pytest.raises(ValidationError):
            TimeUncertaintyState(value="lots")


class TestSaveDialogStateDefaults:
    """The Save dialog opens with sensible defaults; if any of them
    drift, the user's first save behaves unexpectedly."""

    def test_comment_default_empty_string(self):
        assert SaveDialogState().comment == ""

    def test_output_time_default_none(self):
        """``None`` means 'use solver step times'."""
        assert SaveDialogState().output_time is None

    def test_integrator_time_default_false(self):
        assert SaveDialogState().integrator_time is False

    def test_save_plot_default_true(self):
        """Saving a PNG snapshot by default matches the existing UI
        checkbox state."""
        assert SaveDialogState().save_plot is True

    def test_parameters_default_empty_list(self):
        assert SaveDialogState().parameters == []

    def test_species_default_empty_dict(self):
        assert SaveDialogState().species == {}

    def test_reactions_default_empty_dict(self):
        assert SaveDialogState().reactions == {}

    def test_output_time_offset_default_zero(self):
        assert SaveDialogState().output_time_offset == 0.0

    def test_mech_output_dir_default_empty_string(self):
        assert SaveDialogState().mech_output_dir == ""


class TestSaveDialogStateMutableDefaultsAreIndependent:
    """Pydantic's ``default_factory`` must hand each instance its own
    mutable container. Sharing would let a user's earlier save state
    leak into the next dialog opening."""

    def test_parameters_list_independent_across_instances(self):
        a = SaveDialogState()
        b = SaveDialogState()
        a.parameters.append("T_reactor")
        assert b.parameters == []

    def test_species_dict_independent_across_instances(self):
        a = SaveDialogState()
        b = SaveDialogState()
        a.species[0] = "OH"
        assert b.species == {}


class TestSaveDialogStateNdarrayField:
    """``output_time`` is typed ``Any`` to allow ``np.ndarray`` (which
    Pydantic refuses to model with ``arbitrary_types_allowed=False``).
    Round-trip must preserve dtype and values exactly."""

    def test_ndarray_round_trip_preserves_values(self):
        t = np.linspace(0.0, 1e-3, 100)
        s = SaveDialogState(output_time=t)
        np.testing.assert_array_equal(s.output_time, t)

    def test_ndarray_assignment_after_construction(self):
        s = SaveDialogState()
        new_t = np.linspace(0.0, 1.0, 10)
        s.output_time = new_t
        np.testing.assert_array_equal(s.output_time, new_t)


class TestSaveDialogStateValidation:
    def test_extra_kwargs_rejected(self):
        with pytest.raises(ValidationError):
            SaveDialogState(comment="hi", extra_field=1)
