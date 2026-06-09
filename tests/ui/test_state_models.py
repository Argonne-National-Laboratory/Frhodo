"""Tests for typed widget-state models in ``frhodo.gui.state``."""
import numpy as np
import pytest
from pydantic import ValidationError

from frhodo.gui.state import (
    SaveDialogState,
    ShockSelectionState,
    TimeUncertaintyState,
)


class TestShockSelectionState:
    def test_defaults(self):
        s = ShockSelectionState()
        assert s.current == 1
        assert s.previous == 1

    def test_step_direction(self):
        s = ShockSelectionState(current=5, previous=4)
        assert s.current - s.previous == 1

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ShockSelectionState(current=1, previous=1, foo=2)

    def test_validate_assignment_rejects_str(self):
        s = ShockSelectionState()
        with pytest.raises(ValidationError):
            s.current = "not-an-int"


class TestTimeUncertaintyState:
    def test_defaults(self):
        t = TimeUncertaintyState()
        assert t.value == 0.0
        assert t.offset == 0.0

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TimeUncertaintyState(value=0.0, offset=0.0, foo=1)


class TestSaveDialogState:
    def test_defaults(self):
        s = SaveDialogState()
        assert s.comment == ""
        assert s.output_time is None
        assert s.integrator_time is False
        assert s.parameters == []
        assert s.species == {}
        assert s.reactions == {}
        assert s.save_plot is True
        assert s.output_time_offset == 0.0

    def test_accepts_ndarray(self):
        s = SaveDialogState(output_time=np.array([0.0, 1.0]))
        assert isinstance(s.output_time, np.ndarray)
        assert s.output_time.tolist() == [0.0, 1.0]

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            SaveDialogState(comment="x", foo=1)

    def test_model_dump_carries_all_keys(self):
        s = SaveDialogState(
            output_time=np.array([0.5]),
            species={0: "OH"},
            reactions={1: "A + B <=> C"},
        )
        dumped = s.model_dump()
        for key in (
            "comment", "output_time", "integrator_time", "parameters",
            "species", "reactions", "save_plot", "output_time_offset",
        ):
            assert key in dumped
        assert dumped["species"] == {0: "OH"}
