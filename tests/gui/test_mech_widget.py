"""Tests for ``frhodo.gui.widgets.mech_widget``.

Three test categories:
  * Module-level helpers (``_clear_rate_slots``, ``_clear_coef_slots``,
    ``silentSetValue``) — pure Python.
  * ``ScientificLineEditReadOnly`` value/format contract — Qt widget,
    needs ``qtbot`` for cleanup.
  * Destroyed-signal cleanup wiring — regression test for the
    "dangling C++ object" crash that motivated the cleanup helpers.
  * ``Tree._set_mech_tree_data`` type-dispatch — regression for
    Lindemann/Tsang going down the no-coefficients path.
"""
import inspect
import sys

import cantera as ct
import pytest
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QSpinBox, QWidget
from unittest.mock import MagicMock, patch

from frhodo.gui.widgets.mech_widget import (
    COEF_OPTIMIZABLE_TYPES,
    PRESSURE_DEPENDENT_TYPES,
    RATE_OPTIMIZABLE_TYPES,
    QSortFilterProxyModel,
    RecastPressureDialog,
    ScientificLineEditReadOnly,
    Tree,
    TreeFilter,
    _TYPE_TAG_ROLE,
    _clear_coef_slots,
    _clear_rate_slots,
    _rxn_type_tags,
    silentSetValue,
)


class TestClearRateSlots:
    """Wired to the rxnRate widget's ``destroyed`` signal so any dict in
    ``Tree.rxn[rxnNum]`` that pointed into the dying widget gets emptied
    synchronously, before iteration code can read a dangling ref."""

    def test_pops_every_widget_key(self):
        rxn = {
            "item": object(), "num": 0, "rxnType": "Arrhenius Reaction",
            "dependent": False,
            "coef": [["A", "pre_exponential_factor", "ignored"]],
            "rateBox": object(),
            "formulaBox": [object()],
            "valueBox": [object()],
            "uncBox": [object(), object()],
        }
        _clear_rate_slots(rxn)
        for key in ("coef", "rateBox", "formulaBox", "valueBox", "uncBox"):
            assert key not in rxn, (
                f"{key!r} should have been popped; rxn={rxn!r}"
            )

    def test_preserves_identity_bookkeeping(self):
        """Non-widget keys (``item``, ``num``, ``rxnType``, ``dependent``)
        survive — they identify the reaction, not its render."""
        sentinel = object()
        rxn = {
            "item": sentinel, "num": 7, "rxnType": "Falloff Reaction",
            "dependent": True,
            "rateBox": object(), "valueBox": [object()],
        }
        _clear_rate_slots(rxn)
        assert rxn["item"] is sentinel
        assert rxn["num"] == 7
        assert rxn["rxnType"] == "Falloff Reaction"
        assert rxn["dependent"] is True

    def test_idempotent_when_already_clean(self):
        """A coef-widget's destroyed signal may fire after the rate
        widget's — second call must be a no-op, not a KeyError."""
        rxn = {"item": object(), "num": 0}
        _clear_rate_slots(rxn)  # no widget keys to pop
        _clear_rate_slots(rxn)  # still no-op
        assert "rateBox" not in rxn

    def test_safe_on_empty_dict(self):
        """Degenerate input — empty dict should not raise."""
        rxn = {}
        _clear_rate_slots(rxn)
        assert rxn == {}


class TestClearCoefSlots:
    """Sister cleanup for individual rateExpCoefficient widgets. Nulls
    one coefficient's slot in ``formulaBox`` / ``valueBox`` / ``uncBox``
    so the slot is dead-but-addressable while the rate-level pop is
    still pending (signal order isn't guaranteed)."""

    def test_nulls_specified_coef_in_all_three_lists(self):
        rxn = {
            "formulaBox": ["fb0", "fb1", "fb2"],
            "valueBox": ["vb0", "vb1", "vb2"],
            "uncBox": ["rate_unc", "uc0", "uc1", "uc2"],
        }
        _clear_coef_slots(rxn, coef_idx=1)
        assert rxn["formulaBox"] == ["fb0", None, "fb2"]
        assert rxn["valueBox"] == ["vb0", None, "vb2"]
        # uncBox[coef_idx + 1] for the coef slot; uncBox[0] stays
        # because that's the rate-level (rxnRate-widget) uncertainty.
        assert rxn["uncBox"] == ["rate_unc", "uc0", None, "uc2"]

    def test_preserves_rate_level_uncertainty(self):
        """``uncBox[0]`` belongs to the rxnRate widget, not any coef.
        A coef-level cleanup must never touch it."""
        sentinel = object()
        rxn = {
            "formulaBox": [None, None],
            "valueBox": [None, None],
            "uncBox": [sentinel, "coef0_unc", "coef1_unc"],
        }
        _clear_coef_slots(rxn, coef_idx=0)
        assert rxn["uncBox"][0] is sentinel

    def test_out_of_range_coef_idx_is_no_op(self):
        """Defensive: an index past the list end must not IndexError —
        protects against weird ordering during rebuilds."""
        rxn = {
            "formulaBox": ["fb0"],
            "valueBox": ["vb0"],
            "uncBox": ["rate_unc", "uc0"],
        }
        _clear_coef_slots(rxn, coef_idx=999)
        assert rxn["formulaBox"] == ["fb0"]
        assert rxn["valueBox"] == ["vb0"]
        assert rxn["uncBox"] == ["rate_unc", "uc0"]

    def test_missing_keys_no_op(self):
        """If the rate-widget destroyed signal already popped every
        list, a late coef cleanup must not KeyError."""
        rxn = {}
        _clear_coef_slots(rxn, coef_idx=0)  # no crash, no mutation
        assert rxn == {}

    def test_partial_keys_handled(self):
        """Only some lists present — clear only what's available."""
        rxn = {"valueBox": ["vb0"]}
        _clear_coef_slots(rxn, coef_idx=0)
        assert rxn == {"valueBox": [None]}


class TestSilentSetValue:
    """``silentSetValue`` is the canonical way to mutate a widget's
    value without re-firing its valueChanged signal. Used in the
    coefficient sync paths to avoid feedback loops."""

    def test_blocks_signals_during_set(self, qtbot):
        spinbox = QSpinBox()
        qtbot.addWidget(spinbox)
        changes = []
        spinbox.valueChanged.connect(changes.append)

        silentSetValue(spinbox, 5)

        assert spinbox.value() == 5, f"value not set, got {spinbox.value()}"
        assert changes == [], (
            f"valueChanged should have been suppressed, got {changes}"
        )

    def test_signals_re_enabled_after_call(self, qtbot):
        """The block is scoped to the setValue call — subsequent edits
        emit normally."""
        spinbox = QSpinBox()
        qtbot.addWidget(spinbox)
        changes = []
        spinbox.valueChanged.connect(changes.append)

        silentSetValue(spinbox, 5)  # suppressed
        spinbox.setValue(10)        # should fire

        assert changes == [10]


class TestScientificLineEditReadOnly:
    """The rate-display widget at the centre of the dangling-ref bug.
    Its setValue / setDecimals contract is what the rest of the
    mech-tree UI reads."""

    def test_setValue_stores_full_precision(self, qtbot):
        widget = ScientificLineEditReadOnly(parent=None)
        qtbot.addWidget(widget)
        widget.setValue(1.234567890123e-12)
        assert widget.value == 1.234567890123e-12

    @pytest.mark.parametrize("decimals,value,expected_text", [
        (3, 1.234567e-12, "1.23e-12"),
        (4, 1.234567e-12, "1.235e-12"),
        (1, 1.234567e-12, "1e-12"),
        (5, 0.0, "0"),
        (3, 1.0, "1"),
    ])
    def test_setValue_text_uses_g_format(self, qtbot, decimals, value, expected_text):
        """Display format is ``{:.{decimals}g}``; spot-check the
        boundaries (rounding up, zero, exact integer)."""
        widget = ScientificLineEditReadOnly(parent=None)
        qtbot.addWidget(widget)
        widget.setDecimals(decimals)
        widget.setValue(value)
        assert widget.text() == expected_text, (
            f"decimals={decimals}, value={value!r}: expected "
            f"{expected_text!r}, got {widget.text()!r}"
        )

    def test_setDecimals_takes_effect_on_next_setValue(self, qtbot):
        widget = ScientificLineEditReadOnly(parent=None)
        qtbot.addWidget(widget)
        widget.setDecimals(2)
        widget.setValue(1.234567e-12)
        first = widget.text()
        widget.setDecimals(5)
        widget.setValue(1.234567e-12)
        second = widget.text()
        assert first != second, (
            f"setDecimals had no effect; both showed {first!r}"
        )


class TestDestroyedSignalCleanup:
    """Regression for ``RuntimeError: wrapped C/C++ object of type
    ScientificLineEditReadOnly has been deleted``. The fix wires a
    widget's ``destroyed`` signal to ``_clear_rate_slots`` so the dict
    slot is emptied before any later iteration reads it."""

    def test_widget_destroyed_clears_rate_slot(self, qtbot):
        """Destroying a widget wired with the cleanup callback must
        empty the rxn dict's widget keys."""
        widget = QWidget()
        qtbot.addWidget(widget)
        rxn = {
            "item": object(), "num": 0,
            "rateBox": widget, "coef": [], "valueBox": [None],
        }
        widget.destroyed.connect(lambda _, d=rxn: _clear_rate_slots(d))

        widget.deleteLater()
        qtbot.wait(50)  # drain the event loop so destroyed fires

        assert "rateBox" not in rxn, (
            f"rateBox should have been popped, rxn={rxn!r}"
        )
        assert "coef" not in rxn
        assert "valueBox" not in rxn
        assert rxn["num"] == 0  # bookkeeping intact

    def test_widget_destroyed_clears_coef_slot(self, qtbot):
        """Same wiring shape for individual coefficient widgets."""
        widget = QWidget()
        qtbot.addWidget(widget)
        rxn = {
            "formulaBox": [None, widget, None],
            "valueBox": [None, "still_live", None],
            "uncBox": [None, None, widget, None],
        }
        widget.destroyed.connect(
            lambda _, d=rxn: _clear_coef_slots(d, coef_idx=1),
        )

        widget.deleteLater()
        qtbot.wait(50)

        assert rxn["formulaBox"][1] is None
        # uncBox[coef_idx + 1] = uncBox[2] is the coef slot
        assert rxn["uncBox"][2] is None
        # valueBox[1] held a non-widget marker — gets nulled because
        # the slot is identified by coef_idx, not by what's currently
        # in it. That's the right behavior: post-destroy, the slot is
        # invalid regardless of intermediate state.
        assert rxn["valueBox"][1] is None


class TestSetMechTreeDataTypeDispatch:
    """``_set_mech_tree_data`` must classify Lindemann and Tsang rate
    types alongside the other Falloff variants. Before the fix, those
    fell through to the no-coefficient else branch and then
    ``_set_mech_widgets`` raised ``KeyError: 'coeffs_order'`` when the
    user clicked the row."""

    def test_lindemann_gets_coeffs_order(self, loaded_all_rate_types):
        """Bug 1 regression: rxn 1 of the all-rate-types fixture is
        LindemannRate; its entry must carry ``coeffs_order``."""
        mock_tree = MagicMock()
        data = Tree._set_mech_tree_data(mock_tree, "Chemkin", loaded_all_rate_types)
        rxn_1 = data[1]
        assert type(loaded_all_rate_types.gas.reaction(1).rate).__name__ == "LindemannRate"
        assert "coeffs_order" in rxn_1, (
            f"Lindemann should match the Falloff branch; got keys="
            f"{sorted(rxn_1.keys())}"
        )

    def test_lindemann_coeffs_order_matches_falloff_shape(self, loaded_all_rate_types):
        """Order ``[1, 2, 0, 4, 5, 3]`` = A_high, n_high, Ea_high,
        A_low, n_low, Ea_low — same as Plog/Falloff/Troe/Sri."""
        mock_tree = MagicMock()
        data = Tree._set_mech_tree_data(mock_tree, "Chemkin", loaded_all_rate_types)
        assert data[1]["coeffs_order"] == [1, 2, 0, 4, 5, 3]

    def test_all_falloff_variants_have_six_coeff_slots(self, loaded_all_rate_types):
        """LindemannRate, TroeRate, SriRate all share the same
        6-Arrhenius-limb layout the tree displays."""
        mock_tree = MagicMock()
        data = Tree._set_mech_tree_data(mock_tree, "Chemkin", loaded_all_rate_types)
        for rxn_idx, expected_type in [(1, "LindemannRate"), (2, "TroeRate"), (5, "SriRate")]:
            actual_type = type(loaded_all_rate_types.gas.reaction(rxn_idx).rate).__name__
            assert actual_type == expected_type, (
                f"fixture drift: rxn {rxn_idx} expected {expected_type}, "
                f"got {actual_type}"
            )
            assert len(data[rxn_idx]["coeffs"]) == 6, (
                f"{expected_type} rxn {rxn_idx}: expected 6 coefs, got "
                f"{len(data[rxn_idx]['coeffs'])}"
            )

    def test_falloff_family_covers_tsang_and_lindemann(self):
        """The dispatch in ``_set_mech_tree_data`` uses ``_FALLOFF_FAMILY``;
        verify every variant the bug originally covered is still in it."""
        import cantera as ct

        from frhodo.simulation.mechanism.mech_fcns import _FALLOFF_FAMILY

        for cls in (ct.LindemannRate, ct.TsangRate, ct.TroeRate, ct.SriRate):
            assert cls in _FALLOFF_FAMILY, (
                f"{cls.__name__} dropped from _FALLOFF_FAMILY"
            )


class TestRateOptimizableTypes:
    """Type-allowlist constants drive every GUI gate that decides if a
    reaction can be optimized. ``RATE_OPTIMIZABLE_TYPES`` is the broad
    set (rate-level F-factor uncertainty); ``COEF_OPTIMIZABLE_TYPES``
    is the narrower set that additionally supports per-coefficient
    widgets."""

    @pytest.mark.parametrize("rxn_type", [
        "Arrhenius Reaction",
        "Three Body Reaction",
        "Plog Reaction",
        "Falloff Reaction",
        "Chebyshev Reaction",
    ])
    def test_rate_optimizable_includes(self, rxn_type):
        assert rxn_type in RATE_OPTIMIZABLE_TYPES, (
            f"{rxn_type} should accept a rate-level uncertainty box"
        )

    @pytest.mark.parametrize("rxn_type", [
        "Arrhenius Reaction",
        "Three Body Reaction",
        "Plog Reaction",
        "Falloff Reaction",
    ])
    def test_coef_optimizable_includes(self, rxn_type):
        assert rxn_type in COEF_OPTIMIZABLE_TYPES

    def test_chebyshev_not_in_coef_optimizable(self):
        """Chebyshev's 2D coefficient matrix has no per-coef widget
        layout — only the rate-level F factor is editable."""
        assert "Chebyshev Reaction" not in COEF_OPTIMIZABLE_TYPES


class TestRxnTypeTags:
    """``_rxn_type_tags`` maps a (category, rate_class) pair to the set
    of lowercase tokens the filter recognizes. Each rxn must match its
    direct sub-variant AND the catch-all aliases the user might use."""

    @pytest.mark.parametrize("rxn_type,rate_class,expected", [
        ("Arrhenius Reaction", "ArrheniusRate", {"arrhenius"}),
        ("Three Body Reaction", "ArrheniusRate",
         {"arrhenius", "three-body", "threebody"}),
        ("Plog Reaction", "PlogRate", {"plog", "pressure-sensitive"}),
        ("Falloff Reaction", "TroeRate", {"falloff", "troe", "pressure-sensitive"}),
        ("Falloff Reaction", "LindemannRate",
         {"falloff", "lindemann", "pressure-sensitive"}),
        ("Falloff Reaction", "SriRate", {"falloff", "sri", "pressure-sensitive"}),
        ("Falloff Reaction", "TsangRate",
         {"falloff", "tsang", "pressure-sensitive"}),
        ("Falloff Reaction", "FalloffRate", {"falloff", "pressure-sensitive"}),
        ("Chebyshev Reaction", "ChebyshevRate",
         {"chebyshev", "pressure-sensitive"}),
    ])
    def test_tags_match_expected(self, rxn_type, rate_class, expected):
        assert _rxn_type_tags(rxn_type, rate_class) == expected, (
            f"({rxn_type}, {rate_class}) produced wrong tag set"
        )

    def test_unknown_rxn_type_returns_empty(self):
        """A rxn type we don't recognize gets an empty tag set — it
        appears in the unfiltered view but no ``type:X`` filter matches."""
        assert _rxn_type_tags("Surface Reaction", "InterfaceArrheniusRate") == frozenset()

    def test_pressure_sensitive_covers_all_p_dependent_types(self):
        """``type:pressure-sensitive`` must match Plog + Chebyshev +
        every Falloff variant — the union of recast-to-Troe sources."""
        for rxn_type, rate_class in [
            ("Plog Reaction", "PlogRate"),
            ("Chebyshev Reaction", "ChebyshevRate"),
            ("Falloff Reaction", "FalloffRate"),
            ("Falloff Reaction", "TroeRate"),
            ("Falloff Reaction", "LindemannRate"),
            ("Falloff Reaction", "SriRate"),
            ("Falloff Reaction", "TsangRate"),
        ]:
            assert "pressure-sensitive" in _rxn_type_tags(rxn_type, rate_class), (
                f"({rxn_type}, {rate_class}) missing pressure-sensitive tag"
            )

    def test_arrhenius_not_pressure_sensitive(self):
        """Pure Arrhenius (no [M], no falloff) must not match
        ``type:pressure-sensitive``."""
        assert "pressure-sensitive" not in _rxn_type_tags(
            "Arrhenius Reaction", "ArrheniusRate",
        )


class TestPressureDependentTypes:
    def test_covers_plog_falloff_chebyshev(self):
        assert set(PRESSURE_DEPENDENT_TYPES) == {
            "Plog Reaction", "Falloff Reaction", "Chebyshev Reaction",
        }

    def test_excludes_arrhenius(self):
        assert "Arrhenius Reaction" not in PRESSURE_DEPENDENT_TYPES
        assert "Three Body Reaction" not in PRESSURE_DEPENDENT_TYPES


class TestRecastPressureDialog:
    """``RecastPressureDialog`` collects a pressure + unit and reports it
    in pascals — the input side of the per-reaction recast."""

    def test_defaults_reflect_zone_value_and_unit(self, qtbot):
        dialog = RecastPressureDialog(None, default_value=760.0, default_unit="torr")
        qtbot.addWidget(dialog)
        assert dialog.unit_box.currentText() == "torr"
        assert dialog.value_box.value() == pytest.approx(760.0, rel=1e-9)
        assert dialog.pressure_pa() == pytest.approx(101325.0, rel=1e-9)

    @pytest.mark.parametrize(
        "unit, value, expected_pa",
        [
            ("bar", 3.0, 3.0e5),
            ("torr", 760.0, 101325.0),
            ("Pa", 5000.0, 5000.0),
            ("kPa", 50.0, 5.0e4),
        ],
    )
    def test_pressure_pa_converts_units(self, qtbot, unit, value, expected_pa):
        dialog = RecastPressureDialog(None)
        qtbot.addWidget(dialog)
        dialog.unit_box.setCurrentText(unit)
        dialog.value_box.setValue(value)
        assert dialog.pressure_pa() == pytest.approx(expected_pa, rel=1e-9)

    def test_unknown_unit_falls_back_to_atm(self, qtbot):
        dialog = RecastPressureDialog(None, default_value=2.0, default_unit="mmHg")
        qtbot.addWidget(dialog)
        assert dialog.unit_box.currentText() == "atm"

    def test_nonpositive_value_falls_back_to_one(self, qtbot):
        dialog = RecastPressureDialog(None, default_value=0.0, default_unit="bar")
        qtbot.addWidget(dialog)
        assert dialog.value_box.value() == pytest.approx(1.0, rel=1e-9)

    def test_nonfinite_value_falls_back_to_one(self, qtbot):
        dialog = RecastPressureDialog(None, default_value=float("nan"))
        qtbot.addWidget(dialog)
        assert dialog.value_box.value() == pytest.approx(1.0, rel=1e-9)


class TestTreeFilterTokenSplit:
    """``TreeFilter._split_filter_tokens`` pulls ``type:X`` tokens out of
    the filter input so the proxy model can use them separately from
    text matching."""

    def test_plain_text_passes_through(self):
        types, text = TreeFilter._split_filter_tokens("H2 + OH")
        assert types == []
        assert text == ["H2", "+", "OH"]

    def test_single_type_token_extracted(self):
        types, text = TreeFilter._split_filter_tokens("type:plog")
        assert types == ["plog"]
        assert text == []

    def test_type_token_case_insensitive(self):
        types, _ = TreeFilter._split_filter_tokens("Type:Plog")
        assert types == ["plog"]

    def test_type_tokens_lowercased(self):
        types, _ = TreeFilter._split_filter_tokens("type:CHEBYSHEV")
        assert types == ["chebyshev"]

    def test_mixed_input_splits_correctly(self):
        types, text = TreeFilter._split_filter_tokens("OH type:falloff H2")
        assert types == ["falloff"]
        assert text == ["OH", "H2"]

    def test_multiple_type_tokens_accumulate(self):
        types, text = TreeFilter._split_filter_tokens("type:plog type:chebyshev")
        assert types == ["plog", "chebyshev"]
        assert text == []

    def test_empty_input_gives_empty_lists(self):
        types, text = TreeFilter._split_filter_tokens("")
        assert types == []
        # An empty input still splits into [""] via str.split(" "); the
        # text-token branch keeps that, but the regex builder treats
        # length-zero strings as no-ops.
        assert text == [""]


class TestProxyModelTypeFilter:
    """``QSortFilterProxyModel.setTypeFilter`` restricts visible rows to
    those whose UserRole tag set intersects the filter set. Verifies
    the full chain: setData on items → setTypeFilter → row visibility."""

    def _build_proxy_with_rows(self, rows):
        """``rows`` is a list of ``(label, tags)`` pairs."""
        model = QtGui.QStandardItemModel()
        for label, tags in rows:
            item = QtGui.QStandardItem(label)
            item.setData(frozenset(tags), _TYPE_TAG_ROLE)
            model.appendRow(item)
        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        return proxy

    def test_empty_filter_admits_all_rows(self, qtbot):
        proxy = self._build_proxy_with_rows([
            (" R1: A", {"arrhenius"}),
            (" R2: B", {"plog", "pressure-sensitive"}),
        ])
        proxy.setTypeFilter(frozenset())
        assert proxy.rowCount() == 2

    def test_single_type_match(self, qtbot):
        proxy = self._build_proxy_with_rows([
            (" R1: A", {"arrhenius"}),
            (" R2: B", {"plog", "pressure-sensitive"}),
            (" R3: C", {"falloff", "troe", "pressure-sensitive"}),
        ])
        proxy.setTypeFilter(frozenset({"plog"}))
        assert proxy.rowCount() == 1
        assert "R2" in proxy.data(proxy.index(0, 0))

    def test_pressure_sensitive_matches_plog_chebyshev_falloff(self, qtbot):
        """The ``pressure-sensitive`` umbrella is the union of all P-
        dependent rxn types."""
        proxy = self._build_proxy_with_rows([
            (" R1: arrh", {"arrhenius"}),
            (" R2: plog", {"plog", "pressure-sensitive"}),
            (" R3: cheb", {"chebyshev", "pressure-sensitive"}),
            (" R4: troe", {"falloff", "troe", "pressure-sensitive"}),
            (" R5: 3body", {"arrhenius", "three-body", "threebody"}),
        ])
        proxy.setTypeFilter(frozenset({"pressure-sensitive"}))
        assert proxy.rowCount() == 3

    def test_multi_type_filter_is_union(self, qtbot):
        """``type:plog type:chebyshev`` matches rows tagged with EITHER."""
        proxy = self._build_proxy_with_rows([
            (" R1: arrh", {"arrhenius"}),
            (" R2: plog", {"plog", "pressure-sensitive"}),
            (" R3: cheb", {"chebyshev", "pressure-sensitive"}),
            (" R4: troe", {"falloff", "troe", "pressure-sensitive"}),
        ])
        proxy.setTypeFilter(frozenset({"plog", "chebyshev"}))
        assert proxy.rowCount() == 2

    def test_unknown_type_filter_yields_empty(self, qtbot):
        proxy = self._build_proxy_with_rows([
            (" R1: arrh", {"arrhenius"}),
            (" R2: plog", {"plog"}),
        ])
        proxy.setTypeFilter(frozenset({"surface"}))
        assert proxy.rowCount() == 0

    def test_filter_combined_with_text_regex(self, qtbot):
        """Type filter is AND-ed with the text regex; a row must satisfy
        both to be visible."""
        proxy = self._build_proxy_with_rows([
            (" R1: H + O -> OH", {"arrhenius"}),
            (" R2: H + OH -> H2O", {"plog", "pressure-sensitive"}),
            (" R3: H2O + M -> H + OH + M", {"plog", "pressure-sensitive"}),
        ])
        proxy.setFilterRegularExpression("OH")
        proxy.setTypeFilter(frozenset({"plog"}))
        # R1: text matches but not plog → excluded
        # R2: text matches AND plog → included
        # R3: text matches AND plog → included
        assert proxy.rowCount() == 2


class TestSetMechTreeDataRateClass:
    """Each rxn entry must carry its Cantera rate class name so the
    tag-builder can disambiguate Falloff sub-variants (Troe vs
    Lindemann vs Sri vs Tsang)."""

    def test_every_rxn_carries_rate_class(self, loaded_all_rate_types):
        mock_tree = MagicMock()
        data = Tree._set_mech_tree_data(
            mock_tree, "Chemkin", loaded_all_rate_types,
        )
        expected = [
            "ArrheniusRate", "LindemannRate", "TroeRate",
            "PlogRate", "ChebyshevRate", "SriRate",
        ]
        for rxn_idx, expected_class in enumerate(expected):
            assert data[rxn_idx]["rateClass"] == expected_class, (
                f"rxn {rxn_idx}: expected rateClass={expected_class!r}, "
                f"got {data[rxn_idx].get('rateClass')!r}"
            )


class TestUncBoxBootstrap:
    """Bug 3 regression: the initial ``uncBox`` list was built as
    ``[widget.uncValBox] * (len_coef + 1)`` — same ref repeated,
    overwritten in a later loop. Replaced with the explicit
    ``[widget.uncValBox] + [None] * len_coef`` so the structure reads
    correctly without depending on the loop."""

    def test_uncbox_init_is_rate_plus_none_placeholders(self):
        """Source-level check on the coef-row builder; the alternative
        (replacing the widget scaffolding with a mock) costs far more
        than the bug warrants — this is cosmetic clarity."""
        src = inspect.getsource(Tree._build_coef_rxn_row)
        assert "[widget.uncValBox] + [None] * len_coef" in src, (
            "uncBox initialization expected the explicit "
            "[rate-widget] + [None] * len_coef shape"
        )
        assert "[widget.uncValBox] * (len_coef + 1)" not in src, (
            "the aliased-list bootstrap should have been removed"
        )


class TestSnapshotForReloadIncludesExpanded:
    """``snapshot_for_reload`` captures both ``opened`` (rows whose
    widgets were ever built) and ``expanded`` (rows currently visually
    expanded). The latter is used to restore the per-rxn collapsed/
    expanded state across the reload."""

    def test_returns_three_keyed_dict(self):
        tree = MagicMock()
        tree.snapshot_for_reload = Tree.snapshot_for_reload.__get__(tree)
        parent = MagicMock()
        parent.load_state.mech_loaded = True
        parent.mech.gas.n_reactions = 0
        tree.parent.return_value = parent
        tree.model.rowCount.return_value = 0

        with patch(
            "frhodo.gui.widgets.mech_widget.signatures_for_gas",
            return_value=[],
        ), patch(
            "frhodo.gui.widgets.mech_widget.capture_state",
            return_value={},
        ):
            snap = tree.snapshot_for_reload()

        assert snap is not None
        assert set(snap) == {"mech", "opened", "expanded"}


class TestHandleReload:
    """``Tree.handle_reload`` is the orchestration entry point for the
    mech-reload preservation feature: it runs ``optimizables.reset()``,
    delegates engine-side restoration, and pre-builds widgets for
    previously-opened rxns whose signature matched."""

    @pytest.fixture
    def tree(self):
        tree = MagicMock(spec=Tree)
        tree.handle_reload = Tree.handle_reload.__get__(tree)
        tree.snapshot_for_reload = Tree.snapshot_for_reload.__get__(tree)
        return tree

    def test_first_load_just_builds_tree(self, tree):
        parent = MagicMock()
        tree.parent.return_value = parent

        tree.handle_reload(None)

        tree.set_trees.assert_called_once_with(parent.mech)
        parent.optimizables.reset.assert_not_called()

    def test_reload_with_snapshot_resets_optimizables(self, tree):
        parent = MagicMock()
        tree.parent.return_value = parent
        snapshot = {"mech": {}, "opened": set()}

        with patch(
            "frhodo.gui.widgets.mech_widget.restore_state",
            return_value=(set(), set()),
        ):
            tree.handle_reload(snapshot)

        parent.optimizables.reset.assert_called_once_with()

    def test_reload_passes_partial_idxs_to_set_trees(self, tree):
        parent = MagicMock()
        tree.parent.return_value = parent
        snapshot = {"mech": {"sig": "x"}, "opened": set()}

        with patch(
            "frhodo.gui.widgets.mech_widget.restore_state",
            return_value=({1, 3}, {3}),
        ):
            tree.handle_reload(snapshot)

        tree.set_trees.assert_called_once_with(
            parent.mech, partial_match_idxs={3},
        )

    def test_snapshot_returns_none_when_not_loaded(self, tree):
        parent = MagicMock()
        parent.load_state.mech_loaded = False
        tree.parent.return_value = parent

        assert tree.snapshot_for_reload() is None


class TestPartialClearedOnUserInteraction:
    """A user clicking any uncBox on a partial-match rxn should
    remove the partial flag — the dark-red was a "review this" cue,
    and the user just did."""

    @pytest.fixture
    def tree(self):
        tree = MagicMock()
        tree._clear_partial_on_user_interaction = (
            Tree._clear_partial_on_user_interaction.__get__(tree)
        )
        tree._partial_match_idxs = {3, 7}
        tree._building_widgets = False
        return tree

    def test_clears_partial_idx(self, tree):
        tree._clear_partial_on_user_interaction(3)

        assert 3 not in tree._partial_match_idxs
        assert 7 in tree._partial_match_idxs, "other partial flags untouched"

    def test_repaints_after_clear(self, tree):
        tree._clear_partial_on_user_interaction(3)

        tree._paint_rxn_foreground.assert_called_once_with(3)

    def test_noop_when_not_partial(self, tree):
        tree._clear_partial_on_user_interaction(99)

        tree._paint_rxn_foreground.assert_not_called()
        assert tree._partial_match_idxs == {3, 7}

    def test_skipped_during_widget_construction(self, tree):
        tree._building_widgets = True

        tree._clear_partial_on_user_interaction(3)

        assert 3 in tree._partial_match_idxs, (
            "programmatic widget construction must not clear partial flag"
        )
        tree._paint_rxn_foreground.assert_not_called()


class TestApplyInitialForeground:
    """``_apply_initial_foreground`` paints text color at tree-build time
    using a three-way priority: partial-match > optimizable > default."""

    @pytest.fixture
    def tree_with_three_rows(self):
        tree = MagicMock()
        tree._apply_initial_foreground = (
            Tree._apply_initial_foreground.__get__(tree)
        )
        tree._paint_rxn_foreground = (
            Tree._paint_rxn_foreground.__get__(tree)
        )
        tree.color = {
            "variable_rxn": "PURPLE",
            "fixed_rxn": "BLACK",
            "partial_match_rxn": "DARK_RED",
        }
        tree._partial_match_idxs = set()

        items = []
        for n in range(3):
            item = MagicMock()
            item.info = {"rxnNum": n}
            items.append(item)
        tree.model.rowCount.return_value = 3
        tree.model.item.side_effect = lambda r, c: items[r]
        return tree, items

    def test_partial_wins_over_optimizable(self, tree_with_three_rows):
        tree, items = tree_with_three_rows
        tree._partial_match_idxs = {1}
        parent = MagicMock()
        parent.optimizables.is_reaction_optimizable.return_value = True
        tree.parent.return_value = parent

        tree._apply_initial_foreground()

        items[1].setForeground.assert_called_with("DARK_RED")
        items[0].setForeground.assert_called_with("PURPLE")

    def test_non_optimizable_gets_black(self, tree_with_three_rows):
        tree, items = tree_with_three_rows
        parent = MagicMock()
        parent.optimizables.is_reaction_optimizable.return_value = False
        tree.parent.return_value = parent

        tree._apply_initial_foreground()

        items[0].setForeground.assert_called_with("BLACK")


class TestHandleReloadRestoresExpandedState:
    def test_expanded_rxn_gets_re_expanded(self):
        """After reload, a rxn that was previously visually expanded
        and whose signature still matches should be re-expanded (its
        widgets are also pre-built so the row renders the restored
        boxes)."""
        tree = MagicMock()
        tree.handle_reload = Tree.handle_reload.__get__(tree)
        parent = MagicMock()
        parent.mech.gas.n_reactions = 1
        tree.parent.return_value = parent

        item = MagicMock()
        item.info = {"hasExpanded": False}
        tree.model.item.return_value = item
        proxy_idx = object()
        tree.proxy_model.mapFromSource.return_value = proxy_idx

        snapshot = {
            "mech": {"sig_x": "state"},
            "opened": {"sig_x"},
            "expanded": {"sig_x"},
        }

        with patch(
            "frhodo.gui.widgets.mech_widget.restore_state",
            return_value=({0}, set()),
        ), patch(
            "frhodo.gui.widgets.mech_widget.signatures_for_gas",
            return_value=["sig_x"],
        ):
            tree.handle_reload(snapshot)

        parent.mech_tree.expand.assert_called_once_with(proxy_idx)
        assert item.info["hasExpanded"] is True
        assert item.info["isExpanded"] is True

    def test_opened_but_not_expanded_stays_collapsed(self):
        tree = MagicMock()
        tree.handle_reload = Tree.handle_reload.__get__(tree)
        parent = MagicMock()
        parent.mech.gas.n_reactions = 1
        tree.parent.return_value = parent

        item = MagicMock()
        item.info = {"hasExpanded": False}
        tree.model.item.return_value = item

        snapshot = {
            "mech": {"sig_x": "state"},
            "opened": {"sig_x"},
            "expanded": set(),
        }

        with patch(
            "frhodo.gui.widgets.mech_widget.restore_state",
            return_value=({0}, set()),
        ), patch(
            "frhodo.gui.widgets.mech_widget.signatures_for_gas",
            return_value=["sig_x"],
        ):
            tree.handle_reload(snapshot)

        parent.mech_tree.expand.assert_not_called()
        assert item.info["hasExpanded"] is True


class TestUpdateUncertaintiesCoefBranchGuards:
    """Coef branch of ``update_uncertainties`` must not raise KeyError or
    IndexError when the rxn dict has been emptied by widget teardown.

    Scenario: the background recast queue rebuilds the Cantera Solution
    via ``set_mechanism`` while the user is still poking at a stale
    coef widget. The widget's ``destroyed`` signal cleared the rxn
    dict's slot keys; the next signal from that widget then tries to
    read them. The guards here are the boundary that keeps Qt from
    surfacing a crash dialog.
    """

    @pytest.fixture
    def tree(self):
        return MagicMock(spec=Tree)

    @pytest.fixture
    def sender(self):
        sender = MagicMock()
        sender.info = {
            "rxnNum": 0,
            "coefNum": 0,
            "coefName": "activation_energy",
            "coefAbbr": "Ea",
        }
        return sender

    def _setup_parent(self, rxn_dict_state):
        parent = MagicMock()
        parent.mech_tree.rxn = {0: rxn_dict_state}
        parent.mech.coeffs_bnds = {
            0: {"low_rate": {"activation_energy": {
                "value": 1.0, "type": "F",
                "limits": lambda: (0.0, 10.0),
            }}}
        }
        parent.mech.coeffs = {0: {"low_rate": {"activation_energy": 5.0}}}
        return parent

    def test_missing_uncbox_key_returns_quietly(self, tree, sender):
        parent = self._setup_parent({})  # rxn dict has no "uncBox"
        tree.parent.return_value = parent

        with patch(
            "frhodo.gui.widgets.mech_widget.keysFromBox",
            return_value=("low_rate", "low_rate"),
        ):
            Tree.update_uncertainties(tree, event=1.0, sender=sender)

        parent.optimizables.set_coefficient_optimizable.assert_not_called()

    def test_short_uncbox_list_returns_quietly(self, tree, sender):
        parent = self._setup_parent({"uncBox": [object()]})
        tree.parent.return_value = parent

        with patch(
            "frhodo.gui.widgets.mech_widget.keysFromBox",
            return_value=("low_rate", "low_rate"),
        ):
            Tree.update_uncertainties(tree, event=1.0, sender=sender)

        parent.optimizables.set_coefficient_optimizable.assert_not_called()

    def test_none_uncbox_slot_returns_quietly(self, tree, sender):
        parent = self._setup_parent({"uncBox": [object(), None]})
        tree.parent.return_value = parent

        with patch(
            "frhodo.gui.widgets.mech_widget.keysFromBox",
            return_value=("low_rate", "low_rate"),
        ):
            Tree.update_uncertainties(tree, event=1.0, sender=sender)

        parent.optimizables.set_coefficient_optimizable.assert_not_called()
