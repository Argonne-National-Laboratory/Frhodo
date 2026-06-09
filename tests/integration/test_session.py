"""Whole-session save/load serialization and reaction-state round-trips."""
import types

import numpy as np

from frhodo.experiment.data import ExperimentalShock
from frhodo.optimize.parameters import OptimizableSetBuilder
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader
from frhodo.simulation.mechanism.mech_snapshot import restore_state
from frhodo.gui.session import (
    CoefState,
    ReactionState,
    SeriesOverride,
    SessionState,
    ShockOverride,
    apply_shock_overrides,
    reactions_from_mech,
    series_overrides_from_series,
)



def test_session_yaml_round_trip_preserves_nan_and_flags():
    rxn = ReactionState(
        reactants=[("H2", 1.0), ("O", 1.0)],
        products=[("H", 1.0), ("OH", 1.0)],
        reversible=True,
        rate_unc_value=2.5,
        rate_unc_type="F",
        rate_optimizable=True,
        coefs=[
            CoefState(bnds_key="rate", coef_name="A", value=3.0, reset_val=1.2e13,
                      optimizable=True),
            CoefState(bnds_key="rate", coef_name="Ea", value=float("nan"),
                      reset_val=5000.0, optimizable=False),
        ],
    )
    state = SessionState(comment="run", reactions=[rxn])

    back = SessionState.from_yaml_text(state.to_yaml_text())

    assert back.comment == "run"
    assert back.reactions[0].rate_optimizable is True
    assert back.reactions[0].coefs[0].value == 3.0
    assert np.isnan(back.reactions[0].coefs[1].value)
    assert back.config.session.autosnapshot_enabled is True


def test_reaction_signature_survives_yaml_round_trip():
    rxn = ReactionState(
        reactants=[("O", 1.0), ("H2", 1.0)],
        products=[("OH", 1.0), ("H", 1.0)],
        reversible=False,
    )
    state = SessionState(reactions=[rxn])

    back = SessionState.from_yaml_text(state.to_yaml_text())

    assert back.reactions[0].signature() == rxn.signature()


def test_restore_snapshot_shape_matches_capture_contract():
    rxn = ReactionState(
        reactants=[("H2", 1.0)], products=[("H", 2.0)], reversible=True,
        rate_unc_value=4.0, rate_unc_type="%", rate_optimizable=True,
        coefs=[CoefState(bnds_key="rate", coef_name="A", value=2.0,
                         reset_val=1e10, optimizable=True)],
    )
    snapshot = SessionState(reactions=[rxn]).to_restore_snapshot()

    sig = rxn.signature()
    assert sig in snapshot
    entry = snapshot[sig][0]
    assert entry["rate_optimizable"] is True
    assert entry["rate_unc"] == {"value": 4.0, "type": "%"}
    coef = entry["coef_state"][("rate", "A")]
    assert coef["value"] == 2.0
    assert coef["resetVal"] == 1e10
    assert coef["optimizable"] is True


def test_apply_shock_overrides_skips_all_nan_weights():
    shock = ExperimentalShock(num=2)
    shock.weight_max = [99.0]  # seeded finite value the override must not clobber

    nan_override = ShockOverride(
        num=2, weight_max=[float("nan")], include=False,
    )
    apply_shock_overrides([shock], [nan_override])

    assert shock.weight_max == [99.0]
    assert shock.include is False


def test_apply_shock_overrides_restores_finite_weights_and_include():
    shock = ExperimentalShock(num=3)

    override = ShockOverride(
        num=3, include=True, weight_max=[80.0], weight_min=[0.0, 0.0],
        weight_shift=[4.5, 35.0], weight_k=[0.0, 0.7],
        observable_main="Density Gradient", observable_sub=0,
    )
    apply_shock_overrides([shock], [override])

    assert shock.weight_max == [80.0]
    assert shock.weight_shift == [4.5, 35.0]
    assert shock.observable["main"] == "Density Gradient"
    assert shock.include is True


def test_series_overrides_skip_placeholder_and_capture_include():
    sh1 = ExperimentalShock(num=1)
    sh1.include = True
    sh2 = ExperimentalShock(num=2)
    sh2.include = False
    series = types.SimpleNamespace(
        idx=1,
        path=[[], "/data/expA"],  # index 0 is the GUI's empty placeholder series
        name=["", "Set A"],
        in_table=[False, True],
        shock=[[ExperimentalShock(num=1)], [sh1, sh2]],
    )

    overrides = series_overrides_from_series(series)

    assert len(overrides) == 1  # placeholder series skipped
    assert overrides[0].exp_dir == "/data/expA"
    assert overrides[0].in_table is True
    assert [s.include for s in overrides[0].shocks] == [True, False]


def test_selection_round_trip_preserves_time_offset_and_config_uncertainty():
    state = SessionState()
    state.selection.time_offset = 4.2
    state.config.optimization.time_uncertainty = 1.5
    state.config.optimization.random_t_uncertainty = False

    back = SessionState.from_yaml_text(state.to_yaml_text())

    assert back.selection.time_offset == 4.2
    assert back.config.optimization.time_uncertainty == 1.5
    assert back.config.optimization.random_t_uncertainty is False


def test_series_override_round_trip_preserves_include_and_in_table():
    series_ov = SeriesOverride(
        exp_dir="/data/expA", name="Set A", in_table=True,
        shocks=[
            ShockOverride(num=1, include=True, weight_max=[80.0]),
            ShockOverride(num=2, include=False),
        ],
    )
    state = SessionState(load_full_series=True, series=[series_ov])

    back = SessionState.from_yaml_text(state.to_yaml_text())

    assert back.load_full_series is True
    assert back.series[0].exp_dir == "/data/expA"
    assert back.series[0].in_table is True
    assert back.series[0].shocks[0].include is True
    assert back.series[0].shocks[1].include is False


def test_capture_restore_reaction_state_equivalence(cycloheptane_paths, tmp_path):
    """Toggles + uncertainties captured from one mech restore onto a fresh
    load of the same file via signature matching.
    """
    target = 0
    F_rate = 3.0
    F_coef = 2.5

    source = MechanismLoader().load(cycloheptane_paths)
    builder = OptimizableSetBuilder()
    builder.set_reaction_optimizable(target, True)
    source.rate_bnds[target]["value"] = F_rate
    source.rate_bnds[target]["type"] = "F"

    touched = []
    for bnds_key, sub in source.coeffs_bnds[target].items():
        for coef_name in sub:
            if not isinstance(coef_name, str):
                continue

            sub[coef_name]["value"] = F_coef
            sub[coef_name]["type"] = "F"
            builder.set_coefficient_optimizable(target, bnds_key, coef_name, True)
            touched.append((bnds_key, coef_name))

    assert touched, "fixture reaction has no string-named coefficients"

    reactions = reactions_from_mech(source, builder)
    state = SessionState(reactions=reactions)
    restored_state = SessionState.from_yaml_text(state.to_yaml_text())

    fresh_paths = dict(cycloheptane_paths)
    fresh_paths["Cantera_Mech"] = tmp_path / "fresh.yaml"
    fresh = MechanismLoader().load(fresh_paths)
    fresh_builder = OptimizableSetBuilder()

    restored, partial = restore_state(
        fresh, fresh_builder, restored_state.to_restore_snapshot(),
    )

    assert target in restored
    assert target not in partial
    assert fresh_builder.is_reaction_optimizable(target) is True
    assert fresh.rate_bnds[target]["value"] == F_rate
    for bnds_key, coef_name in touched:
        assert fresh_builder.is_coefficient_optimizable(target, bnds_key, coef_name)
        assert fresh.coeffs_bnds[target][bnds_key][coef_name]["value"] == F_coef
