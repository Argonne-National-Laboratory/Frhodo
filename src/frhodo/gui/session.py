"""Whole-session save/load.

A session file captures everything needed to restore a working GUI
state after a crash: directory paths, the full user preference config,
per-shock weight/offset overrides, the current selection, and
per-reaction uncertainty + optimizable state.

Reaction state is keyed by reaction signature (reactants, products,
reversibility) rather than index, so it survives mechanism re-ordering;
the apply path reuses :func:`frhodo.simulation.mechanism.mech_snapshot.restore_state`,
which already handles signature matching, duplicates, and partial-match
flagging.

Loading applies to the live session only; it never overwrites the
persisted ``default_config.yaml``.
"""
import configparser
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from frhodo.common.config import FrhodoConfig
from frhodo.simulation.mechanism.mech_snapshot import restore_state



SESSION_SUFFIX = ".frhodo"
AUTOSAVE_NAME = "session_autosave" + SESSION_SUFFIX


def _finite(values) -> bool:
    """True when at least one entry is a finite number."""
    return any(v is not None and np.isfinite(v) for v in values)


class CoefState(BaseModel):
    """One coefficient's uncertainty + optimizable flag."""

    bnds_key: str
    coef_name: str | int
    value: float | None = None
    type: str = "F"
    reset_val: float | None = None
    optimizable: bool = False

    model_config = ConfigDict(extra="forbid")


class ReactionState(BaseModel):
    """Per-reaction engine state, keyed by stoichiometric signature.

    ``reactants`` / ``products`` are ``[species, coefficient]`` pairs;
    together with ``reversible`` they reconstruct the same signature
    tuple :func:`mech_snapshot.rxn_signature` produces.
    """

    reactants: list[tuple[str, float]]
    products: list[tuple[str, float]]
    reversible: bool
    rate_unc_value: float | None = None
    rate_unc_type: str = "F"
    rate_optimizable: bool = False
    coefs: list[CoefState] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def signature(self) -> tuple:
        reactants = tuple(sorted((s, float(c)) for s, c in self.reactants))
        products = tuple(sorted((s, float(c)) for s, c in self.products))
        reversible = bool(self.reversible)
        sig = (reactants, products, reversible)

        return sig


class ShockOverride(BaseModel):
    """Per-shock weight-function and timing customizations.

    ``filename`` is the experiment file stem, stored alongside ``num``
    so a restore can confirm it lands on the same shock.
    """

    num: int
    filename: str = ""
    include: bool = True
    weight_max: list[float] = Field(default_factory=list)
    weight_min: list[float] = Field(default_factory=list)
    weight_shift: list[float] = Field(default_factory=list)
    weight_k: list[float] = Field(default_factory=list)
    observable_main: str = ""
    observable_sub: int | str | None = None

    model_config = ConfigDict(extra="forbid")


class SeriesOverride(BaseModel):
    """One loaded experiment series and its per-shock state.

    ``exp_dir`` is the experiment directory used both to identify the
    series and to reload it; ``in_table`` records whether it was added
    to the Series Viewer.
    """

    exp_dir: str
    name: str = ""
    in_table: bool = False
    shocks: list[ShockOverride] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SessionPaths(BaseModel):
    exp_main: str = ""
    mech_main: str = ""
    sim_main: str = ""
    mech_file: str = ""
    series_name: str = ""
    species_aliases: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SessionSelection(BaseModel):
    active_exp_dir: str = ""
    shock_num: int = 1
    time_offset: float = 0.0  # time-offset box value, in display units

    model_config = ConfigDict(extra="forbid")


class SessionState(BaseModel):
    """Root model for a ``*.frhodo`` session file."""

    schema_version: Literal[1] = 1
    comment: str = ""
    load_full_series: bool = False
    paths: SessionPaths = Field(default_factory=SessionPaths)
    config: FrhodoConfig = Field(default_factory=FrhodoConfig)
    selection: SessionSelection = Field(default_factory=SessionSelection)
    series: list[SeriesOverride] = Field(default_factory=list)
    reactions: list[ReactionState] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @classmethod
    def from_yaml_text(cls, text: str) -> "SessionState":
        data = yaml.safe_load(text) or {}

        return cls.model_validate(data)

    def to_yaml_text(self) -> str:
        text = yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        )

        return text

    def to_restore_snapshot(self) -> dict:
        """Rebuild the signature-keyed snapshot ``restore_state`` consumes.

        Reactions sharing a signature stack in list order, matching
        :func:`mech_snapshot.capture_state`.
        """
        snapshot: dict[tuple, list[dict]] = {}
        for rxn in self.reactions:
            coef_state = {}
            for coef in rxn.coefs:
                coef_state[(coef.bnds_key, coef.coef_name)] = {
                    "value": coef.value,
                    "type": coef.type,
                    "resetVal": coef.reset_val,
                    "optimizable": coef.optimizable,
                }
            state = {
                "coeffs": None,
                "rate_unc": {
                    "value": rxn.rate_unc_value,
                    "type": rxn.rate_unc_type,
                },
                "rate_optimizable": rxn.rate_optimizable,
                "coef_state": coef_state,
            }
            snapshot.setdefault(rxn.signature(), []).append(state)

        return snapshot


def reactions_from_mech(mech, optimizables) -> list[ReactionState]:
    """Capture per-reaction state from a loaded mech into serializable form."""
    reactions: list[ReactionState] = []
    for rxnIdx in range(mech.gas.n_reactions):
        rxn = mech.gas.reaction(rxnIdx)
        coefs: list[CoefState] = []
        for bnds_key, sub in mech.coeffs_bnds[rxnIdx].items():
            for coef_name, coef_dict in sub.items():
                coefs.append(CoefState(
                    bnds_key=bnds_key,
                    coef_name=coef_name,
                    value=coef_dict["value"],
                    type=coef_dict["type"],
                    reset_val=coef_dict["resetVal"],
                    optimizable=optimizables.is_coefficient_optimizable(
                        rxnIdx, bnds_key, coef_name,
                    ),
                ))
        reactions.append(ReactionState(
            reactants=[(s, float(c)) for s, c in rxn.reactants.items()],
            products=[(s, float(c)) for s, c in rxn.products.items()],
            reversible=bool(rxn.reversible),
            rate_unc_value=mech.rate_bnds[rxnIdx]["value"],
            rate_unc_type=mech.rate_bnds[rxnIdx]["type"],
            rate_optimizable=optimizables.is_reaction_optimizable(rxnIdx),
            coefs=coefs,
        ))

    return reactions


def _shock_override(shock) -> ShockOverride:
    if isinstance(shock.path, dict):
        exp_path = shock.path.get("exp_data")
    else:
        exp_path = None

    if exp_path:
        filename = Path(exp_path).stem
    else:
        filename = ""

    override = ShockOverride(
        num=int(shock.num),
        filename=filename,
        include=bool(shock.include),
        weight_max=[float(v) for v in shock.weight_max],
        weight_min=[float(v) for v in shock.weight_min],
        weight_shift=[float(v) for v in shock.weight_shift],
        weight_k=[float(v) for v in shock.weight_k],
        observable_main=shock.observable.get("main", ""),
        observable_sub=shock.observable.get("sub"),
    )

    return override


def series_overrides_from_series(series) -> list[SeriesOverride]:
    """Capture every loaded experiment series and its per-shock state.

    Skips the placeholder series (empty ``path``) the GUI seeds before
    any experiment directory is loaded.
    """
    overrides: list[SeriesOverride] = []
    for idx, exp_dir in enumerate(series.path):
        if not exp_dir:
            continue

        shocks = [_shock_override(shock) for shock in series.shock[idx]]
        overrides.append(SeriesOverride(
            exp_dir=str(exp_dir),
            name=series.name[idx],
            in_table=bool(series.in_table[idx]),
            shocks=shocks,
        ))

    return overrides


def apply_shock_overrides(shocks, overrides: list[ShockOverride]) -> None:
    """Restore per-shock overrides onto matching loaded shocks.

    Matches by shock number. Weight arrays are restored only when they
    carry finite values, so untouched (NaN-default) shocks keep whatever
    the config seeded rather than being clobbered with NaN.
    """
    by_num = {shock.num: shock for shock in shocks}
    for ov in overrides:
        shock = by_num.get(ov.num)
        if shock is None:
            continue

        shock.include = ov.include
        if _finite(ov.weight_max):
            shock.weight_max = list(ov.weight_max)
        if _finite(ov.weight_min):
            shock.weight_min = list(ov.weight_min)
        if _finite(ov.weight_shift):
            shock.weight_shift = list(ov.weight_shift)
        if _finite(ov.weight_k):
            shock.weight_k = list(ov.weight_k)

        if ov.observable_main:
            shock.observable["main"] = ov.observable_main
            shock.observable["sub"] = ov.observable_sub


def capture_session(parent, comment: str = "") -> SessionState:
    """Gather the full GUI state into a :class:`SessionState`."""
    parent.user_settings.pull_config_from_boxes()
    config = parent.user_settings.config.model_copy(deep=True)

    mech_path = parent.path.get("mech")
    if mech_path:
        mech_file = Path(mech_path).name
    else:
        mech_file = ""

    paths = SessionPaths(
        exp_main=str(parent.path.get("exp_main", "")),
        mech_main=str(parent.path.get("mech_main", "")),
        sim_main=str(parent.path.get("sim_main", "")),
        mech_file=mech_file,
        series_name=parent.display_shock.series_name,
        species_aliases=dict(parent.series.current["species_alias"]),
    )

    active_path = parent.series.path[parent.series.idx]
    if active_path:
        active_exp_dir = str(active_path)
    else:
        active_exp_dir = ""

    selection = SessionSelection(
        active_exp_dir=active_exp_dir,
        shock_num=parent.shock_selection.current,
        time_offset=parent.time_offset_box.value(),
    )
    series = series_overrides_from_series(parent.series)
    reactions = reactions_from_mech(parent.mech, parent.optimizables)

    state = SessionState(
        comment=comment,
        load_full_series=bool(parent.load_state.load_full_series),
        paths=paths,
        config=config,
        selection=selection,
        series=series,
        reactions=reactions,
    )

    return state


def _apply_paths(parent, paths: SessionPaths) -> None:
    """Drive the directory boxes via the proven Dir-file load path.

    Builds a transient Dir.ini and routes it through
    ``Path.load_dir_file`` so the mechanism load fires exactly as it does
    for a user-opened directory file, then selects the recorded mech
    file. The experiment directory is left blank — series loading is
    replayed explicitly in :func:`apply_session` so the Series Viewer's
    "last-loaded is active" assumption always holds.
    """
    cfg = configparser.RawConfigParser()
    cfg["Directories"] = {
        "exp_main": "",
        "mech_main": paths.mech_main,
        "sim_main": paths.sim_main,
    }
    cfg["Species Default Aliases"] = {"aliases": ""}
    cfg["Experiment Set Name"] = {"name": ""}

    with tempfile.NamedTemporaryFile(
        "w", suffix=".ini", delete=False, encoding="utf-8",
    ) as f:
        cfg.write(f)
        tmp_path = f.name

    parent.path_set.load_dir_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if paths.mech_file and parent.mech_select_comboBox.findText(paths.mech_file) >= 0:
        parent.mech_select_comboBox.setCurrentText(paths.mech_file)
        parent.load_mech()


def _series_index(series, exp_dir: str):
    """Index of the loaded series matching ``exp_dir``, or ``None``."""
    for i, path in enumerate(series.path):
        if path and str(path) == exp_dir:
            return i

    return None


def _replay_series(parent, series_override: SeriesOverride) -> None:
    """Load one series, restore its per-shock state, and add it to the viewer.

    Setting the experiment box reproduces a user-typed path, which
    triggers ``add_series``; the just-loaded series is therefore last,
    so ``_add_series_table`` adds the correct one.
    """
    parent.exp_series_name_box.setText(series_override.name)
    parent.exp_main_box.setPlainText(series_override.exp_dir)
    parent.app.processEvents()

    idx = _series_index(parent.series, series_override.exp_dir)
    if idx is None:
        return

    apply_shock_overrides(parent.series.shock[idx], series_override.shocks)

    if series_override.in_table:
        parent.series_viewer._add_series_table(None)


def apply_session(parent, state: SessionState) -> tuple[set, set]:
    """Restore a :class:`SessionState` onto the live GUI.

    Returns ``(restored, partial)`` reaction-index sets from
    :func:`restore_state` so the caller can report match quality.
    """
    parent.load_full_series_box.setChecked(state.load_full_series)
    parent.load_state.load_full_series = state.load_full_series

    _apply_paths(parent, state.paths)
    parent.app.processEvents()

    parent.user_settings.config = state.config.model_copy(deep=True)
    parent.user_settings.apply_config_to_boxes()

    # Set the time-offset box before series load so every shock created
    # by ``_create_shock`` is seeded with it; the config above already
    # restored the time-unit box this value is expressed in.
    parent.time_offset_box.setValue(state.selection.time_offset)

    for series_override in state.series:
        _replay_series(parent, series_override)

    if state.selection.active_exp_dir:
        parent.exp_main_box.setPlainText(state.selection.active_exp_dir)
        parent.app.processEvents()

    parent.optimizables.reset()
    restored, partial = restore_state(
        parent.mech, parent.optimizables, state.to_restore_snapshot(),
    )
    parent.tree.set_trees(parent.mech, partial_match_idxs=partial)

    parent.shock_choice_box.setValue(state.selection.shock_num)
    parent.weight.set_boxes()

    # ``time_unc_box`` does not drive ``update_user_settings``, so push the
    # restored uncertainty boxes onto the live state the optimizer reads.
    t_unit_conv = parent.reactor_state.t_unit_conv
    parent.time_uncertainty.value = parent.time_unc_box.value() * t_unit_conv
    parent.time_uncertainty.random = parent.random_t_unc_box.isChecked()

    return restored, partial


def write_session_file(parent, file_path, comment: str = "") -> None:
    """Capture and serialize the current session to ``file_path``."""
    state = capture_session(parent, comment=comment)
    Path(file_path).write_text(state.to_yaml_text(), encoding="utf-8")


def read_session_file(parent, file_path) -> tuple[set, set]:
    """Load a session file and apply it to the GUI."""
    text = Path(file_path).read_text(encoding="utf-8")
    state = SessionState.from_yaml_text(text)

    return apply_session(parent, state)
