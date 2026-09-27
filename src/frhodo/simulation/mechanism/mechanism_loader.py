"""Mechanism file-format conversion + ``ct.Solution`` build.

Splits the I/O concerns out of ``ChemicalMechanism`` — format detection,
Chemkin/CTI/CTML conversion, atomic file replacement, validation —
producing a populated ``ChemicalMechanism`` ready for runtime.

Failure raises ``MechanismLoadError``. Info messages collected on
``MechanismLoader.messages`` for callers that want them (e.g., GUI log
panel).
"""
import contextlib
import io
import logging
import os
import pathlib
from typing import NamedTuple

import cantera as ct
from cantera import ck2yaml, cti2yaml, ctml2yaml

from frhodo.common.errors import MechanismLoadError
from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism



FORMAT_TABLE = {
    ".yaml": "native",
    ".yml": "native",
    ".cti": "cti2yaml",
    ".ctml": "ctml2yaml",
    ".xml": "ctml2yaml",
    ".inp": "chemkin",
    ".ck": "chemkin",
    ".mech": "chemkin",
}


def supported_mech_suffixes() -> tuple[str, ...]:
    """Lowercase, dot-prefixed suffixes ``FORMAT_TABLE`` dispatches on."""
    return tuple(FORMAT_TABLE)


class _ResolvedYaml(NamedTuple):
    """Where the YAML to load lives, plus what the conversion had to say.

    Attributes:
        path: The YAML file the ``ct.Solution`` is built from.
        wrote_target: ``True`` when ``paths["Cantera_Mech"]`` was written.
        diagnostics: Converter output, empty when it said nothing.
    """

    path: str
    wrote_target: bool
    diagnostics: str


# Cantera opens files through narrow paths, which Windows reads in its ANSI
# code page rather than UTF-8.
_ON_WINDOWS = os.name == "nt"


def _build_solution(resolved):
    """Build the ``ct.Solution`` for a resolved YAML file.

    Converter output is one self-contained file, parsed from its text in
    memory. Cantera caches the files it opens by name and modification time,
    and the conversion target is rewritten for every load, so opening it by
    path can hand back the mechanism converted before it. The text is read in
    the default encoding, the one the converters write with. A native YAML
    source is opened by path, so files it includes resolve against its folder.
    """
    if resolved.wrote_target:
        text = pathlib.Path(resolved.path).read_text()
        gas = ct.Solution(yaml=text)
    else:
        gas = ct.Solution(resolved.path)

    return gas


def _windows_path_note(resolved):
    """Name the likely cause when Cantera cannot open a native YAML path on Windows."""
    if resolved.wrote_target or not _ON_WINDOWS or resolved.path.isascii():
        return ""

    note = (
        "\nThe mechanism's path contains non-ASCII characters, which Cantera may not "
        "be able to open on Windows. A folder whose path is plain ASCII avoids this."
    )

    return note


class _LogCollector(logging.Handler):
    """Accumulates formatted records for the life of one converter call."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _run_converter(convert):
    """Run a Cantera converter with both of its diagnostic channels captured.

    The converters split their output across two channels, printing and
    the ``cantera`` logger, and both bypass a GUI user. Some also call
    ``sys.exit`` on malformed input, an exit that unwinds straight through
    a caller's ``except Exception``.

    Returns:
        Everything the converter emitted, empty when it said nothing.

    Raises:
        MechanismLoadError: The converter exited; the exit code alone says
            nothing, so the diagnostics travel with it.
    """
    stream = io.StringIO()
    collector = _LogCollector()
    logger = logging.getLogger("cantera")
    previous_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.INFO)

    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            convert()
    except SystemExit as e:
        raise MechanismLoadError(
            f"Mechanism conversion failed (exit code {e.code}).\n"
            f"{_collected_text(stream, collector)}"
        ) from e
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)

    diagnostics = _collected_text(stream, collector)

    return diagnostics


def _collected_text(stream, collector):
    """Join the printed and logged output of one converter call."""
    parts = [stream.getvalue(), *collector.lines]
    text = "\n".join(part.strip() for part in parts if part.strip())

    return text


def _atomic_convert(target, do_convert):
    """Write ``target`` via a sibling ``.tmp`` followed by ``os.replace``.

    Readers see either the previous content or the fully converted
    output, never a half-written file, and a failed conversion leaves no
    ``.tmp`` behind.

    Returns:
        The diagnostics the converter emitted.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")

    try:
        diagnostics = _run_converter(lambda: do_convert(tmp))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

    return diagnostics


def _chemkin_to_cantera(paths):
    """Convert a Chemkin mech + optional thermo/transport files into ``paths["Cantera_Mech"]``."""
    kwargs = dict(phase_name="gas", quiet=False, permissive=True)
    for key, arg in (("thermo", "thermo_file"), ("transport", "transport_file")):
        if paths.get(key) is not None:
            kwargs[arg] = paths[key]

    diagnostics = _atomic_convert(
        paths["Cantera_Mech"],
        lambda out: ck2yaml.convert(paths["mech"], out_name=out, **kwargs),
    )

    return diagnostics


def _resolve_yaml_path(paths) -> _ResolvedYaml:
    """Convert non-YAML formats and report the YAML to load.

    Returns:
        A :class:`_ResolvedYaml`. A native YAML source is loaded in place
        and never touches ``paths["Cantera_Mech"]``.

    Raises:
        MechanismLoadError: For unsupported source formats.
    """
    suffix = paths["mech"].suffix.lower()
    kind = FORMAT_TABLE.get(suffix)
    if kind == "native":
        native = _ResolvedYaml(path=str(paths["mech"]), wrote_target=False, diagnostics="")

        return native

    if kind is None:
        supported = ", ".join(sorted(FORMAT_TABLE))
        raise MechanismLoadError(
            f"'{suffix}' is not a supported mechanism format; supported: {supported}"
        )

    if kind == "cti2yaml":
        diagnostics = _atomic_convert(
            paths["Cantera_Mech"],
            lambda out: cti2yaml.convert(paths["mech"], out),
        )
    elif kind == "ctml2yaml":
        diagnostics = _atomic_convert(
            paths["Cantera_Mech"],
            lambda out: ctml2yaml.convert(paths["mech"], out),
        )
    else:  # chemkin
        # Not a fallback detector: ck2yaml.convert runs with permissive=True and does not
        # reject non-Chemkin text (measured), so an unlisted suffix would surface as a
        # Cantera error about a generated file the user never made.
        diagnostics = _chemkin_to_cantera(paths)

    converted = _ResolvedYaml(
        path=str(paths["Cantera_Mech"]), wrote_target=True, diagnostics=diagnostics,
    )

    return converted


class MechanismLoader:
    """Convert source files to YAML, build a ``ct.Solution``, return a mech.

    Attributes:
        silent: When ``True``, suppress the ``messages`` log.
        messages: Free-form info log populated during :meth:`load`.
    """

    def __init__(self, silent: bool = False):
        self.silent = silent
        self.messages: list[str] = []

    def load(
        self, paths: dict, mech: ChemicalMechanism | None = None,
    ) -> ChemicalMechanism:
        """Convert and load a mechanism.

        Args:
            paths: ``{"mech": Path, "thermo": Path | None,
                "transport": Path | None, "Cantera_Mech": Path}``.
                ``"Cantera_Mech"`` is the target for the generated YAML.
                ``"transport"`` is ignored for non-Chemkin sources.
            mech: Existing :class:`ChemicalMechanism` to populate in
                place; useful when callers hold a reference (e.g. the
                GUI). A fresh instance is created when ``None``.

        Returns:
            The populated :class:`ChemicalMechanism` — same instance as
            ``mech`` when supplied.

        Raises:
            MechanismLoadError: Conversion or ``ct.Solution`` build
                failed.
        """
        try:
            resolved = _resolve_yaml_path(paths)
        except MechanismLoadError:
            raise
        except Exception as e:
            raise MechanismLoadError(f"Error converting mechanism: {e}") from e

        try:
            gas = _build_solution(resolved)
        except Exception as e:
            diagnostics_note = ""
            if resolved.diagnostics:
                diagnostics_note = f"\n{resolved.diagnostics}"

            raise MechanismLoadError(
                f"Error in loading mech\n{e}{diagnostics_note}{_windows_path_note(resolved)}"
            ) from e

        if mech is None:
            mech = ChemicalMechanism()
        mech.gas = gas
        mech.isLoaded = True
        mech.set_rate_expression_coeffs()
        mech.set_thermo_expression_coeffs()

        if not self.silent:
            if resolved.diagnostics:
                self.messages.append(resolved.diagnostics)
            if resolved.wrote_target:
                self.messages.append(
                    f'Wrote YAML mechanism file to {paths["Cantera_Mech"]}.'
                )
            self.messages.append(
                f"Mechanism contains {gas.n_species} species "
                f"and {gas.n_reactions} reactions."
            )

        return mech
