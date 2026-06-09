"""Mechanism file-format conversion + ``ct.Solution`` build.

Splits the I/O concerns out of ``ChemicalMechanism`` — format detection,
Chemkin/CTI/CTML conversion, atomic file replacement, validation —
producing a populated ``ChemicalMechanism`` ready for runtime.

Failure raises ``MechanismLoadError``. Info messages collected on
``MechanismLoader.messages`` for callers that want them (e.g., GUI log
panel).
"""
import os

import cantera as ct
from cantera import ck2yaml, cti2yaml

from frhodo.common.errors import MechanismLoadError
from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism


def _atomic_convert(target, do_convert):
    """Write ``target`` via a sibling ``.tmp`` followed by ``os.replace``.

    Guarantees readers either see the prior content or the fully
    converted output — never a half-written file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    result = do_convert(tmp)
    os.replace(tmp, target)

    return result


def _chemkin_to_cantera(paths):
    """Convert a Chemkin mech + optional thermo file into ``paths["Cantera_Mech"]``."""
    kwargs = dict(transport_file=None, surface_file=None,
                  phase_name="gas", quiet=False, permissive=True)
    if paths["thermo"] is not None:
        kwargs["thermo_file"] = paths["thermo"]

    return _atomic_convert(
        paths["Cantera_Mech"],
        lambda out: ck2yaml.convert(paths["mech"], out_name=out, **kwargs),
    )


def _resolve_yaml_path(paths) -> str:
    """Convert non-YAML formats and return the path to the YAML to load.

    Raises:
        MechanismLoadError: For unsupported source formats (CTML, XML).
    """
    suffix = paths["mech"].suffix
    if suffix in (".yaml", ".yml"):
        return str(paths["mech"])

    yaml_path = str(paths["Cantera_Mech"])

    if suffix == ".cti":
        _atomic_convert(
            paths["Cantera_Mech"],
            lambda out: cti2yaml.convert(paths["mech"], out),
        )
    elif suffix in (".ctml", ".xml"):
        raise MechanismLoadError(f"{suffix} format is not supported")
    else:  # chemkin
        _chemkin_to_cantera(paths)

    return yaml_path


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
                "Cantera_Mech": Path}``. ``"Cantera_Mech"`` is the
                target for the generated YAML.
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
            yaml_path = _resolve_yaml_path(paths)
        except MechanismLoadError:
            raise
        except Exception as e:
            raise MechanismLoadError(f"Error converting mechanism: {e}") from e

        try:
            yaml_txt = paths["Cantera_Mech"].read_text()
            gas = ct.Solution(yaml=yaml_txt)
        except Exception as e:
            raise MechanismLoadError(
                f"Error in loading mech\n{e}"
            ) from e

        if mech is None:
            mech = ChemicalMechanism()
        mech.gas = gas
        mech.isLoaded = True
        mech.set_rate_expression_coeffs()
        mech.set_thermo_expression_coeffs()

        if not self.silent:
            self.messages.append(
                f'Wrote YAML mechanism file to {paths["Cantera_Mech"]}.'
            )
            self.messages.append(
                f"Mechanism contains {gas.n_species} species "
                f"and {gas.n_reactions} reactions."
            )

        return mech
