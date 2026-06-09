"""Bootstrap-known paths for the GUI app.

``RuntimePaths`` holds the file-system locations that are fixed at app
startup: package root, appdata directory, derived YAML/graphics paths.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RuntimePaths(BaseModel):
    """Fixed-at-startup file-system paths.

    Built via :meth:`from_package` so derived fields (graphics,
    default_config, cantera_mech) can never drift from their parents.
    """

    package: Path = Field(
        description="The ``frhodo.gui`` package directory; UI files and graphics live under it.",
    )
    main: Path = Field(
        description="The ``frhodo/`` package root. Editable installs use this for sibling resources.",
    )
    appdata: Path = Field(
        description="User config directory (``platformdirs.user_config_dir``); persisted YAML lives here.",
    )
    default_config: Path = Field(
        description="``appdata/default_config.yaml`` — the persisted ``FrhodoConfig`` document.",
    )
    cantera_mech: Path = Field(
        description="``appdata/generated_mech.yaml`` — Cantera-converted mechanism cache.",
    )
    graphics: Path = Field(
        description="``package/ui/graphics`` — icons and decorations for the GUI.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @classmethod
    def from_package(cls, package: Path, appdata: Path) -> "RuntimePaths":
        """Derive a :class:`RuntimePaths` from package + appdata roots.

        Derived fields are constructed from the two inputs; callers
        should not pass them directly so they can't drift.
        """
        return cls(
            package=package,
            main=package.parent,
            appdata=appdata,
            default_config=appdata / "default_config.yaml",
            cantera_mech=appdata / "generated_mech.yaml",
            graphics=package / "ui" / "graphics",
        )
