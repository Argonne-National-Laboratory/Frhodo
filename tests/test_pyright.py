"""Type-safety gate: ``pyright`` must pass on the configured surface.

The included paths are listed in ``pyproject.toml :: [tool.pyright]``.
Warnings are ignored; only errors fail the gate. Skip when pyright is
not installed (covers contributors without the dev extras).
"""
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("pyright") is None,
    reason="pyright is not installed; install via `pip install frhodo[dev]`",
)


def test_pyright_clean():
    result = subprocess.run(
        ["pyright"],
        capture_output=True,
        check=False,
        text=True,
    )
    summary = next(
        (
            line for line in reversed(result.stdout.splitlines())
            if "error" in line and "warning" in line
        ),
        "",
    )
    assert "0 errors" in summary, (
        f"pyright reported errors:\n{result.stdout}\n{result.stderr}"
    )
