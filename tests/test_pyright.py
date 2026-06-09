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
    last_line = (result.stdout.strip().splitlines() or [""])[-1]
    assert "0 errors" in last_line, (
        f"pyright reported errors:\n{result.stdout}\n{result.stderr}"
    )
