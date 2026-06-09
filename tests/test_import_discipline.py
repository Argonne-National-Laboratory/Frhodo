"""Layered architecture lint.

Walks every ``frhodo/`` module and asserts the import-direction rules:

    common/      may import: stdlib, numpy, scipy, cantera, pydantic
    simulation/  may import: common/
                 (within simulation: shock/ may import mechanism/)
    experiment/  may import: common/
    optimize/    may import: common/, simulation/, experiment/
    api/         may import: common/, simulation/, experiment/, optimize/
    gui/         may import: everything below

The ``_vendor/`` tree is neutral and importable from any layer.

AST-based, not runtime: a module that imports a forbidden layer fails
this test even if the offending import is inside a ``try`` block or a
function.
"""
import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "frhodo"

LAYER_FORBIDDEN = {
    "common": {
        "frhodo.simulation", "frhodo.experiment", "frhodo.optimize",
        "frhodo.api", "frhodo.gui",
    },
    "simulation": {
        "frhodo.experiment", "frhodo.optimize", "frhodo.api", "frhodo.gui",
    },
    "experiment": {
        "frhodo.simulation", "frhodo.optimize", "frhodo.api", "frhodo.gui",
    },
    "optimize": {"frhodo.api", "frhodo.gui"},
    "api": {"frhodo.gui"},
    "gui": set(),
}


def _layer_of(module_path: pathlib.Path) -> str | None:
    rel = module_path.relative_to(PACKAGE_ROOT).parts
    if not rel:
        return None
    head = rel[0]
    if head in LAYER_FORBIDDEN:
        return head
    return None


def _imports(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _modules_in_layer(layer: str) -> list[pathlib.Path]:
    layer_dir = PACKAGE_ROOT / layer
    if layer_dir.is_dir():
        return [p for p in layer_dir.rglob("*.py") if "__pycache__" not in p.parts]
    file_module = PACKAGE_ROOT / f"{layer}.py"
    if file_module.is_file():
        return [file_module]
    return []


def _violations_for(module_path: pathlib.Path, forbidden: set[str]) -> list[str]:
    source = module_path.read_text()
    bad = []
    for imp in _imports(source):
        for prefix in forbidden:
            if imp == prefix or imp.startswith(prefix + "."):
                bad.append(imp)
                break
    return bad


@pytest.mark.parametrize("layer", sorted(LAYER_FORBIDDEN))
def test_layer_imports_only_below(layer):
    forbidden = LAYER_FORBIDDEN[layer]
    if not forbidden:
        pytest.skip(f"layer {layer!r} has no upward layers to forbid")

    offenders: dict[str, list[str]] = {}
    for module in _modules_in_layer(layer):
        bad = _violations_for(module, forbidden)
        if bad:
            offenders[str(module.relative_to(PACKAGE_ROOT))] = bad

    if offenders:
        lines = [f"layer {layer!r} has forbidden imports:"]
        for fp, bad in offenders.items():
            lines.append(f"  {fp}: {bad}")
        pytest.fail("\n".join(lines))


@pytest.mark.parametrize("module", [
    "frhodo.api",
    "frhodo.optimize",
    "frhodo.optimize.algorithms",
    "frhodo.optimize.residual",
])
def test_module_does_not_import_qt(module):
    """The headless surface must not pull in qtpy/PyQt/PySide."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; import {module}; "
         f"forbidden = ['qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6']; "
         f"leaked = [m for m in forbidden if m in sys.modules]; "
         f"assert not leaked, f'leaked Qt: {{leaked}}'"],
        capture_output=True, check=False,
    )
    assert result.returncode == 0, (
        f"{module} import leaked Qt: {result.stderr.decode()}"
    )
