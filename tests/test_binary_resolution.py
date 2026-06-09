"""Pin the bonmin/ipopt binary discovery contract.

The binaries are vendored at ``frhodo/_vendor/bonmin/`` and
``frhodo/_vendor/ipopt/``; resolved per-platform via
``optimize.algorithms._resolve_binary``. Each must resolve to an
existing executable.
"""
import os
import platform

import pytest

from frhodo.optimize.algorithms import path


@pytest.mark.skipif(
    platform.system() not in {"Linux", "Darwin", "Windows"},
    reason="binary discovery only covers the three packaged platforms",
)
class TestBinaryResolution:
    @pytest.mark.parametrize("name", ["bonmin", "ipopt"])
    def test_resolved_binary_exists(self, name):
        assert path[name].exists(), (
            f"{name} binary did not resolve to an existing file: "
            f"{path[name]}"
        )

    @pytest.mark.parametrize("name", ["bonmin", "ipopt"])
    def test_resolved_binary_executable(self, name):
        if platform.system() == "Windows":
            pytest.skip("executable bit semantics are POSIX-only")

        assert os.access(path[name], os.X_OK), (
            f"{name} binary at {path[name]} is not executable"
        )
