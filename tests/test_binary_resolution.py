"""Pin the bonmin/ipopt binary discovery contract.

The binaries are vendored at ``frhodo/_vendor/bonmin/`` and
``frhodo/_vendor/ipopt/``; resolved per-platform via
``optimize.algorithms._resolve_binary``. Each must resolve to an
existing executable. The SHA256 pins below guard provenance — see
``frhodo/_vendor/PROVENANCE.md``.
"""
import hashlib
import os
import platform
from pathlib import Path

import pytest

import frhodo
from frhodo.optimize.algorithms import path



VENDOR = Path(frhodo.__file__).parent / "_vendor"

# Provenance pins for Bonmin 1.8.7 / Ipopt 3.12.13, recorded in
# _vendor/PROVENANCE.md.
VENDOR_BINARY_SHA256 = {
    "bonmin/bonmin-linux64/bonmin": "e7b5a7faffea45ff0833e2458113227d5baff5e812015e41e6b52415fedb7ce9",
    "bonmin/bonmin-osx/bonmin": "4318915a6a5d4e255ec070b85e5ee3339a83d818066f48df19057357222769ad",
    "bonmin/bonmin-win64/bonmin.exe": "bf60d34f0a8c02810c6eb53e307ac5f704013150ece545de1d959961fd50c1dc",
    "bonmin/bonmin-win64/libipoptfort.dll": "13efbf4defc057e7e0f5ab0ed85b914e436816e06fb17d09dbcacce2bba466d0",
    "ipopt/ipopt-linux64/ipopt": "5324be632e28cb99b5747003e2b174f10132935dea6ec52d2fc755d1d7e3fd45",
    "ipopt/ipopt-osx/ipopt": "42e813453aba45772a7a4e6793843ddad31de35e576dfc7982d053c281170ad1",
    "ipopt/ipopt-win64/ipopt.exe": "fbfb59416249aeb64c2227712e105abdc96b845df1b685e7273b25fcc883b3b5",
    "ipopt/ipopt-win64/libipoptfort.dll": "7738c924592c6abca75a98be0b374cf280967e63483bdcfad9d1a1fa28c1f361",
}


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


class TestVendorBinaryProvenance:
    """Every vendored solver binary matches its recorded SHA256, so a
    swapped or corrupted artifact is caught in the suite."""

    @pytest.mark.parametrize("rel_path", sorted(VENDOR_BINARY_SHA256))
    def test_binary_matches_recorded_sha256(self, rel_path):
        binary = VENDOR / rel_path
        assert binary.is_file(), f"vendored binary missing: {rel_path}"
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert digest == VENDOR_BINARY_SHA256[rel_path], (
            f"{rel_path} SHA256 drifted from PROVENANCE.md: {digest}"
        )
