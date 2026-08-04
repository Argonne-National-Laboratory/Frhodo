"""Tests for ``frhodo.simulation.numerics.sundials``.

Component-level checks for the ctypes binding against Cantera's
bundled SUNDIALS — RAII wrappers, numpy-view aliasing, vector
tolerances, dense-output interpolation, and end-to-end CVODE forward
solves of toy ODEs with known analytical or reference solutions.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from frhodo.simulation.numerics import sundials
from frhodo.simulation.numerics.sundials import (
    CVodeIntegrator,
    DenseLinearSolver,
    DenseMatrix,
    NVector,
    SundialsContext,
)



@pytest.fixture
def ctx() -> SundialsContext:
    return SundialsContext()



class TestCondaSundialsDiscovery:
    """Locating conda-forge's separately-packaged SUNDIALS libraries.

    conda-forge Cantera links a standalone ``sundials`` package whose
    libraries live in the environment (``Library/bin`` on Windows,
    ``lib`` elsewhere) rather than bundled beside the Cantera extension,
    and split across more objects than the wheel — the dense linear
    solver is its own library.
    """

    def _make_env(self, tmp_path, platform, filenames):
        if platform == "win32":
            subdir = "Library/bin"
        else:
            subdir = "lib"
        lib_dir = tmp_path / subdir
        lib_dir.mkdir(parents=True)
        for name in filenames:
            (lib_dir / name).write_bytes(b"")

        return lib_dir

    def test_finds_core_libs_in_dependency_order(self, tmp_path, monkeypatch):
        self._make_env(tmp_path, "linux", [
            "libsundials_cvodes.so.7",
            "libsundials_core.so.7",
            "libsundials_nvecserial.so.7",
        ])
        monkeypatch.setattr(sundials.sys, "platform", "linux")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        found = [p.name for p in sundials._find_sundials_conda_libs()]

        assert found == [
            "libsundials_nvecserial.so.7",
            "libsundials_core.so.7",
            "libsundials_cvodes.so.7",
        ], "core libraries must lead, in dependency order"

    def test_appends_extra_sundials_objects_like_the_dense_solver(
        self, tmp_path, monkeypatch,
    ):
        self._make_env(tmp_path, "win32", [
            "sundials_nvecserial.dll",
            "sundials_core.dll",
            "sundials_cvodes.dll",
            "sundials_sunlinsoldense.dll",
        ])
        monkeypatch.setattr(sundials.sys, "platform", "win32")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        found = [p.name for p in sundials._find_sundials_conda_libs()]

        assert found[:3] == [
            "sundials_nvecserial.dll",
            "sundials_core.dll",
            "sundials_cvodes.dll",
        ]
        assert "sundials_sunlinsoldense.dll" in found[3:], (
            "the separately-packaged dense solver must be loaded too"
        )
        assert len(found) == len(set(found)), "no library listed twice"

    def test_windows_probes_library_bin(self, tmp_path, monkeypatch):
        self._make_env(tmp_path, "win32", ["sundials_cvodes.dll"])
        monkeypatch.setattr(sundials.sys, "platform", "win32")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        assert sundials._conda_sundials_dir() == tmp_path / "Library" / "bin"

    def test_empty_when_no_sundials_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sundials.sys, "platform", "linux")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        assert sundials._conda_sundials_dir() is None
        assert sundials._find_sundials_conda_libs() == []

    def test_no_false_positive_from_unrelated_libs(self, tmp_path, monkeypatch):
        self._make_env(tmp_path, "linux", ["libopenblas.so", "libpython3.so"])
        monkeypatch.setattr(sundials.sys, "platform", "linux")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        assert sundials._find_sundials_conda_libs() == []

    def test_falls_back_to_sys_prefix_when_unactivated(
        self, tmp_path, monkeypatch,
    ):
        """A shortcut can launch the env's python without activation, so
        CONDA_PREFIX is unset but sys.prefix points at the env root."""
        self._make_env(tmp_path, "linux", ["libsundials_cvodes.so"])
        monkeypatch.setattr(sundials.sys, "platform", "linux")
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(sundials.sys, "prefix", str(tmp_path))

        found = [p.name for p in sundials._find_sundials_conda_libs()]

        assert found == ["libsundials_cvodes.so"]


class TestCondaFallbackGating:
    """The conda search must run only when the Cantera extension does
    not already provide the SUNDIALS symbols. Searching whenever the
    sibling set is empty loads a second SUNDIALS on top of one the
    extension re-exports (Linux/macOS) — it binds but segfaults at
    solve time.
    """

    def test_conda_search_skipped_when_extension_self_provides(
        self, monkeypatch,
    ):
        calls = {"conda": 0}

        def spy_conda():
            calls["conda"] += 1

            return []

        # Extension provides every required symbol on the first load.
        monkeypatch.setattr(
            sundials, "_find_sundials_sibling_libs", lambda ext: [],
        )
        monkeypatch.setattr(sundials, "_find_sundials_conda_libs", spy_conda)
        monkeypatch.setattr(
            sundials, "_find_cantera_extension", lambda: sundials.pathlib.Path("x"),
        )

        class FullLib:
            def __contains__(self, name):
                return True

        monkeypatch.setattr(sundials, "_load_libs", lambda libs, ext: FullLib())
        monkeypatch.setattr(sundials, "_bind_signatures", lambda lib: None)

        sundials._load_sundials()

        assert calls["conda"] == 0, (
            "conda search must not run when the extension self-provides"
        )

    def test_conda_search_runs_when_symbols_missing(self, monkeypatch):
        calls = {"conda": 0}

        def spy_conda():
            calls["conda"] += 1

            return []

        monkeypatch.setattr(
            sundials, "_find_sundials_sibling_libs", lambda ext: [],
        )
        monkeypatch.setattr(sundials, "_find_sundials_conda_libs", spy_conda)
        monkeypatch.setattr(
            sundials, "_find_cantera_extension", lambda: sundials.pathlib.Path("x"),
        )

        class EmptyLib:
            def __contains__(self, name):
                return False

        monkeypatch.setattr(sundials, "_load_libs", lambda libs, ext: EmptyLib())

        with pytest.raises(sundials.SundialsBindingError):
            sundials._load_sundials()

        assert calls["conda"] == 1, (
            "conda search must run when required symbols are missing"
        )


class TestNVector:
    def test_view_aliases_buffer(self, ctx: SundialsContext) -> None:
        v = NVector(4, ctx)
        v.view()[:] = [1.0, 2.0, 3.0, 4.0]
        # Second view call returns same buffer
        assert np.array_equal(v.view(), [1.0, 2.0, 3.0, 4.0])
        # In-place mutation is visible
        v.view()[2] = 99.0
        assert v.view()[2] == 99.0

    def test_from_numpy_roundtrip(self, ctx: SundialsContext) -> None:
        arr = np.linspace(-3.0, 3.0, 7)
        v = NVector.from_numpy(arr, ctx)
        assert np.array_equal(v.view(), arr)
        # Copies the data (not aliased to original)
        arr[0] = 1e9
        assert v.view()[0] != 1e9

    def test_length_property(self, ctx: SundialsContext) -> None:
        v = NVector(12, ctx)
        assert v.length == 12

    def test_handle_is_nonzero(self, ctx: SundialsContext) -> None:
        v = NVector(3, ctx)
        assert v.handle.value is not None
        assert v.handle.value != 0


class TestDenseMatrix:
    def test_view_round_trip(self, ctx: SundialsContext) -> None:
        m = DenseMatrix(3, 3, ctx)
        m.view()[:] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert np.array_equal(m.view(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_shape(self, ctx: SundialsContext) -> None:
        m = DenseMatrix(5, 7, ctx)
        assert m.shape == (5, 7)
        assert m.view().shape == (5, 7)

    def test_zero(self, ctx: SundialsContext) -> None:
        m = DenseMatrix(2, 2, ctx)
        m.view()[:] = [[1, 2], [3, 4]]
        m.zero()
        assert np.array_equal(m.view(), [[0, 0], [0, 0]])

    def test_rectangular_view_is_row_major(self, ctx: SundialsContext) -> None:
        m = DenseMatrix(2, 3, ctx)
        m.view()[0, :] = [10.0, 20.0, 30.0]
        m.view()[1, :] = [40.0, 50.0, 60.0]
        assert m.view()[0, 1] == 20.0
        assert m.view()[1, 2] == 60.0


class TestDenseLinearSolver:
    def test_construct_and_destroy(self, ctx: SundialsContext) -> None:
        v = NVector(4, ctx)
        m = DenseMatrix(4, 4, ctx)
        ls = DenseLinearSolver(v, m, ctx)
        assert ls.handle.value is not None


def _decaying_exponential_rhs(k: float):
    def rhs(t, y, ydot):
        ydot[:] = -k * y

    return rhs


def _decaying_exponential_jac(k: float):
    def jac(t, y, fy, J):
        for i in range(y.size):
            J[i, i] = -k

    return jac


class TestForwardSolve:
    def test_decaying_exponential_matches_analytical(self) -> None:
        k = 3.5
        integ = CVodeIntegrator(
            2,
            _decaying_exponential_rhs(k),
            jac=_decaying_exponential_jac(k),
            rtol=1e-10, atol=1e-13,
        )
        y0 = np.array([2.0, -1.5])
        integ.reinit(0.0, y0)
        t_end = 0.7
        t, y = integ.step_to(t_end)
        assert t == pytest.approx(t_end, abs=1e-12)
        np.testing.assert_allclose(y, y0 * np.exp(-k * t_end), rtol=1e-8)

    def test_harmonic_oscillator_matches_analytical(self) -> None:
        omega = 2.0

        def rhs(t, y, ydot):
            ydot[0] = y[1]
            ydot[1] = -omega * omega * y[0]

        def jac(t, y, fy, J):
            J[0, 1] = 1.0
            J[1, 0] = -omega * omega

        integ = CVodeIntegrator(2, rhs, jac=jac, rtol=1e-10, atol=1e-13)
        integ.reinit(0.0, np.array([1.0, 0.0]))
        t_target = 2.0 * np.pi / omega
        t, y = integ.step_to(t_target)
        np.testing.assert_allclose(y, [1.0, 0.0], atol=1e-7)

    def test_user_jacobian_matches_fd_jacobian(self) -> None:
        """Solving with user-supplied vs internal-FD Jacobian must agree."""
        k = 3.5
        y0 = np.array([2.0, -1.5])
        rhs = _decaying_exponential_rhs(k)
        jac = _decaying_exponential_jac(k)
        rtol, atol = 1e-8, 1e-12

        integ_anal = CVodeIntegrator(2, rhs, jac=jac, rtol=rtol, atol=atol)
        integ_anal.reinit(0.0, y0)
        _, y_anal = integ_anal.step_to(0.7)

        integ_fd = CVodeIntegrator(2, rhs, rtol=rtol, atol=atol)
        integ_fd.reinit(0.0, y0)
        _, y_fd = integ_fd.step_to(0.7)

        np.testing.assert_allclose(y_anal, y_fd, rtol=1e-7, atol=1e-10)

    def test_van_der_pol_stiff_converges(self) -> None:
        """Stiff Van der Pol oscillator (mu=1000); reference via scipy LSODA."""
        mu = 1000.0

        def rhs(t, y, ydot):
            ydot[0] = y[1]
            ydot[1] = mu * (1.0 - y[0] ** 2) * y[1] - y[0]

        def jac(t, y, fy, J):
            J[0, 1] = 1.0
            J[1, 0] = -2.0 * mu * y[0] * y[1] - 1.0
            J[1, 1] = mu * (1.0 - y[0] ** 2)

        y0 = np.array([2.0, 0.0])
        t_end = 100.0

        integ = CVodeIntegrator(2, rhs, jac=jac, rtol=1e-8, atol=1e-10)
        integ.reinit(0.0, y0)
        _, y = integ.step_to(t_end)

        sol = solve_ivp(
            lambda t, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]],
            (0.0, t_end), y0, method="LSODA", rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(y, sol.y[:, -1], rtol=1e-4, atol=1e-6)


class TestTolerances:
    def test_vector_atol_dictates_local_error(self) -> None:
        """A loose per-component atol on one variable lets it drift; a tight
        atol on the other holds its accuracy. Demonstrates per-component
        control vs scalar atol.
        """
        # Two-state decay with different rates: ydot[0] = -y[0]; ydot[1] = -10 y[1]
        def rhs(t, y, ydot):
            ydot[0] = -y[0]
            ydot[1] = -10.0 * y[1]

        def jac(t, y, fy, J):
            J[0, 0] = -1.0
            J[1, 1] = -10.0

        y0 = np.array([1.0, 1.0])
        t_end = 5.0

        # Tight atol everywhere
        integ_tight = CVodeIntegrator(2, rhs, jac=jac, rtol=1e-10, atol=1e-14)
        integ_tight.reinit(0.0, y0)
        _, y_tight = integ_tight.step_to(t_end)

        # Per-component atol: loose on [0], tight on [1]
        integ_mix = CVodeIntegrator(
            2, rhs, jac=jac, rtol=1e-10, atol=np.array([1e-2, 1e-14]),
        )
        integ_mix.reinit(0.0, y0)
        _, y_mix = integ_mix.step_to(t_end)

        analytical = np.array([np.exp(-t_end), np.exp(-10.0 * t_end)])
        # Both component-0 results land in the rtol regime (y ~ 7e-3 >> atol).
        # Bound is set above the integrator's actual accuracy at the default
        # MaxOrd; we're testing per-component atol behavior, not absolute precision.
        assert abs(y_tight[0] - analytical[0]) < 1e-7 * abs(analytical[0])
        assert abs(y_mix[0]   - analytical[0]) < 1e-2 + 1e-7 * abs(analytical[0])
        # Component-1 sits below atol; the absolute error budget is set by atol.
        # |y_mix[1] - analytical[1]| must be within ~ atol = 1e-14
        assert abs(y_mix[1] - analytical[1]) < 1e-13

    def test_atol_length_mismatch_raises(self) -> None:
        def rhs(t, y, ydot):
            ydot[:] = -y

        with pytest.raises(ValueError, match="atol vector length"):
            CVodeIntegrator(3, rhs, atol=np.array([1e-12, 1e-12]))


class TestDenseOutput:
    def test_get_dky_matches_step_to_endpoint(self) -> None:
        k = 1.0

        def rhs(t, y, ydot):
            ydot[:] = -k * y

        integ = CVodeIntegrator(1, rhs, rtol=1e-10, atol=1e-13)
        integ.reinit(0.0, np.array([1.0]))
        t_end = 1.0
        t, y_step = integ.step_to(t_end)
        # get_dky at the integration endpoint should reproduce step_to's result
        y_dky = integ.get_dky(t_end, k=0)
        np.testing.assert_allclose(y_dky, y_step, rtol=1e-12)

    def test_get_dky_first_derivative_equals_rhs(self) -> None:
        k = 0.5

        def rhs(t, y, ydot):
            ydot[:] = -k * y

        integ = CVodeIntegrator(1, rhs, rtol=1e-10, atol=1e-13)
        integ.reinit(0.0, np.array([1.0]))
        t_end = 1.0
        _, y = integ.step_to(t_end)
        ydot_interp = integ.get_dky(t_end, k=1)
        np.testing.assert_allclose(ydot_interp, -k * y, rtol=1e-6)


class TestReinit:
    def test_reinit_from_different_initial_conditions(self) -> None:
        k = 2.0

        def rhs(t, y, ydot):
            ydot[:] = -k * y

        integ = CVodeIntegrator(1, rhs, rtol=1e-10, atol=1e-13)
        integ.reinit(0.0, np.array([1.0]))
        _, y1 = integ.step_to(0.5)
        np.testing.assert_allclose(y1, [np.exp(-k * 0.5)], rtol=1e-8)

        # Re-init with a different y0 and re-solve
        integ.reinit(0.0, np.array([5.0]))
        _, y2 = integ.step_to(0.5)
        np.testing.assert_allclose(y2, [5.0 * np.exp(-k * 0.5)], rtol=1e-8)


class TestCallbackExceptionSafety:
    def test_rhs_exception_is_reraised_from_step(self) -> None:
        """A Python exception in the RHS aborts integration and the original
        exception (not a downstream SundialsError) is re-raised from
        ``step_to``. The interpreter does not crash."""
        call_count = [0]

        def bad_rhs(t, y, ydot):
            call_count[0] += 1
            if call_count[0] > 1:
                raise ValueError("intentional test failure")
            ydot[:] = -y

        integ = CVodeIntegrator(1, bad_rhs, rtol=1e-6, atol=1e-9)
        integ.reinit(0.0, np.array([1.0]))

        with pytest.raises(ValueError, match="intentional test failure"):
            integ.step_to(10.0)
