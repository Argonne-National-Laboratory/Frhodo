"""ctypes binding to Cantera's bundled SUNDIALS CVODES library.

Cantera ships SUNDIALS with every install but the public C symbols
land in different places per platform:

* Linux: ``_cantera*.so`` statically links SUNDIALS and re-exports
  the public C symbols. One library, all symbols.
* Windows: ``_cantera*.pyd`` does **not** export SUNDIALS; instead
  ``cantera.libs/sundials_*.dll`` (nvecserial, core, cvodes) carries
  them as a set of separate shared libraries.
* macOS (.dylib): treated like Linux until verified otherwise.

This module loads the right set per platform and exposes a uniform
view via :class:`_CompositeLib`. Callers see one lib-like object;
symbol lookup probes each underlying library in turn.

The dense linear solver also differs: Linux exposes
``SUNLinSol_LapackDense``; Windows exposes ``SUNLinSol_Dense``. Either
satisfies our needs — the wrapper picks whichever is available.

RAII wrappers exposed by this module:

* :class:`SundialsContext` — SUNContext lifetime owner.
* :class:`NVector` — serial real-valued vector with a numpy view.
* :class:`DenseMatrix` — column-major dense matrix with a numpy view.
* :class:`DenseLinearSolver` — dense LU linear solver.
* :class:`CVodeIntegrator` — CVODE BDF forward integrator with
  vector tolerances, optional user-supplied dense Jacobian, and
  dense interpolated output via :meth:`CVodeIntegrator.get_dky`.
* :class:`AdjointProblem` — CVODES adjoint sensitivity layered over
  a forward integrator.
* :class:`ForwardSensProblem` — CVODES forward sensitivity layered
  over a forward integrator.

All symbols in :data:`_REQUIRED_SYMBOLS` are bound at module load.
Missing symbols raise :class:`SundialsBindingError` immediately —
``development/verify_cantera_sundials.py`` runs the same check.

See ``doc/sundials_api_reference.md`` for the C-level signatures
and the callback-lifetime hazard (``CFUNCTYPE`` instances must be
kept alive while SUNDIALS holds pointers to them).
"""
from __future__ import annotations

import ctypes
import os
import pathlib
import sys
from typing import Callable

import numpy as np


c_sunrealtype     = ctypes.c_double
c_sunindextype    = ctypes.c_int64
c_sunbooleantype  = ctypes.c_int
c_SUNContext      = ctypes.c_void_p
c_N_Vector        = ctypes.c_void_p
c_SUNMatrix       = ctypes.c_void_p
c_SUNLinearSolver = ctypes.c_void_p


CV_ADAMS = 1
CV_BDF   = 2

CV_NORMAL   = 1
CV_ONE_STEP = 2

CV_HERMITE    = 1
CV_POLYNOMIAL = 2

CV_SIMULTANEOUS = 1
CV_STAGGERED    = 2
CV_STAGGERED1   = 3

CV_SUCCESS      = 0
CV_TSTOP_RETURN = 1
CV_ROOT_RETURN  = 2
CV_WARNING      = 99


_FLAG_NAMES = {
      0: "CV_SUCCESS",            1: "CV_TSTOP_RETURN",
      2: "CV_ROOT_RETURN",       99: "CV_WARNING",
     -1: "CV_TOO_MUCH_WORK",     -2: "CV_TOO_MUCH_ACC",
     -3: "CV_ERR_FAILURE",       -4: "CV_CONV_FAILURE",
     -5: "CV_LINIT_FAIL",        -6: "CV_LSETUP_FAIL",
     -7: "CV_LSOLVE_FAIL",       -8: "CV_RHSFUNC_FAIL",
     -9: "CV_FIRST_RHSFUNC_ERR", -10: "CV_REPTD_RHSFUNC_ERR",
    -11: "CV_UNREC_RHSFUNC_ERR", -12: "CV_RTFUNC_FAIL",
    -13: "CV_NLS_INIT_FAIL",     -14: "CV_NLS_SETUP_FAIL",
    -15: "CV_CONSTR_FAIL",       -16: "CV_NLS_FAIL",
    -20: "CV_MEM_FAIL",          -21: "CV_MEM_NULL",
    -22: "CV_ILL_INPUT",         -23: "CV_NO_MALLOC",
    -24: "CV_BAD_K",             -25: "CV_BAD_T",
    -26: "CV_BAD_DKY",           -27: "CV_TOO_CLOSE",
}


class SundialsBindingError(RuntimeError):
    """Cantera's compiled extension is missing required SUNDIALS exports."""


class SundialsError(RuntimeError):
    """A SUNDIALS function returned a negative error flag.

    Attributes:
        flag: The negative integer SUNDIALS returned.
        fn_name: Name of the SUNDIALS function that produced ``flag``.
    """

    def __init__(self, flag: int, fn_name: str) -> None:
        self.flag = int(flag)
        self.fn_name = fn_name
        name = _FLAG_NAMES.get(self.flag, f"unknown_flag={self.flag}")
        super().__init__(f"{fn_name} failed with flag {self.flag} ({name})")


def _check(flag: int, fn_name: str) -> int:
    """Raise :class:`SundialsError` on negative ``flag``; pass through otherwise."""
    if flag < 0:
        raise SundialsError(flag, fn_name)

    return flag


CVRhsFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)

CVLsJacFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector,
    c_SUNMatrix, ctypes.c_void_p,
    c_N_Vector, c_N_Vector, c_N_Vector,
)

CVRhsFnB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)

CVLsJacFnB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, c_N_Vector,
    c_SUNMatrix, ctypes.c_void_p,
    c_N_Vector, c_N_Vector, c_N_Vector,
)

CVQuadRhsFnB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)

CVSensRhsFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int, c_sunrealtype,
    c_N_Vector, c_N_Vector,
    ctypes.POINTER(c_N_Vector), ctypes.POINTER(c_N_Vector),
    ctypes.c_void_p,
    c_N_Vector, c_N_Vector,
)


_REQUIRED_SYMBOLS: tuple[str, ...] = (
    "SUNContext_Create", "SUNContext_Free",
    "N_VNew_Serial", "N_VMake_Serial", "N_VDestroy_Serial",
    "N_VGetArrayPointer", "N_VGetLength", "N_VClone",
    "N_VConst", "N_VScale", "N_VLinearSum",
    "SUNDenseMatrix", "SUNMatDestroy", "SUNMatZero",
    "SUNDenseMatrix_Data", "SUNDenseMatrix_Rows", "SUNDenseMatrix_Columns",
    "SUNDenseMatrix_LData", "SUNDenseMatrix_Column",
    "SUNLinSolFree",
    "CVodeCreate", "CVodeInit", "CVodeReInit", "CVodeFree",
    "CVode", "CVodeSStolerances", "CVodeSVtolerances",
    "CVodeSetUserData", "CVodeSetJacFn", "CVodeSetLinearSolver",
    "CVodeSetMaxStep", "CVodeSetInitStep", "CVodeSetMaxOrd",
    "CVodeSetStabLimDet",
    "CVodeSetMaxNumSteps", "CVodeSetMaxErrTestFails",
    "CVodeSetMaxNonlinIters", "CVodeSetMaxConvFails",
    "CVodeGetDky", "CVodeGetCurrentStep", "CVodeGetNumSteps",
    "CVodeGetNumRhsEvals",
    "CVodeGetNumLinSolvSetups", "CVodeGetNumErrTestFails",
    "CVodeGetNumNonlinSolvIters", "CVodeGetNumNonlinSolvConvFails",
    "CVodeGetLastOrder", "CVodeGetLastStep",
    "CVodeGetNumJacEvals", "CVodeGetNumLinRhsEvals",
    "CVodeGetReturnFlagName",
    "CVodeAdjInit", "CVodeF",
    "CVodeCreateB", "CVodeInitB", "CVodeReInitB",
    "CVodeSStolerancesB", "CVodeSVtolerancesB",
    "CVodeSetLinearSolverB", "CVodeSetJacFnB",
    "CVodeB", "CVodeGetB",
    "CVodeQuadInitB", "CVodeQuadReInitB",
    "CVodeQuadSStolerancesB", "CVodeQuadSVtolerancesB",
    "CVodeGetQuadB",
    "CVodeSensInit", "CVodeSensInit1", "CVodeSensReInit",
    "CVodeSensSStolerances", "CVodeSensSVtolerances",
    "CVodeSensEEtolerances",
    "CVodeGetSens", "CVodeGetSensDky", "CVodeGetSens1",
)

# At least one of these dense linear solvers must be present. The wrapper
# picks whichever is exported on the current platform.
_DENSE_LINEAR_SOLVERS: tuple[str, ...] = (
    "SUNLinSol_LapackDense",  # Linux Cantera
    "SUNLinSol_Dense",         # Windows Cantera
)

# Optional symbols — bound when present, silently skipped when missing.
# Cantera's Linux build (statically linked SUNDIALS) doesn't export the
# backward integrator setters; Cantera's Windows build (dynamic-linked
# DLLs) does. Adjoint code degrades to SUNDIALS defaults on Linux.
_OPTIONAL_SYMBOLS: tuple[str, ...] = (
    "CVodeSetMaxNumStepsB",
)


class _CompositeLib:
    """Single-lib facade over one or more underlying ``ctypes.CDLL`` objects.

    Symbol lookups probe each library in order and cache the resolved
    callable. Used to span the multiple SUNDIALS DLLs on Windows; on
    Linux there's just one library underneath.
    """

    def __init__(self, libs: list[ctypes.CDLL]) -> None:
        self._libs = libs

    def __getattr__(self, name: str) -> object:
        for lib in self._libs:
            if hasattr(lib, name):
                fn = getattr(lib, name)
                self.__dict__[name] = fn

                return fn
        raise AttributeError(
            f"symbol {name!r} not found in any loaded SUNDIALS library"
        )

    def __contains__(self, name: str) -> bool:
        return any(hasattr(lib, name) for lib in self._libs)


_CANTERA_EXT_GLOBS: tuple[str, ...] = (
    "_cantera*.so", "_cantera*.pyd", "_cantera*.dylib",
)

# Names of the bundled SUNDIALS libraries we need when Cantera doesn't
# statically link them into its extension. ``sundials_idas`` is bundled
# too but unused. Loading order is dependency-first.
_SUNDIALS_BUNDLED_STEMS: tuple[str, ...] = (
    "sundials_nvecserial", "sundials_core", "sundials_cvodes",
)

# Where Cantera's wheel-bundlers stash sibling shared libraries.
_BUNDLED_LIB_DIR_CANDIDATES: tuple[str, ...] = (
    # Sibling-of-package directories (Windows auditwheel-style)
    "../cantera.libs",
    # In-package dylibs directories (macOS delocate-style)
    ".dylibs",
    "../cantera.dylibs",
    # In-package generic libs
    "lib", "libs",
)


def _find_cantera_extension() -> pathlib.Path:
    """Locate the single ``_cantera*.{so,pyd,dylib}`` inside the cantera package.

    Raises:
        SundialsBindingError: If zero or multiple extensions are found.
    """
    import cantera

    cantera_pkg = pathlib.Path(cantera.__file__).parent
    candidates = [
        p for pat in _CANTERA_EXT_GLOBS for p in cantera_pkg.glob(pat)
    ]
    if not candidates:
        raise SundialsBindingError(
            f"no _cantera*.{{so,pyd,dylib}} under {cantera_pkg}; "
            f"is Cantera installed?"
        )
    if len(candidates) > 1:
        raise SundialsBindingError(
            f"multiple Cantera extension files under {cantera_pkg}: "
            f"{candidates}"
        )

    return candidates[0]


def _find_sundials_sibling_libs(cantera_ext: pathlib.Path) -> list[pathlib.Path]:
    """Return any sibling SUNDIALS shared libraries shipped with Cantera.

    Wheel-bundlers (auditwheel/delocate) stash dynamically-linked deps
    in a sibling directory; we probe known patterns and return the
    SUNDIALS libraries in dependency order.
    """
    pkg = cantera_ext.parent
    for rel in _BUNDLED_LIB_DIR_CANDIDATES:
        d = (pkg / rel).resolve()
        if not d.is_dir():
            continue
        results: list[pathlib.Path] = []
        for stem in _SUNDIALS_BUNDLED_STEMS:
            # Try ``stem`` (Windows: sundials_cvodes-HASH.dll) and
            # ``libstem`` (macOS/Linux: libsundials_cvodes.7.4.0.dylib).
            matches: list[pathlib.Path] = []
            for prefix in (stem, f"lib{stem}"):
                for ext in (".so", ".dylib", ".dll"):
                    matches.extend(
                        p for p in d.glob(f"{prefix}*{ext}*") if p.is_file()
                    )
            if matches:
                results.append(sorted(matches)[0])
        if results:

            return results

    return []


def _load_sundials() -> _CompositeLib:
    ext_path = _find_cantera_extension()
    loaded: list[ctypes.CDLL] = []

    sibling_libs = _find_sundials_sibling_libs(ext_path)

    # Windows needs the DLL search path set before any loads with deps
    if sibling_libs and sys.platform == "win32":
        os.add_dll_directory(str(sibling_libs[0].parent))

    # Load sibling libs first (in dep order) so the Cantera extension
    # can resolve its imports against already-loaded SUNDIALS symbols.
    mode = 0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL
    for p in sibling_libs:
        loaded.append(ctypes.CDLL(str(p), mode=mode))
    loaded.append(ctypes.CDLL(str(ext_path), mode=mode))

    lib = _CompositeLib(loaded)

    missing = [s for s in _REQUIRED_SYMBOLS if s not in lib]
    if missing:
        raise SundialsBindingError(
            f"Cantera install at {ext_path.parent} is missing required "
            f"SUNDIALS symbols: {missing}. "
            f"Sibling libs searched: {sibling_libs!r}. "
            f"Run development/verify_cantera_sundials.py for diagnostics."
        )
    if not any(s in lib for s in _DENSE_LINEAR_SOLVERS):
        raise SundialsBindingError(
            f"none of {_DENSE_LINEAR_SOLVERS} are exported by Cantera at "
            f"{ext_path.parent}; need at least one dense linear solver"
        )

    _bind_signatures(lib)

    return lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.SUNContext_Create.argtypes = [ctypes.c_void_p, ctypes.POINTER(c_SUNContext)]
    lib.SUNContext_Create.restype  = ctypes.c_int
    lib.SUNContext_Free.argtypes   = [ctypes.POINTER(c_SUNContext)]
    lib.SUNContext_Free.restype    = ctypes.c_int

    lib.N_VNew_Serial.argtypes     = [c_sunindextype, c_SUNContext]
    lib.N_VNew_Serial.restype      = c_N_Vector
    lib.N_VMake_Serial.argtypes    = [
        c_sunindextype, ctypes.POINTER(c_sunrealtype), c_SUNContext,
    ]
    lib.N_VMake_Serial.restype     = c_N_Vector
    lib.N_VDestroy_Serial.argtypes = [c_N_Vector]
    lib.N_VDestroy_Serial.restype  = None
    lib.N_VGetArrayPointer.argtypes = [c_N_Vector]
    lib.N_VGetArrayPointer.restype  = ctypes.POINTER(c_sunrealtype)
    lib.N_VGetLength.argtypes      = [c_N_Vector]
    lib.N_VGetLength.restype       = c_sunindextype
    lib.N_VClone.argtypes          = [c_N_Vector]
    lib.N_VClone.restype           = c_N_Vector
    lib.N_VConst.argtypes          = [c_sunrealtype, c_N_Vector]
    lib.N_VConst.restype           = None
    lib.N_VScale.argtypes          = [c_sunrealtype, c_N_Vector, c_N_Vector]
    lib.N_VScale.restype           = None
    lib.N_VLinearSum.argtypes      = [
        c_sunrealtype, c_N_Vector, c_sunrealtype, c_N_Vector, c_N_Vector,
    ]
    lib.N_VLinearSum.restype       = None

    lib.SUNDenseMatrix.argtypes         = [c_sunindextype, c_sunindextype, c_SUNContext]
    lib.SUNDenseMatrix.restype          = c_SUNMatrix
    lib.SUNDenseMatrix_Data.argtypes    = [c_SUNMatrix]
    lib.SUNDenseMatrix_Data.restype     = ctypes.POINTER(c_sunrealtype)
    lib.SUNDenseMatrix_Rows.argtypes    = [c_SUNMatrix]
    lib.SUNDenseMatrix_Rows.restype     = c_sunindextype
    lib.SUNDenseMatrix_Columns.argtypes = [c_SUNMatrix]
    lib.SUNDenseMatrix_Columns.restype  = c_sunindextype
    lib.SUNDenseMatrix_LData.argtypes   = [c_SUNMatrix]
    lib.SUNDenseMatrix_LData.restype    = c_sunindextype
    lib.SUNDenseMatrix_Column.argtypes  = [c_SUNMatrix, c_sunindextype]
    lib.SUNDenseMatrix_Column.restype   = ctypes.POINTER(c_sunrealtype)
    lib.SUNMatDestroy.argtypes          = [c_SUNMatrix]
    lib.SUNMatDestroy.restype           = None
    lib.SUNMatZero.argtypes             = [c_SUNMatrix]
    lib.SUNMatZero.restype              = ctypes.c_int

    for solver_name in _DENSE_LINEAR_SOLVERS:
        if solver_name in lib:
            fn = getattr(lib, solver_name)
            fn.argtypes = [c_N_Vector, c_SUNMatrix, c_SUNContext]
            fn.restype  = c_SUNLinearSolver
    lib.SUNLinSolFree.argtypes         = [c_SUNLinearSolver]
    lib.SUNLinSolFree.restype          = ctypes.c_int

    lib.CVodeCreate.argtypes = [ctypes.c_int, c_SUNContext]
    lib.CVodeCreate.restype  = ctypes.c_void_p
    lib.CVodeInit.argtypes   = [ctypes.c_void_p, CVRhsFn, c_sunrealtype, c_N_Vector]
    lib.CVodeInit.restype    = ctypes.c_int
    lib.CVodeReInit.argtypes = [ctypes.c_void_p, c_sunrealtype, c_N_Vector]
    lib.CVodeReInit.restype  = ctypes.c_int
    lib.CVodeFree.argtypes   = [ctypes.POINTER(ctypes.c_void_p)]
    lib.CVodeFree.restype    = None

    lib.CVodeSStolerances.argtypes = [ctypes.c_void_p, c_sunrealtype, c_sunrealtype]
    lib.CVodeSStolerances.restype  = ctypes.c_int
    lib.CVodeSVtolerances.argtypes = [ctypes.c_void_p, c_sunrealtype, c_N_Vector]
    lib.CVodeSVtolerances.restype  = ctypes.c_int

    lib.CVodeSetUserData.argtypes  = [ctypes.c_void_p, ctypes.c_void_p]
    lib.CVodeSetUserData.restype   = ctypes.c_int
    lib.CVodeSetJacFn.argtypes     = [ctypes.c_void_p, CVLsJacFn]
    lib.CVodeSetJacFn.restype      = ctypes.c_int
    lib.CVodeSetLinearSolver.argtypes = [ctypes.c_void_p, c_SUNLinearSolver, c_SUNMatrix]
    lib.CVodeSetLinearSolver.restype  = ctypes.c_int
    lib.CVodeSetMaxStep.argtypes      = [ctypes.c_void_p, c_sunrealtype]
    lib.CVodeSetMaxStep.restype       = ctypes.c_int
    lib.CVodeSetInitStep.argtypes     = [ctypes.c_void_p, c_sunrealtype]
    lib.CVodeSetInitStep.restype      = ctypes.c_int
    lib.CVodeSetMaxOrd.argtypes       = [ctypes.c_void_p, ctypes.c_int]
    lib.CVodeSetMaxOrd.restype        = ctypes.c_int
    lib.CVodeSetStabLimDet.argtypes   = [ctypes.c_void_p, ctypes.c_int]
    lib.CVodeSetStabLimDet.restype    = ctypes.c_int
    lib.CVodeSetMaxNumSteps.argtypes  = [ctypes.c_void_p, ctypes.c_long]
    lib.CVodeSetMaxNumSteps.restype   = ctypes.c_int
    lib.CVodeSetMaxErrTestFails.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.CVodeSetMaxErrTestFails.restype  = ctypes.c_int
    lib.CVodeSetMaxNonlinIters.argtypes  = [ctypes.c_void_p, ctypes.c_int]
    lib.CVodeSetMaxNonlinIters.restype   = ctypes.c_int
    lib.CVodeSetMaxConvFails.argtypes    = [ctypes.c_void_p, ctypes.c_int]
    lib.CVodeSetMaxConvFails.restype     = ctypes.c_int

    lib.CVode.argtypes = [
        ctypes.c_void_p, c_sunrealtype, c_N_Vector,
        ctypes.POINTER(c_sunrealtype), ctypes.c_int,
    ]
    lib.CVode.restype  = ctypes.c_int
    lib.CVodeGetDky.argtypes = [ctypes.c_void_p, c_sunrealtype, ctypes.c_int, c_N_Vector]
    lib.CVodeGetDky.restype  = ctypes.c_int
    lib.CVodeGetCurrentStep.argtypes = [ctypes.c_void_p, ctypes.POINTER(c_sunrealtype)]
    lib.CVodeGetCurrentStep.restype  = ctypes.c_int
    lib.CVodeGetNumSteps.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
    lib.CVodeGetNumSteps.restype  = ctypes.c_int
    for _name in (
        "CVodeGetNumRhsEvals",
        "CVodeGetNumLinSolvSetups",
        "CVodeGetNumErrTestFails",
        "CVodeGetNumNonlinSolvIters",
        "CVodeGetNumNonlinSolvConvFails",
        "CVodeGetNumJacEvals",
        "CVodeGetNumLinRhsEvals",
    ):
        fn = getattr(lib, _name)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        fn.restype  = ctypes.c_int
    lib.CVodeGetLastOrder.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.CVodeGetLastOrder.restype  = ctypes.c_int
    lib.CVodeGetLastStep.argtypes  = [ctypes.c_void_p, ctypes.POINTER(c_sunrealtype)]
    lib.CVodeGetLastStep.restype   = ctypes.c_int
    lib.CVodeGetReturnFlagName.argtypes = [ctypes.c_long]
    lib.CVodeGetReturnFlagName.restype  = ctypes.c_char_p

    lib.CVodeAdjInit.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int]
    lib.CVodeAdjInit.restype  = ctypes.c_int
    lib.CVodeF.argtypes = [
        ctypes.c_void_p, c_sunrealtype, c_N_Vector,
        ctypes.POINTER(c_sunrealtype), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.CVodeF.restype  = ctypes.c_int
    lib.CVodeCreateB.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.CVodeCreateB.restype  = ctypes.c_int
    lib.CVodeInitB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, CVRhsFnB, c_sunrealtype, c_N_Vector,
    ]
    lib.CVodeInitB.restype  = ctypes.c_int
    lib.CVodeReInitB.argtypes = [ctypes.c_void_p, ctypes.c_int, c_sunrealtype, c_N_Vector]
    lib.CVodeReInitB.restype  = ctypes.c_int
    lib.CVodeSStolerancesB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, c_sunrealtype, c_sunrealtype,
    ]
    lib.CVodeSStolerancesB.restype  = ctypes.c_int
    lib.CVodeSVtolerancesB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, c_sunrealtype, c_N_Vector,
    ]
    lib.CVodeSVtolerancesB.restype  = ctypes.c_int
    lib.CVodeSetLinearSolverB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, c_SUNLinearSolver, c_SUNMatrix,
    ]
    lib.CVodeSetLinearSolverB.restype  = ctypes.c_int
    lib.CVodeSetJacFnB.argtypes = [ctypes.c_void_p, ctypes.c_int, CVLsJacFnB]
    lib.CVodeSetJacFnB.restype  = ctypes.c_int

    if "CVodeSetMaxNumStepsB" in lib:
        lib.CVodeSetMaxNumStepsB.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        lib.CVodeSetMaxNumStepsB.restype  = ctypes.c_int
    lib.CVodeB.argtypes = [ctypes.c_void_p, c_sunrealtype, ctypes.c_int]
    lib.CVodeB.restype  = ctypes.c_int
    lib.CVodeGetB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(c_sunrealtype), c_N_Vector,
    ]
    lib.CVodeGetB.restype  = ctypes.c_int

    lib.CVodeQuadInitB.argtypes = [ctypes.c_void_p, ctypes.c_int, CVQuadRhsFnB, c_N_Vector]
    lib.CVodeQuadInitB.restype  = ctypes.c_int
    lib.CVodeQuadReInitB.argtypes = [ctypes.c_void_p, ctypes.c_int, c_N_Vector]
    lib.CVodeQuadReInitB.restype  = ctypes.c_int
    lib.CVodeQuadSStolerancesB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, c_sunrealtype, c_sunrealtype,
    ]
    lib.CVodeQuadSStolerancesB.restype  = ctypes.c_int
    lib.CVodeQuadSVtolerancesB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, c_sunrealtype, c_N_Vector,
    ]
    lib.CVodeQuadSVtolerancesB.restype  = ctypes.c_int
    lib.CVodeGetQuadB.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(c_sunrealtype), c_N_Vector,
    ]
    lib.CVodeGetQuadB.restype  = ctypes.c_int

    lib.CVodeSensInit.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        CVSensRhsFn, ctypes.POINTER(c_N_Vector),
    ]
    lib.CVodeSensInit.restype  = ctypes.c_int
    lib.CVodeSensInit1.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(c_N_Vector),
    ]
    lib.CVodeSensInit1.restype  = ctypes.c_int
    lib.CVodeSensReInit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(c_N_Vector)]
    lib.CVodeSensReInit.restype  = ctypes.c_int
    lib.CVodeSensSStolerances.argtypes = [
        ctypes.c_void_p, c_sunrealtype, ctypes.POINTER(c_sunrealtype),
    ]
    lib.CVodeSensSStolerances.restype  = ctypes.c_int
    lib.CVodeSensSVtolerances.argtypes = [
        ctypes.c_void_p, c_sunrealtype, ctypes.POINTER(c_N_Vector),
    ]
    lib.CVodeSensSVtolerances.restype  = ctypes.c_int
    lib.CVodeSensEEtolerances.argtypes = [ctypes.c_void_p]
    lib.CVodeSensEEtolerances.restype  = ctypes.c_int
    lib.CVodeGetSens.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(c_sunrealtype), ctypes.POINTER(c_N_Vector),
    ]
    lib.CVodeGetSens.restype  = ctypes.c_int
    lib.CVodeGetSensDky.argtypes = [
        ctypes.c_void_p, c_sunrealtype, ctypes.c_int, ctypes.POINTER(c_N_Vector),
    ]
    lib.CVodeGetSensDky.restype  = ctypes.c_int
    lib.CVodeGetSens1.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(c_sunrealtype), ctypes.c_int, c_N_Vector,
    ]
    lib.CVodeGetSens1.restype  = ctypes.c_int


_lib: ctypes.CDLL = _load_sundials()


def _nvector_view(nv_ptr: c_N_Vector, length: int) -> np.ndarray:
    """Zero-copy numpy view aliasing the N_Vector's data buffer."""
    raw = _lib.N_VGetArrayPointer(nv_ptr)

    return np.ctypeslib.as_array(raw, shape=(length,))


def _dense_view(matrix_ptr: c_SUNMatrix, rows: int, cols: int) -> np.ndarray:
    """Zero-copy row-major numpy view over the column-major SUNMatrix buffer."""
    raw = _lib.SUNDenseMatrix_Data(matrix_ptr)

    return np.ctypeslib.as_array(raw, shape=(cols, rows)).T


class SundialsContext:
    """SUNContext lifetime owner.

    SUNDIALS 7 requires every object to be created with a context.
    We use one context per :class:`CVodeIntegrator` so that lifetimes
    are tied to the integrator's lifetime cleanly.
    """

    def __init__(self) -> None:
        ctx = c_SUNContext()
        _check(_lib.SUNContext_Create(None, ctypes.byref(ctx)), "SUNContext_Create")
        self._handle = ctx

    @property
    def handle(self) -> c_SUNContext:
        return self._handle

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h is not None and h.value is not None:
            _lib.SUNContext_Free(ctypes.byref(h))
            self._handle = c_SUNContext(None)


class NVector:
    """Serial N_Vector with a zero-copy numpy view over its buffer.

    Holding the view past this object's lifetime is undefined behavior;
    SUNDIALS owns the underlying memory.
    """

    def __init__(self, length: int, ctx: SundialsContext) -> None:
        ptr = _lib.N_VNew_Serial(c_sunindextype(int(length)), ctx.handle)
        if not ptr:
            raise SundialsError(-20, "N_VNew_Serial")
        self._ptr = c_N_Vector(ptr)
        self._length = int(length)
        self._view = _nvector_view(self._ptr, self._length)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, ctx: SundialsContext) -> "NVector":
        v = cls(arr.size, ctx)
        v._view[:] = arr.reshape(-1)

        return v

    @property
    def handle(self) -> c_N_Vector:
        return self._ptr

    @property
    def length(self) -> int:
        return self._length

    def view(self) -> np.ndarray:
        return self._view

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr is not None and ptr.value is not None:
            _lib.N_VDestroy_Serial(ptr)
            self._ptr = c_N_Vector(None)


class DenseMatrix:
    """Dense ``SUNMatrix`` wrapper.

    ``view()`` returns a row-major numpy view of the column-major
    buffer; writes through the view are visible to SUNDIALS.
    """

    def __init__(self, rows: int, cols: int, ctx: SundialsContext) -> None:
        ptr = _lib.SUNDenseMatrix(
            c_sunindextype(int(rows)), c_sunindextype(int(cols)), ctx.handle,
        )
        if not ptr:
            raise SundialsError(-20, "SUNDenseMatrix")
        self._ptr = c_SUNMatrix(ptr)
        self._rows = int(rows)
        self._cols = int(cols)
        self._view = _dense_view(self._ptr, self._rows, self._cols)

    @property
    def handle(self) -> c_SUNMatrix:
        return self._ptr

    @property
    def shape(self) -> tuple[int, int]:
        return (self._rows, self._cols)

    def view(self) -> np.ndarray:
        return self._view

    def zero(self) -> None:
        _check(_lib.SUNMatZero(self._ptr), "SUNMatZero")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr is not None and ptr.value is not None:
            _lib.SUNMatDestroy(ptr)
            self._ptr = c_SUNMatrix(None)


def _resolve_dense_linear_solver():
    for name in _DENSE_LINEAR_SOLVERS:
        if name in _lib:
            return name, getattr(_lib, name)
    raise SundialsBindingError(
        f"none of {_DENSE_LINEAR_SOLVERS} are available; "
        f"cannot construct a dense linear solver"
    )


class DenseLinearSolver:
    """Dense LU linear solver wrapper.

    Picks ``SUNLinSol_LapackDense`` when available (Linux Cantera) and
    falls back to ``SUNLinSol_Dense`` (Windows Cantera). Either is a
    correct direct dense solver; the LAPACK-backed variant is typically
    faster but Cantera's Windows build does not ship it.
    """

    def __init__(
        self, template: NVector, mat: DenseMatrix, ctx: SundialsContext,
    ) -> None:
        name, ctor = _resolve_dense_linear_solver()
        ptr = ctor(template.handle, mat.handle, ctx.handle)
        if not ptr:
            raise SundialsError(-20, name)
        self._ptr = c_SUNLinearSolver(ptr)

    @property
    def handle(self) -> c_SUNLinearSolver:
        return self._ptr

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr is not None and ptr.value is not None:
            _lib.SUNLinSolFree(ptr)
            self._ptr = c_SUNLinearSolver(None)


PyRhsFn = Callable[[float, np.ndarray, np.ndarray], None]
PyJacFn = Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]


class CVodeIntegrator:
    """CVODE BDF integrator with an optional dense user-supplied Jacobian.

    The user RHS is invoked as ``rhs(t, y, ydot)`` with numpy views
    aliasing SUNDIALS' buffers; the user must fill ``ydot`` in place.
    The user Jacobian (if supplied) is invoked as ``jac(t, y, fy, J)``
    where ``J`` is a row-major ``(n_state, n_state)`` view; writes
    through ``J`` populate the dense matrix SUNDIALS uses for the BDF
    Newton iteration.

    Args:
        n_state: ODE system size.
        rhs: ``rhs(t, y, ydot) -> None``. Fills ``ydot`` in place.
        ctx: Existing :class:`SundialsContext` to share, or ``None`` to
            create one owned by this integrator.
        jac: Optional ``jac(t, y, fy, J) -> None``. When ``None``,
            CVODE uses an internal difference-quotient Jacobian.
        rtol: Relative tolerance.
        atol: Absolute tolerance — scalar or per-component vector of
            length ``n_state``.
        max_steps: ``CVodeSetMaxNumSteps`` between user output points.
        max_step: Optional ``CVodeSetMaxStep`` upper bound on the
            internal step size.
        max_order: ``CVodeSetMaxOrd``. Lower orders (~3) damp the
            step-acceptance jitter that high-order BDF can introduce
            on stiff problems with sharp transitions.
        stab_lim_det: ``CVodeSetStabLimDet`` — enable BDF order
            reduction when the stability boundary tightens. Improves
            run-to-run reproducibility on stiff ignition problems.
    """

    def __init__(
        self,
        n_state: int,
        rhs: PyRhsFn,
        *,
        ctx: SundialsContext | None = None,
        jac: PyJacFn | None = None,
        rtol: float = 1e-6,
        atol: float | np.ndarray = 1e-12,
        max_steps: int = 50000,
        max_step: float | None = None,
        max_order: int = 3,
        stab_lim_det: bool = True,
    ) -> None:
        self._owns_ctx = ctx is None
        self._ctx = ctx if ctx is not None else SundialsContext()
        self.n_state = int(n_state)
        self._py_rhs = rhs
        self._py_jac = jac
        self._py_exception: BaseException | None = None

        self._y = NVector(self.n_state, self._ctx)
        self._A = DenseMatrix(self.n_state, self.n_state, self._ctx)
        self._LS = DenseLinearSolver(self._y, self._A, self._ctx)
        self._atol_nv: NVector | None = None
        self._dky_scratch: NVector | None = None

        mem = _lib.CVodeCreate(CV_BDF, self._ctx.handle)
        if not mem:
            raise SundialsError(-20, "CVodeCreate")
        self._mem = ctypes.c_void_p(mem)

        self._rhs_cb = CVRhsFn(self._rhs_trampoline)
        self._jac_cb = CVLsJacFn(self._jac_trampoline) if jac is not None else None

        self._y.view()[:] = 0.0
        _check(
            _lib.CVodeInit(self._mem, self._rhs_cb, c_sunrealtype(0.0), self._y.handle),
            "CVodeInit",
        )

        self.set_tolerances(rtol, atol)

        _check(
            _lib.CVodeSetLinearSolver(self._mem, self._LS.handle, self._A.handle),
            "CVodeSetLinearSolver",
        )
        if self._jac_cb is not None:
            _check(_lib.CVodeSetJacFn(self._mem, self._jac_cb), "CVodeSetJacFn")

        _check(
            _lib.CVodeSetMaxNumSteps(self._mem, ctypes.c_long(int(max_steps))),
            "CVodeSetMaxNumSteps",
        )
        _check(_lib.CVodeSetMaxOrd(self._mem, ctypes.c_int(int(max_order))),
               "CVodeSetMaxOrd")
        _check(
            _lib.CVodeSetStabLimDet(self._mem, ctypes.c_int(1 if stab_lim_det else 0)),
            "CVodeSetStabLimDet",
        )
        if max_step is not None:
            _check(
                _lib.CVodeSetMaxStep(self._mem, c_sunrealtype(float(max_step))),
                "CVodeSetMaxStep",
            )

        self._tret = c_sunrealtype()

    def set_tolerances(self, rtol: float, atol: float | np.ndarray) -> None:
        """Apply scalar-atol or vector-atol tolerances to CVODE.

        Raises:
            ValueError: If ``atol`` is a vector whose length differs
                from ``n_state``.
        """
        atol_arr = np.atleast_1d(np.asarray(atol, dtype=np.float64))
        if atol_arr.size == 1:
            _check(
                _lib.CVodeSStolerances(
                    self._mem, c_sunrealtype(float(rtol)),
                    c_sunrealtype(float(atol_arr[0])),
                ),
                "CVodeSStolerances",
            )
            self._atol_nv = None
        else:
            if atol_arr.size != self.n_state:
                raise ValueError(
                    f"atol vector length {atol_arr.size} != n_state {self.n_state}"
                )
            self._atol_nv = NVector.from_numpy(atol_arr, self._ctx)
            _check(
                _lib.CVodeSVtolerances(
                    self._mem, c_sunrealtype(float(rtol)), self._atol_nv.handle,
                ),
                "CVodeSVtolerances",
            )

    def reinit(self, t0: float, y0: np.ndarray) -> None:
        """Reset the integrator to ``(t0, y0)`` for a fresh solve.

        Raises:
            ValueError: If ``y0.size`` does not equal ``n_state``.
        """
        y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
        if y0.size != self.n_state:
            raise ValueError(f"y0 size {y0.size} != n_state {self.n_state}")
        self._y.view()[:] = y0
        _check(
            _lib.CVodeReInit(self._mem, c_sunrealtype(float(t0)), self._y.handle),
            "CVodeReInit",
        )

    def step_to(self, t_target: float) -> tuple[float, np.ndarray]:
        """Integrate to ``t_target`` in CV_NORMAL mode.

        Returns:
            ``(t_reached, y)`` — SUNDIALS may stop short of
            ``t_target`` on warnings; ``t_reached`` reports the actual
            time integrated to. ``y`` is a copy of the state vector.

        Raises:
            SundialsError: If CVODE returns a negative flag and the
                user RHS did not raise.
            BaseException: Any exception raised inside the user RHS
                callback is captured and re-raised here.
        """
        self._py_exception = None
        try:
            _check(
                _lib.CVode(
                    self._mem, c_sunrealtype(float(t_target)), self._y.handle,
                    ctypes.byref(self._tret), CV_NORMAL,
                ),
                "CVode",
            )
        except SundialsError:
            if self._py_exception is not None:
                raise self._py_exception
            raise

        return float(self._tret.value), self._y.view().copy()

    def get_dky(self, t: float, k: int = 0) -> np.ndarray:
        """Dense interpolated state (``k=0``) or k-th derivative at ``t``.

        Returns:
            Copy of the interpolated vector, shape ``(n_state,)``.
        """
        if self._dky_scratch is None:
            self._dky_scratch = NVector(self.n_state, self._ctx)
        _check(
            _lib.CVodeGetDky(
                self._mem, c_sunrealtype(float(t)),
                ctypes.c_int(int(k)), self._dky_scratch.handle,
            ),
            "CVodeGetDky",
        )

        return self._dky_scratch.view().copy()

    def num_steps(self) -> int:
        n = ctypes.c_long(0)
        _check(_lib.CVodeGetNumSteps(self._mem, ctypes.byref(n)), "CVodeGetNumSteps")

        return int(n.value)

    def current_step(self) -> float:
        h = c_sunrealtype(0.0)
        _check(_lib.CVodeGetCurrentStep(self._mem, ctypes.byref(h)),
               "CVodeGetCurrentStep")

        return float(h.value)

    def stats(self) -> dict:
        """Snapshot of CVODE long-counter and last-step diagnostics.

        Returns:
            Dict with keys:

            * ``num_steps``: total internal steps accepted.
            * ``num_rhs_evals``: user-RHS callback invocations.
            * ``num_jac_evals``: user-Jacobian callback invocations.
            * ``num_lin_solv_setups``: linear-solver setups (LU
              factorizations on the dense path).
            * ``num_err_test_fails``: failed error tests (step rejects).
            * ``num_nonlin_solv_iters``: total Newton iterations.
            * ``num_nonlin_solv_conv_fails``: Newton convergence failures.
            * ``num_lin_rhs_evals``: RHS evals inside a difference-quotient
              Jacobian; should be 0 when a user Jacobian is supplied.
            * ``last_order``: BDF order of the most recent step.
            * ``last_step``: size of the most recent step.
        """
        def _long(name: str) -> int:
            n = ctypes.c_long(0)
            _check(getattr(_lib, name)(self._mem, ctypes.byref(n)), name)
            return int(n.value)

        order = ctypes.c_int(0)
        _check(_lib.CVodeGetLastOrder(self._mem, ctypes.byref(order)),
               "CVodeGetLastOrder")
        last_h = c_sunrealtype(0.0)
        _check(_lib.CVodeGetLastStep(self._mem, ctypes.byref(last_h)),
               "CVodeGetLastStep")

        return {
            "num_steps":                  _long("CVodeGetNumSteps"),
            "num_rhs_evals":              _long("CVodeGetNumRhsEvals"),
            "num_jac_evals":              _long("CVodeGetNumJacEvals"),
            "num_lin_solv_setups":        _long("CVodeGetNumLinSolvSetups"),
            "num_err_test_fails":         _long("CVodeGetNumErrTestFails"),
            "num_nonlin_solv_iters":      _long("CVodeGetNumNonlinSolvIters"),
            "num_nonlin_solv_conv_fails": _long("CVodeGetNumNonlinSolvConvFails"),
            "num_lin_rhs_evals":          _long("CVodeGetNumLinRhsEvals"),
            "last_order":                 int(order.value),
            "last_step":                  float(last_h.value),
        }

    def _rhs_trampoline(self, t, y_ptr, ydot_ptr, _ud):
        try:
            y_view  = _nvector_view(y_ptr, self.n_state)
            yd_view = _nvector_view(ydot_ptr, self.n_state)
            self._py_rhs(float(t), y_view, yd_view)

            return 0
        except BaseException as e:
            self._py_exception = e

            return -1

    def _jac_trampoline(self, t, y_ptr, fy_ptr, J_ptr, _ud, _t1, _t2, _t3):
        try:
            y_view  = _nvector_view(y_ptr, self.n_state)
            fy_view = _nvector_view(fy_ptr, self.n_state)
            J_view  = _dense_view(J_ptr, self.n_state, self.n_state)
            _check(_lib.SUNMatZero(J_ptr), "SUNMatZero")
            self._py_jac(float(t), y_view, fy_view, J_view)

            return 0
        except BaseException as e:
            self._py_exception = e

            return -1

    def __del__(self) -> None:
        mem = getattr(self, "_mem", None)
        if mem is not None and mem.value is not None:
            _lib.CVodeFree(ctypes.byref(mem))
            self._mem = ctypes.c_void_p(None)


PyRhsFnB = Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]
PyJacFnB = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]
PyQuadRhsFnB = Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]


class AdjointProblem:
    """CVODES adjoint sensitivity layered over a :class:`CVodeIntegrator`.

    Lifecycle:

    1. Construction registers ``CVodeAdjInit`` on the parent CVODE
       memory and pre-allocates the backward state, quadrature,
       matrix, and linear solver.
    2. :meth:`run_forward` drives ``CVodeF`` (with checkpoints) to each
       target time in ascending order and returns a mapping ``t → y``.
    3. :meth:`solve_backward` is called once per output time in
       **descending** order. The first call creates the backward
       problem (``CVodeCreateB`` + ``CVodeInitB`` + ``CVodeQuadInitB``);
       subsequent calls reinitialize it.

    Callbacks take numpy views aliasing SUNDIALS buffers:

    * ``rhsB(t, y, yB, yBdot)`` — backward RHS, e.g. ``yBdot = -Jᵀ·yB``.
    * ``jacB(t, y, yB, fyB, JB)`` — backward dense Jacobian; ``JB`` is
      a row-major dense view.
    * ``quad_rhsB(t, y, yB, qBdot)`` — quadrature integrand.

    ``y`` is the forward state at time ``t``, recovered by SUNDIALS via
    cubic Hermite interpolation between checkpoints; ``yB`` is the
    current adjoint state.

    Args:
        integrator: Forward :class:`CVodeIntegrator` whose memory hosts
            the adjoint problem.
        rhsB: Backward RHS callback.
        jacB: Backward dense Jacobian callback.
        quad_rhsB: Quadrature integrand callback.
        n_quadrature: Number of quadrature components, or 0 to skip.
        n_checkpoints: Number of forward checkpoints CVodeF retains
            for re-interpolation during backward solves.
        interp: ``CV_HERMITE`` (default) or ``CV_POLYNOMIAL``.
        rtolB, atolB: Backward-state tolerances (vector ``atolB``
            length must equal ``n_state``).
        rtolQB, atolQB: Quadrature tolerances (vector ``atolQB``
            length must equal ``n_quadrature``).
        max_backward_steps: Forwarded to ``CVodeSetMaxNumStepsB`` when
            the setter is available. Cantera's Linux build does not
            export it; the default 500 then applies and stiff
            backward problems may fail — see module docstring.
    """

    def __init__(
        self,
        integrator: CVodeIntegrator,
        *,
        rhsB: PyRhsFnB,
        jacB: PyJacFnB,
        quad_rhsB: PyQuadRhsFnB,
        n_quadrature: int,
        n_checkpoints: int = 150,
        interp: int = CV_HERMITE,
        rtolB: float = 1e-6,
        atolB: float | np.ndarray = 1e-10,
        rtolQB: float = 1e-6,
        atolQB: float | np.ndarray = 1e-10,
        max_backward_steps: int | None = None,
    ) -> None:
        self._integ = integrator
        self._ctx = integrator._ctx
        self._mem = integrator._mem
        self._max_backward_steps = max_backward_steps
        self._n_state = integrator.n_state
        self._n_quad = int(n_quadrature)
        self._py_rhsB = rhsB
        self._py_jacB = jacB
        self._py_quadB = quad_rhsB

        _check(
            _lib.CVodeAdjInit(
                self._mem, ctypes.c_long(int(n_checkpoints)), ctypes.c_int(int(interp)),
            ),
            "CVodeAdjInit",
        )

        self._yB    = NVector(self._n_state, self._ctx)
        self._qB    = NVector(self._n_quad,  self._ctx) if self._n_quad > 0 else None
        self._matB  = DenseMatrix(self._n_state, self._n_state, self._ctx)
        self._LSB   = DenseLinearSolver(self._yB, self._matB, self._ctx)

        atolB_arr = np.atleast_1d(np.asarray(atolB, dtype=np.float64))
        if atolB_arr.size == 1:
            self._atolB_scalar: float | None = float(atolB_arr[0])
            self._atolB_nv: NVector | None = None
        else:
            if atolB_arr.size != self._n_state:
                raise ValueError(
                    f"atolB length {atolB_arr.size} != n_state {self._n_state}"
                )
            self._atolB_scalar = None
            self._atolB_nv = NVector.from_numpy(atolB_arr, self._ctx)
        self._rtolB = float(rtolB)

        if self._qB is not None:
            atolQB_arr = np.atleast_1d(np.asarray(atolQB, dtype=np.float64))
            if atolQB_arr.size == 1:
                self._atolQB_scalar: float | None = float(atolQB_arr[0])
                self._atolQB_nv: NVector | None = None
            else:
                if atolQB_arr.size != self._n_quad:
                    raise ValueError(
                        f"atolQB length {atolQB_arr.size} != n_quadrature {self._n_quad}"
                    )
                self._atolQB_scalar = None
                self._atolQB_nv = NVector.from_numpy(atolQB_arr, self._ctx)
            self._rtolQB = float(rtolQB)

        self._rhsB_cb  = CVRhsFnB(self._rhsB_trampoline)
        self._jacB_cb  = CVLsJacFnB(self._jacB_trampoline)
        self._quadB_cb = (
            CVQuadRhsFnB(self._quadB_trampoline) if self._qB is not None else None
        )

        self._whichB = ctypes.c_int(-1)
        self._ncheck = ctypes.c_int(0)
        self._tret_b = c_sunrealtype()

    def run_forward(
        self, t_targets: list[float] | np.ndarray,
    ) -> dict[float, np.ndarray]:
        """Drive the forward integration via ``CVodeF`` (with checkpoints).

        Args:
            t_targets: Output times, sorted ascending.

        Returns:
            Mapping ``{t: y(t)}`` keyed by the float-valued targets.
        """
        results: dict[float, np.ndarray] = {}
        for t in t_targets:
            t_f = float(t)
            self._integ._py_exception = None
            try:
                _check(
                    _lib.CVodeF(
                        self._mem, c_sunrealtype(t_f),
                        self._integ._y.handle,
                        ctypes.byref(self._integ._tret),
                        CV_NORMAL, ctypes.byref(self._ncheck),
                    ),
                    "CVodeF",
                )
            except SundialsError:
                if self._integ._py_exception is not None:
                    raise self._integ._py_exception
                raise
            results[t_f] = self._integ._y.view().copy()

        return results

    def solve_backward(
        self, t_m: float, yB0: np.ndarray, qB0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Backward solve from ``t_m`` to ``0`` with inline quadrature.

        On the first call, creates the backward problem
        (``CVodeCreateB`` + ``CVodeInitB`` + ``CVodeQuadInitB``).
        Subsequent calls reinitialize it
        (``CVodeReInitB`` + ``CVodeQuadReInitB``).

        Args:
            t_m: Terminal time for the backward solve.
            yB0: Adjoint terminal condition, shape ``(n_state,)``.
            qB0: Initial quadrature value, shape ``(n_quadrature,)``,
                or ``None`` to start from zero.

        Returns:
            ``(lam_at_0, quadrature)``. ``quadrature`` is ``None`` when
            ``n_quadrature == 0``.

        Raises:
            ValueError: If ``yB0`` or ``qB0`` has the wrong length.
            BaseException: Any exception raised in the user backward
                callbacks is captured and re-raised here.
        """
        yB0 = np.asarray(yB0, dtype=np.float64).reshape(-1)
        if yB0.size != self._n_state:
            raise ValueError(f"yB0 size {yB0.size} != n_state {self._n_state}")
        self._yB.view()[:] = yB0

        if self._qB is not None:
            if qB0 is None:
                self._qB.view()[:] = 0.0
            else:
                q = np.asarray(qB0, dtype=np.float64).reshape(-1)
                if q.size != self._n_quad:
                    raise ValueError(
                        f"qB0 size {q.size} != n_quadrature {self._n_quad}"
                    )
                self._qB.view()[:] = q

        if self._whichB.value < 0:
            self._init_backward(float(t_m))
        else:
            _check(
                _lib.CVodeReInitB(
                    self._mem, self._whichB,
                    c_sunrealtype(float(t_m)), self._yB.handle,
                ),
                "CVodeReInitB",
            )
            if self._qB is not None:
                _check(
                    _lib.CVodeQuadReInitB(self._mem, self._whichB, self._qB.handle),
                    "CVodeQuadReInitB",
                )

        self._integ._py_exception = None
        try:
            _check(
                _lib.CVodeB(self._mem, c_sunrealtype(0.0), CV_NORMAL),
                "CVodeB",
            )
        except SundialsError:
            if self._integ._py_exception is not None:
                raise self._integ._py_exception
            raise

        _check(
            _lib.CVodeGetB(
                self._mem, self._whichB, ctypes.byref(self._tret_b), self._yB.handle,
            ),
            "CVodeGetB",
        )
        lam_at_0 = self._yB.view().copy()

        quadrature = None
        if self._qB is not None:
            _check(
                _lib.CVodeGetQuadB(
                    self._mem, self._whichB,
                    ctypes.byref(self._tret_b), self._qB.handle,
                ),
                "CVodeGetQuadB",
            )
            quadrature = self._qB.view().copy()

        return lam_at_0, quadrature

    def _init_backward(self, t_m: float) -> None:
        _check(
            _lib.CVodeCreateB(self._mem, CV_BDF, ctypes.byref(self._whichB)),
            "CVodeCreateB",
        )
        _check(
            _lib.CVodeInitB(
                self._mem, self._whichB, self._rhsB_cb,
                c_sunrealtype(t_m), self._yB.handle,
            ),
            "CVodeInitB",
        )
        if self._atolB_nv is not None:
            _check(
                _lib.CVodeSVtolerancesB(
                    self._mem, self._whichB,
                    c_sunrealtype(self._rtolB), self._atolB_nv.handle,
                ),
                "CVodeSVtolerancesB",
            )
        else:
            _check(
                _lib.CVodeSStolerancesB(
                    self._mem, self._whichB,
                    c_sunrealtype(self._rtolB), c_sunrealtype(self._atolB_scalar),
                ),
                "CVodeSStolerancesB",
            )
        _check(
            _lib.CVodeSetLinearSolverB(
                self._mem, self._whichB, self._LSB.handle, self._matB.handle,
            ),
            "CVodeSetLinearSolverB",
        )
        _check(
            _lib.CVodeSetJacFnB(self._mem, self._whichB, self._jacB_cb),
            "CVodeSetJacFnB",
        )
        if self._max_backward_steps is not None:
            if "CVodeSetMaxNumStepsB" in _lib:
                _check(
                    _lib.CVodeSetMaxNumStepsB(
                        self._mem, self._whichB,
                        ctypes.c_long(int(self._max_backward_steps)),
                    ),
                    "CVodeSetMaxNumStepsB",
                )
            # On Linux Cantera the setter isn't exported; the cap stays at
            # SUNDIALS' default 500 steps. Adjoint may fail on stiff
            # problems there — use forward_sens instead.
        if self._qB is not None:
            _check(
                _lib.CVodeQuadInitB(
                    self._mem, self._whichB, self._quadB_cb, self._qB.handle,
                ),
                "CVodeQuadInitB",
            )
            if self._atolQB_nv is not None:
                _check(
                    _lib.CVodeQuadSVtolerancesB(
                        self._mem, self._whichB,
                        c_sunrealtype(self._rtolQB), self._atolQB_nv.handle,
                    ),
                    "CVodeQuadSVtolerancesB",
                )
            else:
                _check(
                    _lib.CVodeQuadSStolerancesB(
                        self._mem, self._whichB,
                        c_sunrealtype(self._rtolQB), c_sunrealtype(self._atolQB_scalar),
                    ),
                    "CVodeQuadSStolerancesB",
                )

    def _rhsB_trampoline(self, t, y_ptr, yB_ptr, yBdot_ptr, _ud):
        try:
            y_view     = _nvector_view(y_ptr,     self._n_state)
            yB_view    = _nvector_view(yB_ptr,    self._n_state)
            yBdot_view = _nvector_view(yBdot_ptr, self._n_state)
            self._py_rhsB(float(t), y_view, yB_view, yBdot_view)

            return 0
        except BaseException as e:
            self._integ._py_exception = e

            return -1

    def _jacB_trampoline(self, t, y_ptr, yB_ptr, fyB_ptr, JB_ptr, _ud, _t1, _t2, _t3):
        try:
            y_view   = _nvector_view(y_ptr,   self._n_state)
            yB_view  = _nvector_view(yB_ptr,  self._n_state)
            fyB_view = _nvector_view(fyB_ptr, self._n_state)
            JB_view  = _dense_view(JB_ptr, self._n_state, self._n_state)
            _check(_lib.SUNMatZero(JB_ptr), "SUNMatZero")
            self._py_jacB(float(t), y_view, yB_view, fyB_view, JB_view)

            return 0
        except BaseException as e:
            self._integ._py_exception = e

            return -1

    def _quadB_trampoline(self, t, y_ptr, yB_ptr, qBdot_ptr, _ud):
        try:
            y_view    = _nvector_view(y_ptr,    self._n_state)
            yB_view   = _nvector_view(yB_ptr,   self._n_state)
            qBdot_view = _nvector_view(qBdot_ptr, self._n_quad)
            self._py_quadB(float(t), y_view, yB_view, qBdot_view)

            return 0
        except BaseException as e:
            self._integ._py_exception = e

            return -1


PySensRhsFn = Callable[
    [int, float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]],
    None,
]


class ForwardSensProblem:
    """CVODES forward sensitivity layered over a :class:`CVodeIntegrator`.

    Augments the forward solve with ``N_params`` sensitivity vectors
    ``S_k(t) = ∂y/∂p_k`` that obey ``dS_k/dt = (∂f/∂y)·S_k + ∂f/∂p_k``,
    with ``S_k(t0) = 0``.

    The user provides ``sens_rhs(Ns, t, y, ydot, yS, ySdot)``. ``yS``
    and ``ySdot`` are Python lists of length ``Ns`` of numpy views
    over each sensitivity and its derivative; the callback fills each
    ``ySdot[k]`` in place with ``∂f/∂y · yS[k] + ∂f/∂p_k``.

    Args:
        integrator: Forward :class:`CVodeIntegrator` whose memory hosts
            the sensitivity problem.
        n_params: Number of parameters (e.g. reactions).
        sens_rhs: Sensitivity RHS callback.
        ism: ``CV_SIMULTANEOUS``, ``CV_STAGGERED`` (default), or
            ``CV_STAGGERED1`` — sets how the sens solve is coupled to
            the state solve.
        rtol: Sensitivity relative tolerance.
        atol: Scalar or per-state-component vector absolute tolerance;
            broadcast to every sensitivity column.

    After :meth:`run_forward`, each entry of the returned dict has
    shape ``(n_state, n_params)`` — column ``k`` is ``∂y/∂p_k``.
    """

    def __init__(
        self,
        integrator: CVodeIntegrator,
        *,
        n_params: int,
        sens_rhs: PySensRhsFn,
        ism: int = CV_STAGGERED,
        rtol: float = 1e-6,
        atol: float | np.ndarray = 1e-10,
    ) -> None:
        self._integ = integrator
        self._ctx = integrator._ctx
        self._mem = integrator._mem
        self._n_state = integrator.n_state
        self._n_params = int(n_params)
        self._py_sens_rhs = sens_rhs

        self._yS_storage = [
            NVector(self._n_state, self._ctx) for _ in range(self._n_params)
        ]
        for v in self._yS_storage:
            v.view()[:] = 0.0
        self._yS_array = (c_N_Vector * self._n_params)(
            *(v.handle for v in self._yS_storage)
        )
        self._yS_views = [v.view() for v in self._yS_storage]

        self._sens_cb = CVSensRhsFn(self._sens_trampoline)

        _check(
            _lib.CVodeSensInit(
                self._mem, ctypes.c_int(self._n_params),
                ctypes.c_int(int(ism)), self._sens_cb, self._yS_array,
            ),
            "CVodeSensInit",
        )

        atol_arr = np.atleast_1d(np.asarray(atol, dtype=np.float64))
        if atol_arr.size == 1:
            atol_arr_p = (c_sunrealtype * self._n_params)(
                *([float(atol_arr[0])] * self._n_params),
            )
            _check(
                _lib.CVodeSensSStolerances(
                    self._mem, c_sunrealtype(float(rtol)), atol_arr_p,
                ),
                "CVodeSensSStolerances",
            )
            self._atolS_nvs: list[NVector] | None = None
        else:
            if atol_arr.size != self._n_state:
                raise ValueError(
                    f"atol vector length {atol_arr.size} != n_state {self._n_state}"
                )
            self._atolS_nvs = [
                NVector.from_numpy(atol_arr, self._ctx) for _ in range(self._n_params)
            ]
            atol_array_c = (c_N_Vector * self._n_params)(
                *(v.handle for v in self._atolS_nvs),
            )
            _check(
                _lib.CVodeSensSVtolerances(
                    self._mem, c_sunrealtype(float(rtol)), atol_array_c,
                ),
                "CVodeSensSVtolerances",
            )

    def run_forward(
        self, t_targets: list[float] | np.ndarray,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Drive the augmented forward solve to each target time.

        Args:
            t_targets: Output times, sorted ascending.

        Returns:
            Mapping ``{t: (y, S)}`` where ``y.shape == (n_state,)``
            and ``S.shape == (n_state, n_params)``; column ``k`` of
            ``S`` is ``∂y/∂p_k`` at ``t``.
        """
        results: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        tret_local = c_sunrealtype()
        for t in t_targets:
            t_f = float(t)
            self._integ._py_exception = None
            try:
                _check(
                    _lib.CVode(
                        self._mem, c_sunrealtype(t_f),
                        self._integ._y.handle,
                        ctypes.byref(self._integ._tret),
                        CV_NORMAL,
                    ),
                    "CVode",
                )
            except SundialsError:
                if self._integ._py_exception is not None:
                    raise self._integ._py_exception
                raise

            _check(
                _lib.CVodeGetSens(
                    self._mem, ctypes.byref(tret_local), self._yS_array,
                ),
                "CVodeGetSens",
            )
            y_copy = self._integ._y.view().copy()
            S = np.column_stack([v.copy() for v in self._yS_views])
            results[t_f] = (y_copy, S)

        return results

    def _sens_trampoline(self, Ns, t, y_ptr, ydot_ptr, yS_ptr, ySdot_ptr, _ud, _t1, _t2):
        try:
            y_view    = _nvector_view(y_ptr,    self._n_state)
            ydot_view = _nvector_view(ydot_ptr, self._n_state)
            yS_list    = [_nvector_view(yS_ptr[k],    self._n_state) for k in range(int(Ns))]
            ySdot_list = [_nvector_view(ySdot_ptr[k], self._n_state) for k in range(int(Ns))]
            self._py_sens_rhs(int(Ns), float(t), y_view, ydot_view, yS_list, ySdot_list)

            return 0
        except BaseException as e:
            self._integ._py_exception = e

            return -1
