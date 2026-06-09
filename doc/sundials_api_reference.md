# SUNDIALS ctypes binding reference

ctypes bindings for the SUNDIALS CVODES C API as exposed by Cantera's
compiled extension. This document is the **authoritative reference for
function signatures, types, and callback patterns** used by
`frhodo/simulation/numerics/sundials.py`.

Sourced from the [SUNDIALS 7.4.0 documentation](https://sundials.readthedocs.io/en/v7.4.0/),
cross-checked against the symbols actually exported by Cantera 3.2.0
(verified via `development/verify_cantera_sundials.py`).

---

## 1. Pinned versions

| Component | Version | Where verified |
|---|---|---|
| Cantera | 3.2.0 | `pyproject.toml`; install command output |
| SUNDIALS (bundled by Cantera) | 7.4.0 | `strings _cantera*.so | grep -E "^[0-9]+\.[0-9]+\.[0-9]+"` returns `7.4.0` |
| API stability window | `cantera>=3.2,<4.0` | Plan §13, decision 9 |

If a future Cantera release fails the symbol checks in
`verify_cantera_sundials.py`, the binding raises at module load with a
clear error.

### 1.1 Platform packaging differences

Cantera ships SUNDIALS the same way on every platform but the public
C symbols land in different files:

| Platform | Cantera extension | SUNDIALS public symbols | Loaded via |
|---|---|---|---|
| Linux | `_cantera*.so` | Statically linked; re-exported from the extension | The single `.so` |
| Windows | `_cantera*.pyd` | **Not** in the extension; dynamically linked to `cantera.libs/sundials_*.dll` | Multiple DLLs loaded in dep order (nvecserial → core → cvodes) plus the .pyd |
| macOS | `_cantera*.cpython-*-darwin.so` | Not in the extension; dynamically linked to `cantera/.dylibs/libsundials_*.dylib` | Same composite-lib pattern as Windows (verified via wheel inspection; runtime not yet exercised) |

The binding's `_load_sundials()` detects sibling SUNDIALS libraries
at module load and combines them into a single virtual library via
`_CompositeLib`. Symbol lookup transparently spans the underlying
libraries.

### 1.2 Platform-specific symbol differences

| Symbol | Linux | Windows | macOS |
|---|---|---|---|
| `SUNLinSol_LapackDense` | ✅ exported | ❌ missing | ✅ via `libsundials_sunlinsollapackdense.dylib` |
| `SUNLinSol_Dense` | ❌ missing | ✅ exported | ✅ via `libsundials_sunmatrixdense.dylib` |
| `CVodeSetMaxNumStepsB` | ❌ missing | ✅ exported | ✅ exported (in `libsundials_cvodes.dylib`) |
| `CVodeSetMaxStepB` | ❌ missing | ❌ missing | unknown |

The `DenseLinearSolver` wrapper picks whichever dense solver is
present. Because `CVodeSetMaxNumStepsB` is unavailable on Linux, the
adjoint backward solve there is capped at SUNDIALS' default 500 steps
per output time and may fail on stiff backward problems (e.g., GRI
3.0 ignition). On Windows and macOS the same problem succeeds because
the setter can raise the cap — `compute_adjoint_sensitivity` passes
``max_backward_steps=50_000``. ``forward_sens`` is unaffected by this
limit on any platform.

---

## 2. Type definitions

In SUNDIALS 7.x the canonical scalar and index types are:

```c
typedef double                sunrealtype;        /* default build */
typedef int64_t               sunindextype;       /* 64-bit system default */
typedef int                   sunbooleantype;
```

ctypes mapping:

```python
import ctypes

c_sunrealtype     = ctypes.c_double
c_sunindextype    = ctypes.c_int64
c_sunbooleantype  = ctypes.c_int
c_SUNContext      = ctypes.c_void_p   # opaque handle
c_N_Vector        = ctypes.c_void_p
c_SUNMatrix       = ctypes.c_void_p
c_SUNLinearSolver = ctypes.c_void_p
c_void_mem        = ctypes.c_void_p   # cvode_mem and similar handles
```

The void-pointer aliases are the right abstraction — SUNDIALS treats
these as opaque from Python's perspective. Use ctypes' opaque pointer
type via `ctypes.c_void_p`; the wrapper class keeps the lifetime alive.

---

## 3. Constants

```python
# CVode iteration method (CVodeCreate)
CV_ADAMS = 1
CV_BDF   = 2

# CVode itask (CVode, CVodeF, CVodeB)
CV_NORMAL   = 1
CV_ONE_STEP = 2

# CVodeAdjInit interpolation
CV_HERMITE    = 1
CV_POLYNOMIAL = 2

# CVodeSensInit ism
CV_SIMULTANEOUS = 1
CV_STAGGERED    = 2
CV_STAGGERED1   = 3

# Return flags (selected; full list in sundials_types.h)
CV_SUCCESS         =   0
CV_TSTOP_RETURN    =   1
CV_ROOT_RETURN     =   2
CV_WARNING         =  99
CV_TOO_MUCH_WORK   =  -1
CV_TOO_MUCH_ACC    =  -2
CV_ERR_FAILURE     =  -3
CV_CONV_FAILURE    =  -4
CV_LINIT_FAIL      =  -5
CV_LSETUP_FAIL     =  -6
CV_LSOLVE_FAIL     =  -7
CV_RHSFUNC_FAIL    =  -8
CV_FIRST_RHSFUNC_ERR    =  -9
CV_REPTD_RHSFUNC_ERR    = -10
CV_UNREC_RHSFUNC_ERR    = -11
CV_RTFUNC_FAIL          = -12
CV_NLS_INIT_FAIL        = -13
CV_NLS_SETUP_FAIL       = -14
CV_CONSTR_FAIL          = -15
CV_NLS_FAIL             = -16
CV_MEM_FAIL             = -20
CV_MEM_NULL             = -21
CV_ILL_INPUT            = -22
CV_NO_MALLOC            = -23
CV_BAD_K                = -24
CV_BAD_T                = -25
CV_BAD_DKY              = -26
CV_TOO_CLOSE            = -27
```

---

## 4. Error handling pattern

SUNDIALS returns `int` flags. Wrap every call in a `_check()` helper
that converts negative flags to Python exceptions:

```python
class SundialsError(RuntimeError):
    def __init__(self, flag, fn_name):
        self.flag = flag
        super().__init__(f"{fn_name} failed with flag {flag} ({_flag_name(flag)})")

_FLAG_NAMES = {
    -1: "CV_TOO_MUCH_WORK",   -2: "CV_TOO_MUCH_ACC",
    -3: "CV_ERR_FAILURE",     -4: "CV_CONV_FAILURE",
    # ... (full mapping above)
}

def _flag_name(f):
    return _FLAG_NAMES.get(f, f"unknown_flag={f}")

def _check(flag, fn_name):
    if flag < 0:
        raise SundialsError(flag, fn_name)
    return flag
```

Positive return flags are non-errors (CV_TSTOP_RETURN, CV_ROOT_RETURN)
and should be passed through.

---

## 5. Callback function types

ctypes `CFUNCTYPE` declarations for each callback SUNDIALS expects.
**Critically: the CFUNCTYPE instance must be kept alive** as long as
SUNDIALS holds a pointer to it — see §10 below.

### Forward RHS

```c
typedef int (*CVRhsFn)(sunrealtype t, N_Vector y, N_Vector ydot,
                       void *user_data);
```

```python
CVRhsFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)
```

### Forward Jacobian (dense)

```c
typedef int (*CVLsJacFn)(sunrealtype t, N_Vector y, N_Vector fy,
                         SUNMatrix Jac, void *user_data,
                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
```

```python
CVLsJacFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector,
    c_SUNMatrix, ctypes.c_void_p,
    c_N_Vector, c_N_Vector, c_N_Vector,
)
```

### Backward RHS

```c
typedef int (*CVRhsFnB)(sunrealtype t, N_Vector y, N_Vector yB,
                        N_Vector yBdot, void *user_dataB);
```

```python
CVRhsFnB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)
```

### Backward Jacobian

```c
typedef int (*CVLsJacFnB)(sunrealtype t, N_Vector y, N_Vector yB,
                          N_Vector fyB, SUNMatrix JB, void *user_dataB,
                          N_Vector tmp1B, N_Vector tmp2B, N_Vector tmp3B);
```

```python
CVLsJacFnB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, c_N_Vector,
    c_SUNMatrix, ctypes.c_void_p,
    c_N_Vector, c_N_Vector, c_N_Vector,
)
```

### Backward quadrature RHS (for `CVodeQuadInitB`)

```c
typedef int (*CVQuadRhsFnB)(sunrealtype t, N_Vector y, N_Vector yB,
                            N_Vector qBdot, void *user_dataB);
```

```python
CVQuadRhsFnB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    c_sunrealtype, c_N_Vector, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)
```

### Forward sensitivity RHS

```c
typedef int (*CVSensRhsFn)(int Ns, sunrealtype t,
                           N_Vector y, N_Vector ydot,
                           N_Vector *yS, N_Vector *ySdot,
                           void *user_data,
                           N_Vector tmp1, N_Vector tmp2);
```

```python
CVSensRhsFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int, c_sunrealtype,
    c_N_Vector, c_N_Vector,
    ctypes.POINTER(c_N_Vector), ctypes.POINTER(c_N_Vector),
    ctypes.c_void_p,
    c_N_Vector, c_N_Vector,
)
```

Note `yS` and `ySdot` are *arrays of N_Vector*. Treat as
`ctypes.POINTER(c_N_Vector)`; index into them via Python.

---

## 6. SUNContext and serial N_Vector

```c
int  SUNContext_Create(SUNComm comm, SUNContext *sunctx);
int  SUNContext_Free(SUNContext *sunctx);

N_Vector N_VNew_Serial(sunindextype length, SUNContext sunctx);
N_Vector N_VMake_Serial(sunindextype length, sunrealtype *data, SUNContext sunctx);
void     N_VDestroy_Serial(N_Vector v);
sunrealtype *N_VGetArrayPointer(N_Vector v);
N_Vector N_VClone(N_Vector w);
```

ctypes:

```python
def _bind(lib):
    lib.SUNContext_Create.argtypes = [ctypes.c_void_p, ctypes.POINTER(c_SUNContext)]
    lib.SUNContext_Create.restype  = ctypes.c_int
    # Note: SUNComm in 7.x is an opaque MPI_Comm wrapper; pass None (NULL) when no MPI.

    lib.SUNContext_Free.argtypes = [ctypes.POINTER(c_SUNContext)]
    lib.SUNContext_Free.restype  = ctypes.c_int

    lib.N_VNew_Serial.argtypes = [c_sunindextype, c_SUNContext]
    lib.N_VNew_Serial.restype  = c_N_Vector

    lib.N_VMake_Serial.argtypes = [c_sunindextype,
                                    ctypes.POINTER(c_sunrealtype), c_SUNContext]
    lib.N_VMake_Serial.restype  = c_N_Vector

    lib.N_VDestroy_Serial.argtypes = [c_N_Vector]
    lib.N_VDestroy_Serial.restype  = None

    lib.N_VGetArrayPointer.argtypes = [c_N_Vector]
    lib.N_VGetArrayPointer.restype  = ctypes.POINTER(c_sunrealtype)

    lib.N_VClone.argtypes = [c_N_Vector]
    lib.N_VClone.restype  = c_N_Vector
```

Reading/writing N_Vector data via numpy (zero-copy):

```python
def nvector_view(nv, length):
    """Return a numpy array that aliases the N_Vector's data buffer."""
    ptr = lib.N_VGetArrayPointer(nv)
    return np.ctypeslib.as_array(ptr, shape=(length,))
```

This gives a numpy view directly into SUNDIALS' memory. **Mutate in
place.** Don't replace the array; SUNDIALS only sees writes through
this aliased buffer.

---

## 7. Dense linear solver

`SUNLinSol_Dense` is not exported from Cantera; use
`SUNLinSol_LapackDense` instead (LAPACK-backed, generally faster
anyway).

```c
SUNMatrix       SUNDenseMatrix(sunindextype M, sunindextype N, SUNContext sunctx);
void            SUNMatDestroy(SUNMatrix A);
int             SUNMatZero(SUNMatrix A);
SUNLinearSolver SUNLinSol_LapackDense(N_Vector y, SUNMatrix A, SUNContext sunctx);
int             SUNLinSolFree(SUNLinearSolver S);
```

```python
lib.SUNDenseMatrix.argtypes = [c_sunindextype, c_sunindextype, c_SUNContext]
lib.SUNDenseMatrix.restype  = c_SUNMatrix
lib.SUNMatDestroy.argtypes  = [c_SUNMatrix]
lib.SUNMatDestroy.restype   = None
lib.SUNMatZero.argtypes     = [c_SUNMatrix]
lib.SUNMatZero.restype      = ctypes.c_int
lib.SUNLinSol_LapackDense.argtypes = [c_N_Vector, c_SUNMatrix, c_SUNContext]
lib.SUNLinSol_LapackDense.restype  = c_SUNLinearSolver
lib.SUNLinSolFree.argtypes  = [c_SUNLinearSolver]
lib.SUNLinSolFree.restype   = ctypes.c_int
```

The dense matrix is column-major laid out flat. Cantera exports
the official accessors for user-supplied Jacobian callbacks:

```c
sunrealtype  *SUNDenseMatrix_Data(SUNMatrix A);    /* flat column-major buffer */
sunindextype  SUNDenseMatrix_Rows(SUNMatrix A);
sunindextype  SUNDenseMatrix_Columns(SUNMatrix A);
sunindextype  SUNDenseMatrix_LData(SUNMatrix A);   /* M*N */
sunrealtype  *SUNDenseMatrix_Column(SUNMatrix A, sunindextype j);
```

```python
lib.SUNDenseMatrix_Data.argtypes    = [c_SUNMatrix]
lib.SUNDenseMatrix_Data.restype     = ctypes.POINTER(c_sunrealtype)
lib.SUNDenseMatrix_Rows.argtypes    = [c_SUNMatrix]
lib.SUNDenseMatrix_Rows.restype     = c_sunindextype
lib.SUNDenseMatrix_Columns.argtypes = [c_SUNMatrix]
lib.SUNDenseMatrix_Columns.restype  = c_sunindextype
```

Recommended Jacobian callback pattern: take a column-major numpy view
over `SUNDenseMatrix_Data`, write `J` directly in column-major order
(or write `J.T` if it's easier to think in row-major and let numpy
handle the transpose).

---

## 8. CVODE core (forward solver)

```c
void *CVodeCreate(int lmm, SUNContext sunctx);
int   CVodeInit(void *cvode_mem, CVRhsFn f, sunrealtype t0, N_Vector y0);
int   CVodeReInit(void *cvode_mem, sunrealtype t0, N_Vector y0);
void  CVodeFree(void **cvode_mem);

int   CVodeSStolerances(void *cvode_mem, sunrealtype reltol, sunrealtype abstol);
int   CVodeSVtolerances(void *cvode_mem, sunrealtype reltol, N_Vector abstol);

int   CVodeSetUserData(void *cvode_mem, void *user_data);
int   CVodeSetLinearSolver(void *cvode_mem, SUNLinearSolver LS, SUNMatrix M);
int   CVodeSetJacFn(void *cvode_mem, CVLsJacFn jac);
int   CVodeSetMaxStep(void *cvode_mem, sunrealtype hmax);
int   CVodeSetInitStep(void *cvode_mem, sunrealtype hin);
int   CVodeSetMaxOrd(void *cvode_mem, int maxord);
int   CVodeSetMaxNumSteps(void *cvode_mem, long int mxsteps);
int   CVodeSetMaxErrTestFails(void *cvode_mem, int maxnef);
int   CVodeSetMaxNonlinIters(void *cvode_mem, int maxcor);
int   CVodeSetMaxConvFails(void *cvode_mem, int maxncf);

int   CVode(void *cvode_mem, sunrealtype tout, N_Vector yout,
            sunrealtype *tret, int itask);
int   CVodeGetDky(void *cvode_mem, sunrealtype t, int k, N_Vector dky);
int   CVodeGetCurrentStep(void *cvode_mem, sunrealtype *hcur);
int   CVodeGetNumSteps(void *cvode_mem, long int *nsteps);
char *CVodeGetReturnFlagName(long int flag);
```

ctypes (representative subset; the full set goes in `sundials.py`):

```python
lib.CVodeCreate.argtypes = [ctypes.c_int, c_SUNContext]
lib.CVodeCreate.restype  = ctypes.c_void_p

lib.CVodeInit.argtypes = [ctypes.c_void_p, CVRhsFn, c_sunrealtype, c_N_Vector]
lib.CVodeInit.restype  = ctypes.c_int

lib.CVodeReInit.argtypes = [ctypes.c_void_p, c_sunrealtype, c_N_Vector]
lib.CVodeReInit.restype  = ctypes.c_int

lib.CVodeFree.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.CVodeFree.restype  = None

lib.CVodeSStolerances.argtypes = [ctypes.c_void_p, c_sunrealtype, c_sunrealtype]
lib.CVodeSStolerances.restype  = ctypes.c_int

lib.CVodeSVtolerances.argtypes = [ctypes.c_void_p, c_sunrealtype, c_N_Vector]
lib.CVodeSVtolerances.restype  = ctypes.c_int

lib.CVode.argtypes = [ctypes.c_void_p, c_sunrealtype, c_N_Vector,
                       ctypes.POINTER(c_sunrealtype), ctypes.c_int]
lib.CVode.restype  = ctypes.c_int

lib.CVodeGetDky.argtypes = [ctypes.c_void_p, c_sunrealtype,
                             ctypes.c_int, c_N_Vector]
lib.CVodeGetDky.restype  = ctypes.c_int
```

---

## 9. CVODES adjoint sensitivity

```c
int CVodeAdjInit(void *cvode_mem, long int steps, int interp);
int CVodeF(void *cvode_mem, sunrealtype tout, N_Vector yout,
           sunrealtype *tret, int itask, int *ncheckPtr);

int CVodeCreateB(void *cvode_mem, int lmmB, int *whichB);
int CVodeInitB(void *cvode_mem, int whichB, CVRhsFnB fB,
               sunrealtype tB0, N_Vector yB0);
int CVodeReInitB(void *cvode_mem, int whichB,
                 sunrealtype tB0, N_Vector yB0);
int CVodeSStolerancesB(void *cvode_mem, int whichB,
                       sunrealtype reltolB, sunrealtype abstolB);
int CVodeSVtolerancesB(void *cvode_mem, int whichB,
                       sunrealtype reltolB, N_Vector abstolB);
int CVodeSetLinearSolverB(void *cvode_mem, int whichB,
                          SUNLinearSolver LS, SUNMatrix M);
int CVodeSetJacFnB(void *cvode_mem, int whichB, CVLsJacFnB jacB);
int CVodeB(void *cvode_mem, sunrealtype tBout, int itask);
int CVodeGetB(void *cvode_mem, int whichB,
              sunrealtype *tBret, N_Vector yB);

int CVodeQuadInitB(void *cvode_mem, int whichB,
                   CVQuadRhsFnB fQB, N_Vector yQB0);
int CVodeQuadReInitB(void *cvode_mem, int whichB, N_Vector yQB0);
int CVodeQuadSStolerancesB(void *cvode_mem, int whichB,
                           sunrealtype reltolQB, sunrealtype abstolQB);
int CVodeQuadSVtolerancesB(void *cvode_mem, int whichB,
                           sunrealtype reltolQB, N_Vector abstolQB);
int CVodeGetQuadB(void *cvode_mem, int whichB,
                  sunrealtype *tBret, N_Vector qB);
```

ctypes:

```python
lib.CVodeAdjInit.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int]
lib.CVodeAdjInit.restype  = ctypes.c_int

lib.CVodeF.argtypes = [ctypes.c_void_p, c_sunrealtype, c_N_Vector,
                        ctypes.POINTER(c_sunrealtype), ctypes.c_int,
                        ctypes.POINTER(ctypes.c_int)]
lib.CVodeF.restype  = ctypes.c_int

lib.CVodeCreateB.argtypes = [ctypes.c_void_p, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int)]
lib.CVodeCreateB.restype  = ctypes.c_int

lib.CVodeInitB.argtypes = [ctypes.c_void_p, ctypes.c_int, CVRhsFnB,
                            c_sunrealtype, c_N_Vector]
lib.CVodeInitB.restype  = ctypes.c_int

lib.CVodeReInitB.argtypes = [ctypes.c_void_p, ctypes.c_int,
                              c_sunrealtype, c_N_Vector]
lib.CVodeReInitB.restype  = ctypes.c_int

lib.CVodeB.argtypes = [ctypes.c_void_p, c_sunrealtype, ctypes.c_int]
lib.CVodeB.restype  = ctypes.c_int

lib.CVodeGetB.argtypes = [ctypes.c_void_p, ctypes.c_int,
                           ctypes.POINTER(c_sunrealtype), c_N_Vector]
lib.CVodeGetB.restype  = ctypes.c_int

lib.CVodeQuadInitB.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                CVQuadRhsFnB, c_N_Vector]
lib.CVodeQuadInitB.restype  = ctypes.c_int

lib.CVodeQuadReInitB.argtypes = [ctypes.c_void_p, ctypes.c_int, c_N_Vector]
lib.CVodeQuadReInitB.restype  = ctypes.c_int

lib.CVodeGetQuadB.argtypes = [ctypes.c_void_p, ctypes.c_int,
                               ctypes.POINTER(c_sunrealtype), c_N_Vector]
lib.CVodeGetQuadB.restype  = ctypes.c_int
```

### Per-output-time loop for adjoint

The CVODES adjoint API requires this exact sequence per output time
`t_m`:

```python
def adjoint_at_tm(t_m, terminal_condition, user_dataB):
    """Compute sensitivities at one output time."""
    # 1. (re)initialize the backward problem at this t_m
    yB0 = np_to_nvector(terminal_condition)
    if first_call:
        whichB = ctypes.c_int()
        _check(lib.CVodeCreateB(cvode_mem, CV_BDF, ctypes.byref(whichB)),
               "CVodeCreateB")
        _check(lib.CVodeInitB(cvode_mem, whichB, fB_callback, t_m, yB0),
               "CVodeInitB")
        _check(lib.CVodeSVtolerancesB(cvode_mem, whichB, rtolB, abstolB_vec),
               "CVodeSVtolerancesB")
        _check(lib.CVodeSetLinearSolverB(cvode_mem, whichB, LS_B, mat_B),
               "CVodeSetLinearSolverB")
        _check(lib.CVodeSetJacFnB(cvode_mem, whichB, jacB_callback),
               "CVodeSetJacFnB")
        # quadrature
        qB0 = np_to_nvector(np.zeros(n_rxns))
        _check(lib.CVodeQuadInitB(cvode_mem, whichB, fQB_callback, qB0),
               "CVodeQuadInitB")
        _check(lib.CVodeQuadSVtolerancesB(cvode_mem, whichB, rtolQB, atolQB_vec),
               "CVodeQuadSVtolerancesB")
    else:
        # subsequent output times reuse the same problem ID
        _check(lib.CVodeReInitB(cvode_mem, whichB, t_m, yB0), "CVodeReInitB")
        _check(lib.CVodeQuadReInitB(cvode_mem, whichB, qB0_zero), "CVodeQuadReInitB")

    # 2. integrate backward from t_m to 0
    _check(lib.CVodeB(cvode_mem, 0.0, CV_NORMAL), "CVodeB")

    # 3. read back the adjoint state and quadrature
    tBret = c_sunrealtype()
    yB_final = np_to_nvector(np.zeros(n_state))
    qB_final = np_to_nvector(np.zeros(n_rxns))
    _check(lib.CVodeGetB(cvode_mem, whichB, ctypes.byref(tBret), yB_final),
           "CVodeGetB")
    _check(lib.CVodeGetQuadB(cvode_mem, whichB, ctypes.byref(tBret), qB_final),
           "CVodeGetQuadB")

    return nvector_view(qB_final, n_rxns).copy()
```

For multiple output times, call `adjoint_at_tm` in **decreasing** order
of `t_m` — SUNDIALS' adjoint internally moves backward through the
forward trajectory, so re-initialization is most efficient when the
new `t_m` is earlier than (or equal to) the previous one.

Each `t_m` independently provides its own terminal condition; the
backward problem is reset on every call via `CVodeReInitB +
CVodeQuadReInitB`.

---

## 10. ctypes callback lifetime (critical)

ctypes `CFUNCTYPE` instances must be kept alive as long as SUNDIALS
holds a pointer to them. **This is the #1 cause of mysterious segfaults
in SUNDIALS-via-ctypes work.** If the wrapper class doesn't store the
CFUNCTYPE objects as attributes, Python may garbage-collect them while
SUNDIALS is mid-step, and the next callback dispatch crashes.

Recommended pattern:

```python
class CVodeIntegrator:
    def __init__(self, n_state, rhs, jac, user_data=None, ...):
        # Wrap the Python callables in CFUNCTYPE instances. Hold the
        # wrappers as attributes — DO NOT pass freshly-created
        # CFUNCTYPE instances to SUNDIALS without storing them.
        self._rhs_cb = CVRhsFn(self._rhs_trampoline)
        self._jac_cb = CVLsJacFn(self._jac_trampoline)
        self._py_rhs = rhs                 # the user's Python function
        self._py_jac = jac
        self._user_data = user_data
        # ... call CVodeInit with self._rhs_cb ...

    def _rhs_trampoline(self, t, y, ydot, user_data):
        """Static-signature wrapper. Forwards to self._py_rhs."""
        try:
            y_view  = nvector_view(y, self.n_state)
            yd_view = nvector_view(ydot, self.n_state)
            self._py_rhs(t, y_view, yd_view)
            return 0
        except Exception:
            traceback.print_exc()
            return -1                       # negative = recoverable error
```

Wrappers can return:
- `0` for success
- positive flag for "recoverable error" (SUNDIALS may retry with smaller step)
- negative flag for "unrecoverable error" (integration aborts)

For Python exceptions in callbacks: log via traceback, return -1.
Letting an exception propagate from a ctypes callback **crashes the
interpreter**.

---

## 11. Forward sensitivity (CVODES native)

```c
int CVodeSensInit(void *cvode_mem, int Ns, int ism,
                  CVSensRhsFn fS, N_Vector *yS0);
int CVodeSensInit1(void *cvode_mem, int Ns, int ism,
                   CVSensRhs1Fn fS1, N_Vector *yS0);
int CVodeSensReInit(void *cvode_mem, int ism, N_Vector *yS0);
int CVodeSensSStolerances(void *cvode_mem, sunrealtype reltol,
                          sunrealtype *abstol);
int CVodeSensSVtolerances(void *cvode_mem, sunrealtype reltol,
                          N_Vector *abstol);
int CVodeSensEEtolerances(void *cvode_mem);   /* estimate from estimated reltol/atol */
int CVodeGetSens(void *cvode_mem, sunrealtype *tret, N_Vector *yS);
int CVodeGetSensDky(void *cvode_mem, sunrealtype t, int k, N_Vector *dkyS);
int CVodeGetSens1(void *cvode_mem, sunrealtype *tret, int is, N_Vector yS);
```

`N_Vector *` here means an array of N_Vectors (one per parameter).

For Frhodo's forward-sens path, the simplest choice is:
- `ism = CV_STAGGERED` (separate sensitivity solve after each step;
  more stable than `CV_SIMULTANEOUS` for stiff problems)
- `fS = CVSensRhsFn` that computes `∂f/∂p_j · 1 + Jᵀ · S_j` for each
  reaction `j` — same `_shock_param_rhs_gradient` we already have
- `CVodeSensSVtolerances` for vector atol on each sensitivity vector
  (same scaling as the state-vector atol)

---

## 12. Worked example — forward solve of a toy ODE

A complete, self-contained example: integrate the harmonic oscillator
`dx/dt = v; dv/dt = -ω²x` from t=0 to t=2π with rtol=1e-8 against
analytical sin/cos.

```python
import ctypes
import numpy as np

# Load Cantera's compiled extension (which contains SUNDIALS)
cantera_so = "/path/to/cantera/_cantera.cpython-311-x86_64-linux-gnu.so"
lib = ctypes.CDLL(cantera_so, mode=ctypes.RTLD_GLOBAL)

# Set up types (abbreviated; full set lives in sundials.py)
c_sunrealtype = ctypes.c_double
c_SUNContext  = ctypes.c_void_p
c_N_Vector    = ctypes.c_void_p
c_SUNMatrix   = ctypes.c_void_p
c_SUNLS       = ctypes.c_void_p

CVRhsFn = ctypes.CFUNCTYPE(
    ctypes.c_int, c_sunrealtype, c_N_Vector, c_N_Vector, ctypes.c_void_p,
)

# Bind only what we use here
for fn, args, res in [
    ("SUNContext_Create", [ctypes.c_void_p, ctypes.POINTER(c_SUNContext)], ctypes.c_int),
    ("SUNContext_Free",   [ctypes.POINTER(c_SUNContext)], ctypes.c_int),
    ("N_VNew_Serial",     [ctypes.c_int64, c_SUNContext], c_N_Vector),
    ("N_VDestroy_Serial", [c_N_Vector], None),
    ("N_VGetArrayPointer",[c_N_Vector], ctypes.POINTER(c_sunrealtype)),
    ("SUNDenseMatrix",    [ctypes.c_int64, ctypes.c_int64, c_SUNContext], c_SUNMatrix),
    ("SUNMatDestroy",     [c_SUNMatrix], None),
    ("SUNLinSol_LapackDense", [c_N_Vector, c_SUNMatrix, c_SUNContext], c_SUNLS),
    ("SUNLinSolFree",     [c_SUNLS], ctypes.c_int),
    ("CVodeCreate",       [ctypes.c_int, c_SUNContext], ctypes.c_void_p),
    ("CVodeInit",         [ctypes.c_void_p, CVRhsFn, c_sunrealtype, c_N_Vector], ctypes.c_int),
    ("CVodeSStolerances", [ctypes.c_void_p, c_sunrealtype, c_sunrealtype], ctypes.c_int),
    ("CVodeSetLinearSolver", [ctypes.c_void_p, c_SUNLS, c_SUNMatrix], ctypes.c_int),
    ("CVode",             [ctypes.c_void_p, c_sunrealtype, c_N_Vector,
                            ctypes.POINTER(c_sunrealtype), ctypes.c_int], ctypes.c_int),
    ("CVodeFree",         [ctypes.POINTER(ctypes.c_void_p)], None),
]:
    f = getattr(lib, fn)
    f.argtypes = args
    f.restype = res

CV_BDF = 2
CV_NORMAL = 1

# --- problem setup ---
omega = 2.0
def rhs_py(t, y_ptr, ydot_ptr, ud):
    y = np.ctypeslib.as_array(lib.N_VGetArrayPointer(y_ptr), shape=(2,))
    ydot = np.ctypeslib.as_array(lib.N_VGetArrayPointer(ydot_ptr), shape=(2,))
    ydot[0] = y[1]
    ydot[1] = -omega * omega * y[0]
    return 0

rhs_cb = CVRhsFn(rhs_py)   # MUST stay alive while CVode holds it

# --- SUNDIALS objects ---
ctx = c_SUNContext()
lib.SUNContext_Create(None, ctypes.byref(ctx))

y0 = lib.N_VNew_Serial(2, ctx)
y0_view = np.ctypeslib.as_array(lib.N_VGetArrayPointer(y0), shape=(2,))
y0_view[0] = 1.0  # x(0) = 1
y0_view[1] = 0.0  # v(0) = 0

A = lib.SUNDenseMatrix(2, 2, ctx)
LS = lib.SUNLinSol_LapackDense(y0, A, ctx)

cvode_mem = lib.CVodeCreate(CV_BDF, ctx)
lib.CVodeInit(cvode_mem, rhs_cb, 0.0, y0)
lib.CVodeSStolerances(cvode_mem, 1e-8, 1e-12)
lib.CVodeSetLinearSolver(cvode_mem, LS, A)

# --- integrate to t = 2π / ω (one full period) ---
t_target = 2 * np.pi / omega
tret = c_sunrealtype(0.0)
flag = lib.CVode(cvode_mem, t_target, y0, ctypes.byref(tret), CV_NORMAL)
assert flag == 0, f"CVode failed: flag {flag}"

# --- compare to analytical ---
x_analytical = np.cos(omega * t_target)
v_analytical = -omega * np.sin(omega * t_target)
print(f"x: numerical={y0_view[0]:.6e}  analytical={x_analytical:.6e}")
print(f"v: numerical={y0_view[1]:.6e}  analytical={v_analytical:.6e}")

# --- cleanup ---
cvode_mem_p = ctypes.c_void_p(cvode_mem)
lib.CVodeFree(ctypes.byref(cvode_mem_p))
lib.SUNLinSolFree(LS)
lib.SUNMatDestroy(A)
lib.N_VDestroy_Serial(y0)
lib.SUNContext_Free(ctypes.byref(ctx))
```

Expected output (one full period of the harmonic oscillator from rest):

```
x: numerical=1.000000e+00  analytical=1.000000e+00
v: numerical=0.000000e+00  analytical=0.000000e+00
```

agreeing to ~`1e-8`.

This example demonstrates the full lifecycle: context, vectors,
matrix, linear solver, CVode setup, integration, cleanup. It also
shows the callback-lifetime pattern (storing `rhs_cb` in a local that
outlives the `CVode` call).

---

## 13. References

- SUNDIALS 7.4.0 user documentation:
  <https://sundials.readthedocs.io/en/v7.4.0/>
- CVODES user guide:
  <https://sundials.readthedocs.io/en/v7.4.0/cvodes/index.html>
- CVODES adjoint sensitivity (§5):
  <https://sundials.readthedocs.io/en/v7.4.0/cvodes/Mathematics_link.html#adjoint-sensitivity-analysis>
- CVODES forward sensitivity (§4):
  <https://sundials.readthedocs.io/en/v7.4.0/cvodes/Mathematics_link.html#forward-sensitivity-analysis>
- SUNDIALS source for type and constant definitions:
  <https://github.com/LLNL/sundials/tree/v7.4.0/include/sundials>
- [`development/verify_cantera_sundials.py`](../development/verify_cantera_sundials.py)
  — symbol verification recipe for any Cantera/SUNDIALS version
- [`doc/incident_shock_derivation.md`](incident_shock_derivation.md)
  — analytical `∂f/∂y` and `∂f/∂p` used as user-supplied callbacks
