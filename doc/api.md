# Frhodo Public API

The `frhodo` package exposes a typed, pydantic-validated API for loading
mechanisms, running shock-tube and 0-D reactor simulations, and
optimizing rate coefficients against experimental data. Every name
listed in `frhodo.__all__` is part of the supported surface.

```python
import frhodo
print(sorted(frhodo.__all__))
```

---

## 1. Quick start

```python
from frhodo import (
    load_mechanism,
    PreShockState, ShockTubeConfig,
    run_shock_tube,
)

mech = load_mechanism("mech.yaml")

cfg = ShockTubeConfig(
    initial=PreShockState(
        T1=294.0, P1=601.0, u1=1029.0,
        composition={"Kr": 0.96, "cC7H14": 0.04},
    ),
    t_end=5e-5,
)
result = run_shock_tube(mech, cfg)
if result.success:
    print(f"post-shock T2 = {result.shock.T2:.0f} K")
    print(f"observable peak @ t = {result.t[result.observable.argmax()]*1e6:.2f} µs")
```

---

## 2. Mechanism loading

### `load_mechanism(mech, *, thermo=None, converted_yaml=None) -> ChemicalMechanism`

Loads a Cantera mechanism from YAML, CTI, or Chemkin input.

- YAML inputs (`.yaml` / `.yml`) load directly.
- CTI and Chemkin inputs are converted to YAML on the fly. The output
  path defaults to `<mech>.converted.yaml` next to the input; override
  with `converted_yaml=`.
- `thermo` is the Chemkin thermodynamic database file (ignored for
  YAML inputs).

```python
mech = load_mechanism("mech.yaml")
mech = load_mechanism("mech.inp", thermo="thermo.dat")
mech = load_mechanism(
    "mech.inp", thermo="thermo.dat",
    converted_yaml="/tmp/converted.yaml",
)
```

---

## 3. Composition

`composition` is `dict[str, float]` everywhere. Use `parse_composition`
to convert Cantera-style strings at the boundary:

```python
from frhodo import parse_composition
mix = parse_composition("Kr:0.96, cC7H14:0.04")  # → {"Kr": 0.96, ...}
```

The parser accepts entries separated by commas or newlines and tolerates
whitespace around names and fractions. Malformed input raises
`ValueError`.

---

## 4. Unit helpers

Cantera works internally in J/kmol. Two helpers convert from the units
most users think in:

```python
from frhodo import kcal_per_mol, kJ_per_mol
kcal_per_mol(5.0)   # 5 kcal/mol → 2.092e7 J/kmol
kJ_per_mol(20.0)    # 20 kJ/mol → 2e7 J/kmol
```

---

## 5. Reactor configuration

### Initial state — `PreShockState` vs. `PostShockState`

`ShockTubeConfig.initial` is a discriminated union (`kind` field).

`PreShockState` carries measured zone-1 conditions; `run_shock_tube`
auto-solves the normal-shock jump before integrating the reactor.

```python
PreShockState(
    T1=294.0, P1=601.0, u1=1029.0,
    composition={"Kr": 0.96, "cC7H14": 0.04},
)
```

`PostShockState` carries already-resolved reactor-inlet conditions.
The reactor starts directly from these values; no jump is solved.

```python
PostShockState(
    T_reac=1616.0, P_reac=147140.0,
    u_incident=181.85, rho1=0.023,
    composition={"Kr": 0.96, "cC7H14": 0.04},
)
```

### `ShockTubeConfig`

```python
ShockTubeConfig(
    initial=PreShockState | PostShockState,
    t_end: float,                             # seconds
    A1: float = 0.2,
    As: float = 0.2,
    L: float = 0.1,
    area_change: bool = False,
    solver: SolverSettings = SolverSettings(),
    observable: ObservableSettings = ObservableSettings(),
    save_times: list[float] | None = None,
)
```

### `ZeroDConfig`

```python
ZeroDConfig(
    mode: Literal["constant_volume", "constant_pressure"],
    T_reac: float,
    P_reac: float,
    composition: dict[str, float],
    t_end: float,
    solve_energy: bool = True,
    frozen_comp: bool = False,
    solver: SolverSettings = SolverSettings(rtol=1e-4, atol=1e-7),
    observable: ObservableSettings = ObservableSettings(
        main="Concentration", sub=0,
    ),
    save_times: list[float] | None = None,
)
```

### `SolverSettings`

```python
SolverSettings(
    solver: Literal["CVODES", "BDF", "LSODA", "Radau"] = "CVODES",
    rtol: float | None = None,
    atol: float | None = None,
    sim_interp_factor: int = 1,
)
```

`solver="CVODES"` runs through Frhodo's in-tree SUNDIALS binding
(`frhodo.simulation.numerics.sundials`) with the analytical Jacobian.
Default tolerances are `rtol=1e-6 / atol=1e-10`. `BDF`, `LSODA`, and
`Radau` go through `scipy.solve_ivp` with looser defaults
(`rtol=1e-4 / atol=1e-7`); leaving `rtol`/`atol` unset selects the
defaults matching the chosen backend.

### `ObservableSettings`

Selects which trace the reactor emits as
`SimulationResult.observable`.

```python
ObservableSettings(
    main: str = "Density Gradient",
    sub: int | str = 0,
)
```

---

## 6. Running simulations

### `run_shock_tube(mech, cfg) -> SimulationResult`

One incident-shock-reactor run. When `cfg.initial` is a
`PreShockState`, the jump conditions are computed first and attached
to `result.shock`.

### `run_zero_d(mech, cfg) -> SimulationResult`

One 0-D constant-volume or constant-pressure reactor run.

### `solve_shock_jump(initial, mech) -> ShockState`

Standalone normal-shock solver. Used internally by `run_shock_tube`
when `initial=PreShockState`; expose it for callers that need only the
jump conditions.

### `run_shock_tubes(mech, cfgs, *, workers=None) -> list[SimulationResult]`

Batch runner.

- `workers=None` or `1`: sequential, in-process.
- `workers > 1` on Linux/macOS: `fork` pool inherits the loaded mech.
- `workers > 1` on Windows: requires `mech` as a `str | Path`; each
  worker loads its own copy via `load_mechanism`.

```python
mech = load_mechanism("mech.yaml")
cfgs = [ShockTubeConfig(initial=..., t_end=5e-5) for ...]
results = run_shock_tubes(mech, cfgs, workers=8)        # Linux / macOS
results = run_shock_tubes("mech.yaml", cfgs, workers=8) # Windows-safe
```

---

## 7. Result types

### `SimulationResult` (frozen dataclass)

```
success           : bool
failure_reason    : FailureReason | None
message           : str
t                 : np.ndarray            # shape (N,)
T, P, rho         : np.ndarray            # shape (N,)
Y                 : np.ndarray            # mass fractions, shape (N, n_species)
species           : tuple[str, ...]
observable        : np.ndarray            # shape (N,)
h_tot, s_tot      : np.ndarray | None     # optional thermo properties
wdot              : np.ndarray | None
HRR_tot           : np.ndarray | None
drhodz_tot        : np.ndarray | None
shock             : ShockState | None     # populated when PreShockState was used
cantera_array     : ct.SolutionArray | None  # internal; not part of contract
```

Optional fields are `None` when the underlying reactor didn't expose
them, never empty arrays.

### `ShockState`

```
success    : bool
message    : str
initial    : PreShockState | None  # the input that produced this state
T1, P1, u1 : float                 # zone 1
T2, P2, u2 : float                 # zone 2 (post-incident-shock)
T5, P5     : float                 # zone 5 (post-reflected-shock)
rho1       : float
```

---

## 8. Optimization

### What gets optimized — `OptimizableSpec`

`OptimizableSpec.rates` is a list of `OptimizableRate` entries. Each
entry declares one reaction is optimizable and supplies an
uncertainty.

```python
from frhodo import (
    OptimizableSpec, OptimizableRate,
    RateUncertainty, CoefUncertainty,
    kcal_per_mol,
)

spec = OptimizableSpec(rates=[
    # rate-level uncertainty: k ∈ (k0 / 2, k0 * 2)
    OptimizableRate(rxn_idx=2, rate=RateUncertainty(factor=2.0)),

    # per-coefficient overrides; only A and Ea are fit
    OptimizableRate(
        rxn_idx=7,
        coefficients={
            "pre_exponential_factor": CoefUncertainty(factor=2.0),
            "activation_energy":      CoefUncertainty(delta=kcal_per_mol(5.0)),
        },
        optimize=["pre_exponential_factor", "activation_energy"],
    ),

    # pressure-dependent reactions are recast to Troe and the full
    # 10-element parameter set is always fit; only the rate-level
    # uncertainty applies
    OptimizableRate(rxn_idx=21, rate=RateUncertainty(factor=10.0)),
])
```

#### `CoefUncertainty`

Exactly one of `factor` / `delta` / `bounds`:

- `factor`: multiplicative band around the nominal value.
- `delta`: additive band (nominal ± delta).
- `bounds = (lo, hi)`: absolute bounds; must bracket the nominal value.

#### `RateUncertainty(factor=2.0)`

Rate-level multiplicative uncertainty. Applies at every (T, P) sample
point the optimizer evaluates.

#### Pressure-dependent reactions

Plog, Falloff, Lindemann, Sri, Tsang, and Troe reactions are recast to
Troe before fitting. Passing `coefficients` overrides or `optimize`
subsets for such reactions raises at `.build()` time — the recast
demands all 10 Troe parameters be fit jointly.

#### Mutable builder (for GUI tree state)

```python
from frhodo import OptimizableSpecBuilder
b = OptimizableSpecBuilder()
b.set_rxn(2, enabled=True, rate=RateUncertainty(factor=2.0))
b.clear_rxn(5)
spec = b.build()
```

### Experimental data — `ExperimentShock`

One trace + initial state per shock. Storage is `list[float]` (so the
model serializes cleanly); call `.t_array()` / `.observable_array()`
to obtain `np.ndarray` views.

```python
from frhodo import ExperimentShock, PreShockState

shock = ExperimentShock(
    t=t_array,                       # ndarray accepted; auto-coerced
    observable=obs_array,
    initial=PreShockState(T1=..., P1=..., u1=..., composition={...}),
    t_end=5e-5,
    scalar_weight=1.0,
    weight_profile=WeightProfile(...),       # optional override
    uncertainty_profile=UncertaintyProfile(...),  # Bayesian mode only
)
```

### Weighting and uncertainty envelopes

`WeightProfile` is a two-sided sigmoid envelope applied per sample.

```python
WeightProfile(
    peak: float = 100.0,
    floor_pre: float = 0.0,
    floor_post: float = 0.0,
    time_rise: float = 4.5,            # % of trace duration
    time_fall: float = 35.0,
    growth_rate_rise: float = 0.0,
    growth_rate_fall: float = 0.7,
    absolute_time: bool = False,
    cutoff_pre: float = 0.0,           # samples outside are dropped
    cutoff_post: float = 100.0,
)
```

`UncertaintyProfile` is the same shape for Bayesian-cost mode, plus a
`kind` literal (`"percent"`, `"factor"`, `"abs"`, `"abs_plus"`,
`"abs_minus"`) and `wavelet_levels`.

### Algorithm — `AlgorithmSettings`

Two stages (global then local), each independently enabled.

```python
from frhodo import AlgorithmSettings, AlgorithmStage

algo = AlgorithmSettings(
    global_stage=AlgorithmStage(
        algorithm="RBFOpt", enabled=True, max_eval=500,
    ),
    local_stage=AlgorithmStage(
        algorithm="Subplex", enabled=True, max_eval=2500,
        xtol_rel=1e-4,
    ),
)
```

Algorithm labels mirror the GUI: `"Subplex"`, `"Nelder-Mead Simplex"`,
`"COBYLA"`, `"BOBYQA"`, `"DIRECT"`, `"DIRECT-L"`, `"CRS2 (Controlled
Random Search)"`, `"RBFOpt"`, plus pygmo variants (`"DE
(Differential Evolution)"`, `"PSO (Particle Swarm Optimization)"`,
etc.). Unknown labels raise on `to_legacy_dict()`.

Stage stop criteria: `"Iteration Maximum"` (default) or `"Maximum
Time [min]"`; `stop_value` is the limit.

### Cost — `CostSettings`

Set the objective and loss shape:

```python
from frhodo import CostSettings

cost = CostSettings(
    obj_fcn_type="Residual",     # or "Bayesian"
    scale="Linear",              # "Linear" | "Log" | "Bisymlog"
    loss_alpha=2.0,              # or "Adaptive"
    loss_c=1.0,
    bisymlog_scaling_factor=1.0,
    bayes_dist_type="Automatic",
    bayes_unc_sigma=2.0,
)
```

### `OptimizationRequest`

Bundles everything one optimization run needs:

```python
from frhodo import OptimizationRequest, ObservableSettings

request = OptimizationRequest(
    shocks=[ExperimentShock(...), ExperimentShock(...)],
    optimizable=spec,
    reactor_state=runtime_reactor_state,    # see note below
    cost=cost,
    algorithm=algo,
    observable=ObservableSettings(main="Density Gradient", sub=0),
    default_weight_profile=WeightProfile(),
    default_uncertainty_profile=UncertaintyProfile(),
    time_uncertainty=0.0,
    multiprocessing=True,
    max_processors=8,
    display_shock_index=None,            # which shock's trace fires in IterationUpdate
    save_recast_path=None,               # Path to drop the recast Troe mech
)
```

`reactor_state` is currently `RuntimeReactorState` from
`frhodo.simulation.shock.state`; it carries solver knobs and the
reactor-type label. (Cleanup of this type's overlap with
`SolverSettings` is deferred; the field is still required.)

A shock-level `uncertainty_profile` is rejected unless
`cost.obj_fcn_type == "Bayesian"`.

### Callbacks — domain events

```python
from frhodo import (
    OptimizationCallbacks,
    StartInfo, IterationUpdate, StageComplete,
)

def on_start(info: StartInfo):
    print(f"starting: {info.n_shocks} shocks, "
          f"{len(info.optimizable_used.coefficients)} coefs")

def on_iteration(u: IterationUpdate):
    if u.is_best:
        print(f"  [{u.stage}] iter {u.iter}: best fval = {u.fval:.4g}")

def on_stage_complete(sc: StageComplete):
    print(f"{sc.stage} done: {sc.nfev} iters in {sc.elapsed_s:.1f}s, "
          f"fval={sc.fval:.4g}")

callbacks = OptimizationCallbacks(
    on_start=on_start,
    on_iteration=on_iteration,
    on_stage_complete=on_stage_complete,
    log=print,
    abort=lambda: stop_requested,
)
```

#### `StartInfo`

```
optimizable_used : OptimizableSet
recast_rxns      : tuple[int, ...]    # post-recast rxn indices
n_shocks         : int
max_processors   : int
```

#### `IterationUpdate`

```
iter         : int
fval         : float
x            : np.ndarray
is_best      : bool                   # True when this iter beat the prior best
stage        : "global" | "local"
elapsed_s    : float
ind_var      : np.ndarray | None      # display-shock trace, when configured
observable   : np.ndarray | None
stat_plot    : dict | None
```

#### `StageComplete`

```
stage        : "global" | "local"
success      : bool
message      : str
fval         : float
nfev         : int
elapsed_s    : float
shock_evals  : int                    # nfev * n_shocks
```

### `optimize_residual(mech, request, *, callbacks=None) -> OptimizationResult`

Runs the two-stage optimization.

```python
result = optimize_residual(mech, request, callbacks=callbacks)
print(f"final: fval={result.fval}, x={result.x}")
if result.aborted:
    print("aborted by user")
```

### `OptimizationResult`

```
success           : bool
message           : str
aborted           : bool
x                 : np.ndarray
fval              : float
nfev              : int
elapsed_s         : float
optimizable_used  : OptimizableSet | None
raw               : Mapping | None     # full driver dict
```

Use `optimizable_used` to interpret `x` slot-by-slot:

```python
i = result.optimizable_used.slot_index(rxn_idx=7, coef_name="activation_energy")
if i is not None:
    Ea_optimized = result.x[i]
```

### Applying results back to the mech — `apply_optimization_result`

```python
from frhodo import apply_optimization_result

# In-place: mutate mech.coeffs with the optimized x
apply_optimization_result(mech, result)

# Or write to disk; format is inferred from the suffix
apply_optimization_result(mech, result, save_path="optimized.yaml")
apply_optimization_result(mech, result, save_path="optimized.inp")
```

Suffix dispatch:
| Suffix | Format |
|---|---|
| `.yaml`, `.yml` | Cantera YAML |
| `.inp`, `.dat`, `.mech` | Chemkin |

Unknown suffixes raise `ValueError`.

To write both formats, call twice:

```python
apply_optimization_result(mech, result, save_path="optimized.yaml")
apply_optimization_result(mech, result, save_path="optimized.inp")
```

---

## 9. Logging

Diagnostic emission from core modules uses Python's standard `logging`:

```python
import logging

# Library / CLI use: silent unless the caller configures handlers.
logging.basicConfig(level=logging.INFO)

# Then any frhodo warning surfaces normally:
import frhodo
mech = frhodo.load_mechanism("mech.yaml")
```

Severity policy:
- `WARNING+` for recoverable load failures, unit-conversion overflows,
  shock-jump failures.
- `INFO` for successful step completions ("loaded 24 species…").
- Hard errors are raised, not logged.

GUI callers install `frhodo.common.logging.GuiLogHandler`, which routes
records to the log widget and maps `WARNING+` to the blinking-tab
alert.

The optimization callbacks (`log`, `on_iteration`, etc.) are a
separate channel — real-time progress hooks for one long-running
operation. They are not driven by `logging` records.

---

## 10. End-to-end example

```python
import numpy as np
from frhodo import (
    load_mechanism, parse_composition, kcal_per_mol,
    ShockTubeConfig, PreShockState,
    run_shock_tube, run_shock_tubes, solve_shock_jump,
    OptimizableSpec, OptimizableRate,
    RateUncertainty, CoefUncertainty,
    OptimizationRequest, OptimizationCallbacks,
    AlgorithmSettings, AlgorithmStage, CostSettings,
    ObservableSettings,
    WeightProfile, UncertaintyProfile, ExperimentShock,
    optimize_residual, apply_optimization_result,
    IterationUpdate, StartInfo, StageComplete,
)
from frhodo.simulation.shock.state import RuntimeReactorState

# 1. Load mech
mech = load_mechanism("mech.yaml")

# 2. Single shock
cfg = ShockTubeConfig(
    initial=PreShockState(T1=294, P1=601, u1=1029,
                          composition={"Kr": 0.96, "cC7H14": 0.04}),
    t_end=5e-5,
)
result = run_shock_tube(mech, cfg)
assert result.success
print(f"T2={result.shock.T2:.0f} K, peak @ {result.t[result.observable.argmax()]*1e6:.2f} µs")

# 3. Batch
cfgs = [
    ShockTubeConfig(
        initial=PreShockState(T1=294, P1=601, u1=u,
                              composition={"Kr": 0.96, "cC7H14": 0.04}),
        t_end=5e-5,
    )
    for u in np.linspace(900, 1200, 30)
]
results = run_shock_tubes(mech, cfgs, workers=4)

# 4. Optimization
spec = OptimizableSpec(rates=[
    OptimizableRate(rxn_idx=2, rate=RateUncertainty(factor=2.0)),
    OptimizableRate(
        rxn_idx=7,
        coefficients={
            "pre_exponential_factor": CoefUncertainty(factor=2.0),
            "activation_energy":      CoefUncertainty(delta=kcal_per_mol(5.0)),
        },
        optimize=["pre_exponential_factor", "activation_energy"],
    ),
])

emphasize_ignition = WeightProfile(
    peak=100, floor_pre=1, floor_post=5,
    time_rise=20, time_fall=60,
    growth_rate_rise=0.2, growth_rate_fall=1.0,
)

shocks = [
    ExperimentShock(
        t=t_array, observable=obs_array,
        initial=PreShockState(T1=t1, P1=p1, u1=u1, composition={"Kr": 0.96, "cC7H14": 0.04}),
        t_end=5e-5,
        scalar_weight=w, weight_profile=emphasize_ignition,
    )
    for (t_array, obs_array, t1, p1, u1, w) in experiment_data
]

request = OptimizationRequest(
    shocks=shocks,
    optimizable=spec,
    reactor_state=RuntimeReactorState(
        name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
        sim_interp_factor=1, ode_solver="CVODES",
        ode_rtol=1e-6, ode_atol=1e-10,
    ),
    cost=CostSettings(obj_fcn_type="Residual", scale="Bisymlog"),
    algorithm=AlgorithmSettings(),
    multiprocessing=True, max_processors=8,
)

def on_iter(u: IterationUpdate):
    if u.is_best:
        print(f"  [{u.stage}] iter {u.iter}: best fval = {u.fval:.4g}")

result = optimize_residual(mech, request, callbacks=OptimizationCallbacks(
    on_start=lambda info: print(f"starting on {info.n_shocks} shocks"),
    on_iteration=on_iter,
    on_stage_complete=lambda sc: print(f"{sc.stage} done in {sc.elapsed_s:.1f}s"),
    log=print,
    abort=lambda: should_stop(),
))

print(f"final fval={result.fval}, x={result.x}")

# 5. Apply back to mech, save
apply_optimization_result(mech, result, save_path="optimized.yaml")
```

---

## 11. Package surface — exported names

```
ChemicalMechanism
PreShockState, PostShockState, ShockTubeConfig, ZeroDConfig
SolverSettings, ObservableSettings
SimulationResult, ShockState, FailureReason

OptimizableSpec, OptimizableRate, OptimizableSet, OptimizableSpecBuilder
RateUncertainty, CoefUncertainty
WeightProfile, UncertaintyProfile, ExperimentShock
AlgorithmSettings, AlgorithmStage
CostSettings
OptimizationRequest, OptimizationCallbacks, OptimizationResult
StartInfo, IterationUpdate, StageComplete

load_mechanism, parse_composition
run_shock_tube, run_zero_d, run_shock_tubes, solve_shock_jump
optimize_residual, apply_optimization_result
kcal_per_mol, kJ_per_mol
```

All are importable from `frhodo` or `frhodo.api`.
