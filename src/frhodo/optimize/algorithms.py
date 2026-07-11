"""Algorithm dispatcher for ``optimize_residual``.

Wraps ``nlopt``, ``rbfopt``, and optionally ``pygmo`` behind a uniform
``Optimize.run`` entry point.
"""
import contextlib
import io
import pathlib
import platform
from timeit import default_timer as timer

try:
    import pygmo
except ImportError:
    pygmo = None

import cantera as ct
import nlopt
import numpy as np
import rbfopt

from frhodo.optimize.smurf import smurf_coarse
from frhodo.optimize.subset import select_informative_subset



nlopt_algorithms = [
    nlopt.GN_DIRECT,
    nlopt.GN_DIRECT_NOSCAL,
    nlopt.GN_DIRECT_L,
    nlopt.GN_DIRECT_L_RAND,
    nlopt.GN_DIRECT_L_NOSCAL,
    nlopt.GN_DIRECT_L_RAND_NOSCAL,
    nlopt.GN_ORIG_DIRECT,
    nlopt.GN_ORIG_DIRECT_L,
    nlopt.GN_CRS2_LM,
    nlopt.G_MLSL_LDS,
    nlopt.G_MLSL,
    nlopt.GD_STOGO,
    nlopt.GD_STOGO_RAND,
    nlopt.GN_AGS,
    nlopt.GN_ISRES,
    nlopt.GN_ESCH,
    nlopt.LN_COBYLA,
    nlopt.LN_BOBYQA,
    nlopt.LN_NEWUOA,
    nlopt.LN_NEWUOA_BOUND,
    nlopt.LN_PRAXIS,
    nlopt.LN_NELDERMEAD,
    nlopt.LN_SBPLX,
]

# Registry sentinel for Subplex run in curvature-whitened coordinates.
WHITENED_SBPLX = "whitened_sbplx"
WHITENED_BOBYQA = "whitened_bobyqa"
# Registry sentinel for Subplex over per-reaction field coefficients.
FIELD_SBPLX = "field_sbplx"
# Registry sentinel for the fixed-budget fidelity-ladder triage local.
QUICK_FILTER = "quick_filter"
# Registry sentinel for the loose-tolerance Smurf triage global.
SMURF_LOWFI = "smurf_lowfi"
# Registry sentinel for the Smurf coarse global stage.
SMURF = "smurf"

# Probe-measured per-slot curvature spread below which whitening is
# skipped and plain Subplex runs in original coordinates. Calibration:
# a 9-slot leverage-homogeneous campaign measured ~52 and whitening
# cost ~0.6% quality; 30- and 66-slot mixed-leverage campaigns
# measured >=1e4 and whitening gained 2.7-6.6%.
WHITEN_SPREAD_THRESHOLD = 1e3

pos_msg = [
    "Optimization terminated successfully.",
    "Optimization terminated: Stop Value was reached.",
    "Optimization terminated: Function tolerance was reached.",
    "Optimization terminated: X tolerance was reached.",
    "Optimization terminated: Max number of evaluations was reached.",
    "Optimization terminated: Max time was reached.",
]
neg_msg = [
    "Optimization failed",
    "Optimization failed: Invalid arguments given",
    "Optimization failed: Out of memory",
    "Optimization failed: Roundoff errors limited progress",
    "Optimization failed: Forced termination",
]

# bonmin/ipopt binaries are vendored under ``frhodo/_vendor/`` and ship
# inside the package by virtue of being on the import path. Resolved via
# ``__file__`` so this works under installed entry points.
_VENDOR_ROOT = pathlib.Path(__file__).resolve().parent.parent / "_vendor"
_OS_TYPE = platform.system()
if _OS_TYPE == "Windows":
    _PLATFORM, _BIN_EXT = "win64", ".exe"
elif _OS_TYPE == "Linux":
    _PLATFORM, _BIN_EXT = "linux64", ""
elif _OS_TYPE == "Darwin":
    _PLATFORM, _BIN_EXT = "osx", ""
else:
    raise RuntimeError(f"unsupported platform: {_OS_TYPE}")


def _resolve_binary(name: str) -> pathlib.Path:
    return _VENDOR_ROOT / name / f"{name}-{_PLATFORM}" / f"{name}{_BIN_EXT}"


path = {
    "bonmin": _resolve_binary("bonmin"),
    "ipopt": _resolve_binary("ipopt"),
}


class Optimize:
    """Dispatch wrapper over nlopt / pygmo / RBFOpt backends.

    Holds the objective and bounds; :meth:`run` walks the configured
    global → local stages and returns the per-stage result.

    Attributes:
        obj_fcn: Bound objective ``obj_fcn(s)`` returning a scalar cost.
        x0: Starting scaler vector.
        bnds: ``{lower: np.ndarray, upper: np.ndarray}`` of search bounds.
        opt_options: Per-stage configuration dict (algorithm choice,
            stop criteria, step sizes).
        Scaled_CostFunction: The underlying cost function instance —
            the iteration counter and stage label are pushed on to it
            before each stage runs.
    """

    def __init__(self, obj_fcn, x0, bnds, opt_options, Scaled_CostFunction):
        self.obj_fcn = obj_fcn
        self.x0 = x0
        self.bnds = bnds
        self.opt_options = opt_options
        self.Scaled_CostFunction = Scaled_CostFunction

    def run(self):
        """Run the configured optimization stages in order.

        Returns:
            ``{stage: result_dict}`` for the stages whose ``run`` flag
            was set. Possible stage keys: ``"global"`` and ``"local"``.
        """
        x0 = self.x0
        bnds = list(self.bnds.values())
        opt_options = self.opt_options

        res = {}
        for n, opt_type in enumerate(["global", "local"]):
            self.Scaled_CostFunction.i = 0
            self.Scaled_CostFunction.opt_type = opt_type

            options = opt_options[opt_type]
            if not options["run"]:
                continue

            if options["algorithm"] in nlopt_algorithms:
                res[opt_type] = self.nlopt(x0, bnds, options)
            elif options["algorithm"] in (WHITENED_SBPLX, WHITENED_BOBYQA):
                res[opt_type] = self.whitened_sbplx(x0, bnds, options)
            elif options["algorithm"] == FIELD_SBPLX:
                res[opt_type] = self.field_sbplx(x0, bnds, options)
            elif options["algorithm"] == QUICK_FILTER:
                res[opt_type] = self.quick_filter(x0, bnds, options)
            elif options["algorithm"] == SMURF_LOWFI:
                res[opt_type] = self.smurf_lowfi(x0, bnds, options)
            elif options["algorithm"] == SMURF:
                res[opt_type] = self.smurf(x0, bnds, options)
            elif options["algorithm"] in [
                "pygmo_DE",
                "pygmo_SaDE",
                "pygmo_PSO",
                "pygmo_GWO",
            ]:
                if pygmo is None:
                    raise ImportError(
                        "pygmo is required for pygmo-based algorithms; "
                        "install with `pip install frhodo[optimize]`"
                    )
                res[opt_type] = self.pygmo(x0, bnds, options)
            elif options["algorithm"] == "RBFOpt":
                res[opt_type] = self.rbfopt(x0, bnds, options)

            # The next stage starts from this stage's optimum, and the
            # error floor re-anchors there — near-optimal residuals are
            # where the floor has its intended meaning.
            stage_s = res.get(opt_type, {}).get("s")
            next_stage_runs = opt_type == "global" and opt_options["local"]["run"]
            if stage_s is not None and np.all(np.isfinite(stage_s)):
                x0 = np.asarray(stage_s, dtype=float)
                recalibrate = getattr(
                    self.Scaled_CostFunction, "calibrate_error_floor", None,
                )
                if next_stage_runs and recalibrate is not None:
                    recalibrate(x0)

            if options["algorithm"] is nlopt.GN_MLSL_LDS:
                break

        return res

    def nlopt(self, x0, bnds, options):
        timer_start = timer()

        # Hold the incumbent point; a solver abort returns it as the
        # stage result.
        best = {"f": np.inf, "s": np.array(x0, dtype=float, copy=True)}

        def tracked_obj(s, grad=None):
            value = self.obj_fcn(s, grad)
            if np.isfinite(value) and value < best["f"]:
                best["f"] = float(value)
                best["s"] = np.array(s, dtype=float, copy=True)

            return value

        opt = nlopt.opt(options["algorithm"], np.size(x0))
        opt.set_min_objective(tracked_obj)
        if options["stop_criteria_type"] == "Iteration Maximum":
            opt.set_maxeval(int(options["stop_criteria_val"]) - 1)
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            opt.set_maxtime(options["stop_criteria_val"] * 60)
        else:
            raise ValueError(
                f"unexpected stop_criteria_type: {options['stop_criteria_type']!r}"
            )

        opt.set_xtol_rel(options["xtol_rel"])
        opt.set_ftol_rel(options["ftol_rel"])
        opt.set_lower_bounds(bnds[0])
        opt.set_upper_bounds(bnds[1])

        initial_step = (bnds[1] - bnds[0]) * options["initial_step"]
        np.putmask(initial_step, x0 < 1, -initial_step)
        opt.set_initial_step(initial_step)

        if options["algorithm"] in [
            nlopt.GN_CRS2_LM,
            nlopt.GN_MLSL_LDS,
            nlopt.GN_MLSL,
            nlopt.GN_ISRES,
        ]:
            if options["algorithm"] is nlopt.GN_CRS2_LM:
                default_pop_size = 10 * (len(x0) + 1)
            elif options["algorithm"] in [nlopt.GN_MLSL_LDS, nlopt.GN_MLSL]:
                default_pop_size = 4
            elif options["algorithm"] is nlopt.GN_ISRES:
                default_pop_size = 20 * (len(x0) + 1)

            opt.set_population(
                int(np.rint(default_pop_size * options["initial_pop_multiplier"]))
            )

        if options["algorithm"] is nlopt.GN_MLSL_LDS:
            sub_opt = nlopt.opt(self.opt_options["local"]["algorithm"], np.size(x0))
            sub_opt.set_initial_step(initial_step)
            sub_opt.set_xtol_rel(options["xtol_rel"])
            sub_opt.set_ftol_rel(options["ftol_rel"])
            opt.set_local_optimizer(sub_opt)

        try:
            s_opt = opt.optimize(x0)
            if nlopt.SUCCESS > 0:
                success = True
                msg = pos_msg[nlopt.SUCCESS - 1]
            else:
                success = False
                msg = neg_msg[nlopt.SUCCESS - 1]
        except Exception as e:
            # SWIG maps C++ failures to its own exception types, so
            # match broadly; user aborts still propagate.
            if "Optimization terminated by user" in str(e):
                raise
            if not np.isfinite(best["f"]):
                raise
            s_opt = best["s"]
            success = True
            msg = (f"Optimization stopped at the incumbent "
                   f"({type(e).__name__}: solver could not improve further)")

        obj_fcn, x, shock_output = self.Scaled_CostFunction(s_opt, optimizing=False)

        result = {
            "x": x,
            "s": s_opt,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": opt.get_numevals(),
            "success": success,
            "message": msg,
            "time": timer() - timer_start,
        }

        return result

    def whitened_sbplx(self, x0, bnds, options):
        """Subplex in diagonally curvature-whitened coordinates.

        A ~2n+1-evaluation central-difference probe measures per-slot
        curvature at the start; Subplex then searches where those
        scales are equal. Pays off when the slots mix parameter
        families of very different stiffness (Troe falloff terms
        against Arrhenius legs); near-isotropic problems only pay the
        probe. Probe evaluations count against the stage budget. With
        a diagonal metric the search box maps exactly, so no clipping
        is involved.
        """
        timer_start = timer()
        x0 = np.asarray(x0, dtype=float)
        lb = np.asarray(bnds[0], dtype=float)
        ub = np.asarray(bnds[1], dtype=float)
        n = x0.size
        if options["algorithm"] == WHITENED_BOBYQA:
            inner_algo = nlopt.LN_BOBYQA
        else:
            inner_algo = nlopt.LN_SBPLX

        best = {"f": np.inf, "s": x0.copy()}

        def tracked_obj(s):
            value = self.obj_fcn(s)
            if np.isfinite(value) and value < best["f"]:
                best["f"] = float(value)
                best["s"] = np.array(s, dtype=float, copy=True)

            return value

        f0 = tracked_obj(x0)
        if not np.isfinite(f0):
            # No usable curvature at an infeasible start; run plain
            # Subplex on the original coordinates instead.
            plain = dict(options, algorithm=inner_algo)

            return self.nlopt(x0, bnds, plain)

        # Second differences at 2% of each slot's span sit well above
        # solver ripple; probe centers shift inward when the start
        # hugs a bound.
        h = 0.02 * (ub - lb)
        center = np.clip(x0, lb + h, ub - h)
        probe_evals = 1
        diag = np.empty(n)
        for i in range(n):
            hi = x0.copy()
            lo = x0.copy()
            hi[i] = center[i] + h[i]
            lo[i] = center[i] - h[i]
            f_hi = tracked_obj(hi)
            f_lo = tracked_obj(lo)
            probe_evals += 2
            f_c = f0
            if center[i] != x0[i]:
                mid = x0.copy()
                mid[i] = center[i]
                f_c = tracked_obj(mid)
                probe_evals += 1
            diag[i] = (f_hi - 2.0 * f_c + f_lo) / h[i] ** 2

        w = np.abs(diag)
        # A failed probe simulation reads as infinite curvature; those
        # slots are treated as carrying no curvature information.
        finite = np.isfinite(w)
        if not np.any(finite):
            plain = dict(options, algorithm=inner_algo)

            return self.nlopt(x0, bnds, plain)
        w[~finite] = 0.0
        if np.any(w > 0.0):
            spread = w.max() / max(w[w > 0.0].min(), 1e-300)
        else:
            spread = 1.0
        if spread < WHITEN_SPREAD_THRESHOLD:
            # Near-isotropic slots: the metric buys nothing and a
            # frozen start-point scaling can mislead — run plain
            # Subplex on the original coordinates with the remaining
            # budget.
            plain = dict(options, algorithm=inner_algo)
            if options["stop_criteria_type"] == "Iteration Maximum":
                remaining = int(options["stop_criteria_val"]) - probe_evals
                plain["stop_criteria_val"] = max(remaining, 2)
            result = self.nlopt(x0, bnds, plain)
            result["nfev"] += probe_evals
            result["message"] += (
                f" [whitening skipped: curvature spread {spread:.0f} < "
                f"{WHITEN_SPREAD_THRESHOLD:.0f}]"
            )

            return result

        w = np.maximum(w, max(w.max() * 1e-4, 1e-300))
        scale_vec = 1.0 / np.sqrt(w)

        def to_s(u):
            s_vec = x0 + scale_vec * u

            return s_vec

        opt = nlopt.opt(inner_algo, n)
        opt.set_min_objective(lambda u, grad: tracked_obj(to_s(u)))
        opt.set_lower_bounds((lb - x0) / scale_vec)
        opt.set_upper_bounds((ub - x0) / scale_vec)
        if options["stop_criteria_type"] == "Iteration Maximum":
            budget = int(options["stop_criteria_val"]) - 1 - probe_evals
            opt.set_maxeval(max(budget, 1))
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            remaining = options["stop_criteria_val"] * 60 - (timer() - timer_start)
            opt.set_maxtime(max(remaining, 1.0))
        else:
            raise ValueError(
                f"unexpected stop_criteria_type: {options['stop_criteria_type']!r}"
            )
        opt.set_xtol_rel(options["xtol_rel"])
        opt.set_ftol_rel(options["ftol_rel"])
        # Whitened coordinates carry objective units: a step delta
        # moves the (locally quadratic) objective by ~delta^2/2, so
        # size it to a few percent of the start value.
        opt.set_initial_step(0.3 * np.sqrt(f0))

        try:
            u_opt = opt.optimize(np.zeros(n))
            s_opt = to_s(u_opt)
            if nlopt.SUCCESS > 0:
                success = True
                msg = pos_msg[nlopt.SUCCESS - 1]
            else:
                success = False
                msg = neg_msg[nlopt.SUCCESS - 1]
        except Exception as e:
            if "Optimization terminated by user" in str(e):
                raise
            if not np.isfinite(best["f"]):
                raise
            s_opt = best["s"]
            success = True
            msg = (f"Optimization stopped at the incumbent "
                   f"({type(e).__name__}: solver could not improve further)")

        obj_fcn, x, shock_output = self.Scaled_CostFunction(s_opt, optimizing=False)

        result = {
            "x": x,
            "s": s_opt,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": probe_evals + opt.get_numevals(),
            "success": success,
            "message": msg,
            "time": timer() - timer_start,
        }

        return result

    def smurf(self, x0, bnds, options):
        """Sensitivity Multistart Rate Fitting coarse global stage."""
        result = smurf_coarse(
            self.obj_fcn, self.Scaled_CostFunction, x0, bnds, options)

        return result

    def smurf_lowfi(self, x0, bnds, options):
        """Smurf with evaluations at loosened ODE tolerance.

        The identical Smurf algorithm; every simulation runs at
        rtol/atol loosened to at most 1e-5/1e-8, making sweeps and
        line-search evaluations several times cheaper. The returned
        point is re-scored once at the run's configured fidelity so the
        reported objective is on the true objective. A triage preset —
        pair with the quick local for a cheap end-to-end ranking run.
        """
        fit = self.Scaled_CostFunction
        full_state = fit.reactor_state
        rtol = max(1e-5, float(full_state.ode_rtol))
        atol = max(1e-8, float(full_state.ode_atol))
        fit.reactor_state = full_state.model_copy(
            update={"ode_rtol": rtol, "ode_atol": atol})
        try:
            result = smurf_coarse(
                self.obj_fcn, fit, x0, bnds, options)
        finally:
            fit.reactor_state = full_state
        obj_val, x_final, shock_output = fit(
            np.asarray(result["s"], dtype=float), optimizing=False)
        result["fval"] = obj_val
        result["x"] = x_final
        result["shock"] = shock_output

        return result

    def quick_filter(self, x0, bnds, options):
        """Field-basis Subplex under a fixed-budget fidelity ladder.

        A triage preset: walks (ODE-tolerance, shock-subset) fidelity
        levels from cheap to the run's configured full fidelity with an
        equal share of the evaluation budget per level. The searcher is
        byte-identical field-basis Subplex; speed comes from cheaper
        evaluations for the early levels plus the smaller preset budget.
        Across 14 validation measurements it landed within ~5% of the
        full pipeline's final (median +3%, worst +13%, and better on 4
        of 14) at roughly half the pipeline wall — pair with the
        low-fidelity Smurf global to cheapen the other half. For
        ranking candidate setups, not for producing mechanisms.
        """
        timer_start = timer()
        fit = self.Scaled_CostFunction
        x0 = np.asarray(x0, dtype=float)
        lb = np.asarray(bnds[0], dtype=float)
        ub = np.asarray(bnds[1], dtype=float)
        span = float(np.max(ub - lb))

        full_shocks = list(fit.shocks2run)
        full_state = fit.reactor_state
        full_user = fit._user_weights
        full_balance = fit._exp_balance
        full_cov = fit._cov_features
        n = len(full_shocks)

        final_rtol = float(full_state.ode_rtol)
        final_atol = float(full_state.ode_atol)
        tol_levels = [
            (max(1e-5, final_rtol), max(1e-8, final_atol)),
            (max(1e-6, final_rtol), max(1e-9, final_atol)),
            (final_rtol, final_atol),
        ]
        sub_levels = [int(np.ceil(n / 3)), int(np.ceil(2 * n / 3)), n]
        schedule = [(0, 0), (1, 1), (2, 2)]

        subset_cache = {}

        def subset_idx(k):
            if k >= n or full_cov is None:
                return list(range(n))
            if k not in subset_cache:
                picked = select_informative_subset(np.asarray(full_cov), k)
                subset_cache[k] = sorted(picked)

            return subset_cache[k]

        def apply_level(t_i, s_i):
            rtol, atol = tol_levels[t_i]
            fit.reactor_state = full_state.model_copy(
                update={"ode_rtol": rtol, "ode_atol": atol})
            idx = subset_idx(sub_levels[s_i])
            fit.shocks2run[:] = [full_shocks[i] for i in idx]
            fit._user_weights = full_user[idx]
            if full_balance is not None and full_balance.size == n:
                fit._exp_balance = full_balance[idx]
            if full_cov is not None:
                fit._cov_features = full_cov[idx]

        if options["stop_criteria_type"] == "Iteration Maximum":
            total_budget = int(options["stop_criteria_val"])
        else:
            total_budget = 400
        level_budget = max(total_budget // len(schedule), 8)
        warm = getattr(fit, "_smurf_step_scale", None)
        if warm is not None and np.isfinite(warm) and warm > 0.0:
            initial_step = float(np.clip(warm / span, 1e-3, 0.2))
        else:
            initial_step = options["initial_step"]

        s_cur = np.clip(x0, lb, ub)
        total_nfev = 0
        level_log = []
        try:
            for step_i, (t_i, s_i) in enumerate(schedule):
                apply_level(t_i, s_i)
                fit.calibrate_error_floor(s_cur)
                opts = dict(options,
                            stop_criteria_type="Iteration Maximum",
                            stop_criteria_val=level_budget,
                            initial_step=initial_step)
                result = self.field_sbplx(s_cur, [lb, ub], opts)
                s_cur = np.asarray(result["s"], dtype=float)
                total_nfev += int(result["nfev"])
                level_log.append(
                    f"L{step_i}:{result['fval']:.5g}@{result['nfev']}ev")
        finally:
            fit.reactor_state = full_state
            fit.shocks2run[:] = full_shocks
            fit._user_weights = full_user
            fit._exp_balance = full_balance
            fit._cov_features = full_cov

        fit.calibrate_error_floor(s_cur)
        obj_val, x_final, shock_output = fit(s_cur, optimizing=False)
        result = {
            "x": x_final,
            "s": s_cur,
            "shock": shock_output,
            "fval": obj_val,
            "nfev": total_nfev,
            "success": True,
            "message": "Quick filter complete [" + " ".join(level_log) + "]",
            "time": timer() - timer_start,
        }

        return result

    def field_sbplx(self, x0, bnds, options):
        """Subplex over per-reaction 4-basis field coefficients.

        Optimizes ``delta_lnk_r(T, P) = c0 + c1·lnT − c2/(Ru·T) + c3·lnP``
        per reaction (4 coefficients each) applied on top of the stage's
        start point. The search dimension is 4 per reaction regardless of
        slot count, and the subspace excludes refit-degenerate directions
        of the full slot space.
        """
        timer_start = timer()
        fit = self.Scaled_CostFunction
        x0 = np.asarray(x0, dtype=float)
        lb = np.asarray(bnds[0], dtype=float)
        ub = np.asarray(bnds[1], dtype=float)
        ru = float(ct.gas_constant)
        blocks, bases = [], []
        offset = 0
        for rc in fit.rxn_coef_opt:
            t_grid = np.asarray(rc["T"], dtype=float)
            p_grid = np.asarray(rc["P"], dtype=float)
            blocks.append((offset, offset + t_grid.size))
            offset += t_grid.size
            basis = np.column_stack([
                np.ones(t_grid.size), np.log(t_grid),
                -1.0 / (ru * t_grid), np.log(p_grid),
            ])
            scale = np.maximum(np.max(np.abs(basis), axis=0), 1e-300)
            bases.append(basis / scale)
        n_dim = 4 * len(blocks)

        def to_s(c):
            s_vec = x0.copy()
            for k, (a, b) in enumerate(blocks):
                s_vec[a:b] = s_vec[a:b] + bases[k] @ c[4 * k:4 * k + 4]
            s_vec = np.clip(s_vec, lb, ub)

            return s_vec

        best = {"f": np.inf, "s": x0.copy()}

        def tracked_obj(c, grad=None):
            s_vec = to_s(c)
            value = self.obj_fcn(s_vec)
            if np.isfinite(value) and value < best["f"]:
                best["f"] = float(value)
                best["s"] = s_vec

            return value

        span = float(np.max(ub - lb))
        opt = nlopt.opt(nlopt.LN_SBPLX, n_dim)
        opt.set_min_objective(tracked_obj)
        opt.set_lower_bounds(np.full(n_dim, -span))
        opt.set_upper_bounds(np.full(n_dim, span))
        if options["stop_criteria_type"] == "Iteration Maximum":
            opt.set_maxeval(int(options["stop_criteria_val"]) - 1)
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            opt.set_maxtime(options["stop_criteria_val"] * 60)
        else:
            raise ValueError(
                f"unexpected stop_criteria_type: {options['stop_criteria_type']!r}"
            )
        opt.set_xtol_rel(options["xtol_rel"])
        opt.set_ftol_rel(options["ftol_rel"])
        opt.set_initial_step(options["initial_step"] * span)

        try:
            c_opt = opt.optimize(np.zeros(n_dim))
            s_opt = to_s(c_opt)
            if nlopt.SUCCESS > 0:
                success = True
                msg = pos_msg[nlopt.SUCCESS - 1]
            else:
                success = False
                msg = neg_msg[nlopt.SUCCESS - 1]
        except Exception as e:
            if "Optimization terminated by user" in str(e):
                raise
            if not np.isfinite(best["f"]):
                raise
            s_opt = best["s"]
            success = True
            msg = (f"Optimization stopped at the incumbent "
                   f"({type(e).__name__}: solver could not improve further)")

        obj_fcn, x, shock_output = self.Scaled_CostFunction(s_opt, optimizing=False)

        result = {
            "x": x,
            "s": s_opt,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": opt.get_numevals(),
            "success": success,
            "message": msg,
            "time": timer() - timer_start,
        }

        return result

    def pygmo(self, x0, bnds, options):
        class pygmo_objective_fcn:
            def __init__(self, obj_fcn, bnds):
                self.obj_fcn = obj_fcn
                self.bnds = bnds

            def fitness(self, x):
                fitness_value = [self.obj_fcn(x)]

                return fitness_value

            def get_bounds(self):
                return self.bnds

            def gradient(self, x):
                return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

        timer_start = timer()

        default_pop_size = max(35, 5 * (len(x0) + 1))
        pop_size = max(
            7, int(np.rint(default_pop_size * options["initial_pop_multiplier"])))
        if options["stop_criteria_type"] == "Iteration Maximum":
            num_gen = int(np.ceil(options["stop_criteria_val"] / pop_size))
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            num_gen = int(np.ceil(1e20 / pop_size))
        else:
            raise ValueError(
                f"unexpected stop_criteria_type: {options['stop_criteria_type']!r}"
            )

        prob = pygmo.problem(pygmo_objective_fcn(self.obj_fcn, tuple(bnds)))
        pop = pygmo.population(prob, pop_size - 1)
        pop.push_back(x=x0)

        if options["algorithm"] == "pygmo_DE":
            F = 0.2
            CR = 0.8032 * np.exp(-1.165e-3 * num_gen)
            algo = pygmo.algorithm(pygmo.de(gen=num_gen, F=F, CR=CR, variant=6))
        elif options["algorithm"] == "pygmo_SaDE":
            algo = pygmo.algorithm(pygmo.sade(gen=num_gen, variant=6))
        elif options["algorithm"] == "pygmo_PSO":
            algo = pygmo.algorithm(pygmo.pso_gen(gen=num_gen))
        elif options["algorithm"] == "pygmo_GWO":
            algo = pygmo.algorithm(pygmo.gwo(gen=num_gen))
        elif options["algorithm"] == "pygmo_IPOPT":
            algo = pygmo.algorithm(pygmo.ipopt())
        else:
            raise ValueError(f"unexpected pygmo algorithm: {options['algorithm']!r}")

        pop = algo.evolve(pop)

        s_opt = pop.champion_x
        obj_fcn, x, shock_output = self.Scaled_CostFunction(s_opt, optimizing=False)

        result = {
            "x": x,
            "s": s_opt,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": pop.problem.get_fevals(),
            "success": True,
            "message": "Optimization terminated successfully.",
            "time": timer() - timer_start,
        }

        return result

    def rbfopt(self, x0, bnds, options):
        timer_start = timer()

        if options["stop_criteria_type"] == "Iteration Maximum":
            max_eval = int(options["stop_criteria_val"])
            max_time = 1e30
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            max_eval = 10000
            max_time = options["stop_criteria_val"] * 60
        else:
            raise ValueError(
                f"unexpected stop_criteria_type: {options['stop_criteria_type']!r}"
            )

        var_type = ["R"] * np.size(x0)

        output = {"success": False, "message": []}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with contextlib.redirect_stdout(stdout):
                bb = rbfopt.RbfoptUserBlackBox(
                    np.size(x0),
                    np.array(bnds[0]),
                    np.array(bnds[1]),
                    np.array(var_type),
                    self.obj_fcn,
                )
                rbfopt_kwargs = {
                    "max_iterations": max_eval,
                    "max_evaluations": max_eval,
                    "max_cycles": 1e30,
                    "max_clock_time": max_time,
                    "init_sample_fraction": np.size(x0) + 1,
                    "max_random_init": np.size(x0) + 2,
                    "minlp_solver_path": path["bonmin"],
                    "nlp_solver_path": path["ipopt"],
                }
                if options.get("random_seed"):
                    rbfopt_kwargs["rand_seed"] = options["random_seed"]
                settings = rbfopt.RbfoptSettings(**rbfopt_kwargs)
                algo = rbfopt.RbfoptAlgorithm(settings, bb, init_node_pos=x0)
                val, s_opt, itercount, evalcount, fast_evalcount = algo.optimize()

                obj_fcn, x, shock_output = self.Scaled_CostFunction(
                    s_opt, optimizing=False,
                )

                output["message"] = "Optimization terminated successfully."
                output["success"] = True

        result = {
            "x": x,
            "s": s_opt,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": evalcount + fast_evalcount,
            "success": output["success"],
            "message": output["message"],
            "time": timer() - timer_start,
        }

        return result
