"""Algorithm dispatcher for ``optimize_residual``.

Wraps ``nlopt``, ``rbfopt``, and optionally ``pygmo`` behind a uniform
``Optimize.run`` entry point.
"""
import contextlib
import io
import pathlib
import platform
from timeit import default_timer as timer

import nlopt
import numpy as np
import rbfopt

try:
    import pygmo
except ImportError:
    pygmo = None


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
    nlopt.LD_MMA,
    nlopt.LD_CCSAQ,
    nlopt.LD_SLSQP,
    nlopt.LD_LBFGS,
    nlopt.LD_TNEWTON,
    nlopt.LD_TNEWTON_PRECOND,
    nlopt.LD_TNEWTON_RESTART,
    nlopt.LD_TNEWTON_PRECOND_RESTART,
    nlopt.LD_VAR1,
    nlopt.LD_VAR2,
]

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

            if options["algorithm"] is nlopt.GN_MLSL_LDS:
                break

        return res

    def nlopt(self, x0, bnds, options):
        timer_start = timer()

        opt = nlopt.opt(options["algorithm"], np.size(x0))
        opt.set_min_objective(self.obj_fcn)
        if options["stop_criteria_type"] == "Iteration Maximum":
            opt.set_maxeval(int(options["stop_criteria_val"]) - 1)
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            opt.set_maxtime(options["stop_criteria_val"] * 60)

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

        x = opt.optimize(x0)

        obj_fcn, x, shock_output = self.Scaled_CostFunction(x, optimizing=False)

        if nlopt.SUCCESS > 0:
            success = True
            msg = pos_msg[nlopt.SUCCESS - 1]
        else:
            success = False
            msg = neg_msg[nlopt.SUCCESS - 1]

        return {
            "x": x,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": opt.get_numevals(),
            "success": success,
            "message": msg,
            "time": timer() - timer_start,
        }

    def pygmo(self, x0, bnds, options):
        class pygmo_objective_fcn:
            def __init__(self, obj_fcn, bnds):
                self.obj_fcn = obj_fcn
                self.bnds = bnds

            def fitness(self, x):
                return [self.obj_fcn(x)]

            def get_bounds(self):
                return self.bnds

            def gradient(self, x):
                return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

        timer_start = timer()

        pop_size = int(np.max([35, 5 * (len(x0) + 1)]))
        if options["stop_criteria_type"] == "Iteration Maximum":
            num_gen = int(np.ceil(options["stop_criteria_val"] / pop_size))
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            num_gen = int(np.ceil(1e20 / pop_size))

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

        pop = algo.evolve(pop)

        x = pop.champion_x
        obj_fcn, x, shock_output = self.Scaled_CostFunction(x, optimizing=False)

        return {
            "x": x,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": pop.problem.get_fevals(),
            "success": True,
            "message": "Optimization terminated successfully.",
            "time": timer() - timer_start,
        }

    def rbfopt(self, x0, bnds, options):
        timer_start = timer()

        if options["stop_criteria_type"] == "Iteration Maximum":
            max_eval = int(options["stop_criteria_val"])
            max_time = 1e30
        elif options["stop_criteria_type"] == "Maximum Time [min]":
            max_eval = 10000
            max_time = options["stop_criteria_val"] * 60

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
                settings = rbfopt.RbfoptSettings(
                    max_iterations=max_eval,
                    max_evaluations=max_eval,
                    max_cycles=1e30,
                    max_clock_time=max_time,
                    init_sample_fraction=np.size(x0) + 1,
                    max_random_init=np.size(x0) + 2,
                    minlp_solver_path=path["bonmin"],
                    nlp_solver_path=path["ipopt"],
                )
                algo = rbfopt.RbfoptAlgorithm(settings, bb, init_node_pos=x0)
                val, x, itercount, evalcount, fast_evalcount = algo.optimize()

                obj_fcn, x, shock_output = self.Scaled_CostFunction(x, optimizing=False)

                output["message"] = "Optimization terminated successfully."
                output["success"] = True

        return {
            "x": x,
            "shock": shock_output,
            "fval": obj_fcn,
            "nfev": evalcount + fast_evalcount,
            "success": output["success"],
            "message": output["message"],
            "time": timer() - timer_start,
        }
