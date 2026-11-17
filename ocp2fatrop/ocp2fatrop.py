import time
from dataclasses import dataclass, field
from functools import wraps

import casadi as cs

TIMING_LOGS = []


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        TIMING_LOGS.append((func.__qualname__, end - start))
        return result

    return wrapper


def time_all_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        # Time regular functions, ignoring magic methods like __init__
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, timer(attr_value))
    return cls


from collections import defaultdict


def print_timings(clear=True):
    if not TIMING_LOGS:
        return

    stats = defaultdict(lambda: {"calls": 0, "time": 0.0})
    for name, duration in TIMING_LOGS:
        stats[name]["calls"] += 1
        stats[name]["time"] += duration

    total_calls = sum(item["calls"] for item in stats.values())
    total_time = sum(item["time"] for item in stats.values())

    print("\n" + "=" * 70)
    print(f"{'Method':<30} | {'Calls':>5} | {'Time (s)':>10} | {'Avg / Call (s)':>14}")
    print("-" * 70)
    for name, data in stats.items():
        calls = data["calls"]
        duration = data["time"]
        avg_str = f"{duration / calls:.4f}" if calls > 1 else "-"
        print(f"{name:<30} | {calls:>5} | {duration:>10.4f} | {avg_str:>14}")
    print("-" * 70)
    print(f"{'Total':<30} | {total_calls:>5} | {total_time:>10.4f} | {'-':>14}")
    print("=" * 70)

    if clear:
        TIMING_LOGS.clear()


@time_all_methods
@dataclass
class Ocp2Fatrop:
    nx: int
    nu: int
    T0: float = 0
    Tf: float = 1

    # syms
    t: cs.SX = cs.SX()
    x: cs.SX = cs.SX()
    u: cs.SX = cs.SX()
    t0: cs.SX = cs.SX()
    x0: cs.SX = cs.SX()
    tf: cs.SX = cs.SX()
    xf: cs.SX = cs.SX()

    # funcs
    _L: cs.SX = cs.SX()
    _E: cs.SX = cs.SX()
    _f: cs.SX = cs.SX()
    _h: cs.SX = cs.SX()
    _e0: cs.SX = cs.SX()
    _ef: cs.SX = cs.SX()

    # constants
    _e0_lb: cs.DM = cs.DM()
    _e0_ub: cs.DM = cs.DM()
    _ef_lb: cs.DM = cs.DM()
    _ef_ub: cs.DM = cs.DM()
    _h_lb: cs.DM = cs.DM()
    _h_ub: cs.DM = cs.DM()

    nlp: cs.Opti = field(default_factory=cs.Opti)

    _dis_ocp: dict = field(default_factory=dict)

    ineqeq: list = field(default_factory=list)

    @classmethod
    def check_type(cls, names: list[str], obj, req_type):
        for name in names:
            var = getattr(obj, name)
            if not isinstance(var, req_type):
                raise TypeError(
                    f"{name} must be of type {req_type.__name__} but received type {type(var)}"
                )

    @classmethod
    def ge(cls, names: list[str], obj, rhs):
        for name in names:
            var = getattr(obj, name)
            if var < rhs:
                raise ValueError(f"{name} must be > {rhs} but received {var}<{rhs}")

    @classmethod
    def le(cls, names: list[str], obj, rhs):
        for name in names:
            var = getattr(obj, name)
            if var > rhs:
                raise ValueError(f"{name} must be < {rhs} but received {var}>{rhs}")

    @classmethod
    def shape(cls, names: list[str], obj, req_shapes: list[tuple]):
        for name, shape in zip(names, req_shapes):
            var = getattr(obj, name)
            if var.shape != shape:
                raise ValueError(
                    f"{name} must be of shape {shape} but received {var.shape}"
                )

    def __post_init__(self):

        Ocp2Fatrop.check_type(["nx", "nu"], self, int)
        Ocp2Fatrop.check_type(["T0", "Tf"], self, (float, int))
        Ocp2Fatrop.ge(["nx", "nu", "Tf"], self, 1e-16)
        Ocp2Fatrop.ge(["T0"], self, 0)
        Ocp2Fatrop.le(["T0"], self, self.Tf)

        self.t = cs.SX.sym("t")
        self.x = cs.SX.sym("x", self.nx)
        self.u = cs.SX.sym("u", self.nu)
        self.t0 = cs.SX.sym("t0")
        self.x0 = cs.SX.sym("x0", self.nx)
        self.tf = cs.SX.sym("tf")
        self.xf = cs.SX.sym("xf", self.nx)

        Ocp2Fatrop.check_type(["t", "x", "u", "t0", "tf", "x0", "xf"], self, cs.SX)
        Ocp2Fatrop.shape(
            ["t", "x", "u", "t0", "tf", "x0", "xf"],
            self,
            [
                (1, 1),
                (self.nx, 1),
                (self.nu, 1),
                (1, 1),
                (1, 1),
                (self.nx, 1),
                (self.nx, 1),
            ],
        )

    def get_path_var(self):
        return [self.t, self.x, self.u]

    def get_e0_var(self):
        return [self.t0, self.x0]

    def get_ef_var(self):
        return [self.tf, self.xf]

    def get_event_var(self):
        return [self.t0, self.tf, self.x0, self.xf]

    def set_expressions(self, exps: dict):
        self.check_objective(exps)
        self.check_dynamics(exps)
        self.check_bounds(exps)
        absent = []
        for name in ["L", "f", "h"]:
            if name in exps:
                setattr(
                    self,
                    f"_{name}",
                    cs.Function(name, self.get_path_var(), [exps[name]]),
                )
            else:
                absent.append(name)

        for name in ["e0", "ef", "E"]:
            if name in exps:
                if name == "e0":
                    setattr(
                        self,
                        f"_{name}",
                        cs.Function(name, self.get_e0_var(), [exps[name]]),
                    )
                elif name == "ef":
                    setattr(
                        self,
                        f"_{name}",
                        cs.Function(name, self.get_ef_var(), [exps[name]]),
                    )
                elif name == "E":
                    setattr(
                        self,
                        f"_{name}",
                        cs.Function(name, self.get_event_var(), [exps[name]]),
                    )
            else:
                absent.append(name)

        # stores constants (list)  as DM vectors
        for name in ["e0_lb", "ef_lb", "e0_ub", "ef_ub", "h_lb", "h_ub"]:
            if name in exps:
                setattr(self, f"_{name}", cs.DM(exps[name]))
            else:
                absent.append(name)

        # catches if madatory expression not set by user
        for name in absent:
            if name in ["L", "E", "f", "e0", "e0_lb", "e0_ub"]:
                raise ValueError(f"{name} is mandatory and user has not set")

    def check_objective(self, exps):
        # checks shape of scalr objective
        # check type as int float and SX DM
        for name in ["L", "E"]:
            if name in exps:
                if isinstance(exps[name], (cs.SX, cs.DM)) and exps[name].shape != (
                    1,
                    1,
                ):
                    raise ValueError(
                        f"{name} must be scalar but received expression of shape {exps[name].shape}"
                    )
                if not isinstance(exps[name], (int, float, cs.SX, cs.DM)):
                    raise TypeError(
                        f"{name} must be of type int,float or SX but recieved {type(exps[name]).__name__}"
                    )

    def check_dynamics(self, exps):
        # state and dynamics vector dims must match
        if not isinstance(exps["f"], cs.SX):
            raise TypeError(
                f"f must be of type SX but recieved {type(exps['f']).__name__}"
            )
        if exps["f"].shape != (self.nx, 1):
            raise ValueError(
                f"f must match the size of state vector but recieved nx:{self.nx}  and f of shape: {exps['f'].shape}"
            )

    def check_bounds(self, exps: dict):
        # shape
        # type type of elements of bounds
        # element wise bound check
        for name in ["e0", "ef", "h"]:
            if name in exps:
                if not (f"{name}_lb" in exps and f"{name}_ub"):
                    raise ValueError(
                        f"{name} must have a valid bounds {name}_lb and {name}_ub"
                    )
                else:
                    if not isinstance(exps[name], cs.SX):
                        raise TypeError(
                            f"{name} must be SX but recieved {type(exps[name]).__name__}"
                        )

                    if not (
                        isinstance(exps[f"{name}_lb"], list)
                        and isinstance(exps[f"{name}_ub"], list)
                    ):
                        raise TypeError(
                            f"{name}_lb/ub must be of type list[float|int] but received lb :"
                            f"{type(exps[f'{name}_lb']).__name__} and ub :{type(exps[f'{name}_ub']).__name__}"
                        )

                    if not (
                        (exps[f"{name}"]).numel()
                        == len(exps[f"{name}_lb"])
                        == len(exps[f"{name}_ub"])
                    ):
                        raise ValueError(
                            f"Shapes of {name} and its bounds are not same. {name}:{exps[name].shape}, "
                            f"{name}_lb:{(len(exps[f'{name}_lb']), 1)}, {name}_ub:{(len(exps[f'{name}_ub']), 1)}"
                        )

                    for i in range((exps[f"{name}"]).numel()):
                        if not (
                            isinstance(exps[f"{name}_lb"][i], (int, float))
                            and isinstance(exps[f"{name}_ub"][i], (int, float))
                        ):
                            raise TypeError(
                                f"All elements of {name}_lb or {name}_ub must be int/float but received for {name}_lb[{i}]:"
                                f" {type(exps[f'{name}_lb'][i]).__name__}"
                                f" and {name}_ub[{i}]: {type(exps[f'{name}_ub'][i]).__name__}"
                            )

                    if not all(
                        exps[f"{name}_ub"][i] >= exps[f"{name}_lb"][i]
                        for i in range(exps[name].numel())
                    ):
                        raise ValueError(
                            f"The bounds must be ub>lb for {name} but recieved {name}_lb > {name}_ub"
                        )

    def transcribe(self, N: int = 50, method: str = "trap"):

        Ocp2Fatrop.check_type(["_L", "_E"], self, (cs.Function, cs.SX, cs.DM))
        Ocp2Fatrop.check_type(["_e0", "_f"], self, (cs.Function))
        Ocp2Fatrop.check_type(["_h", "_ef"], self, (cs.Function, cs.SX))
        Ocp2Fatrop.check_type(
            ["_e0_lb", "_e0_ub", "_ef_lb", "_ef_ub", "_h_lb", "_h_ub"], self, cs.DM
        )

        W = self.nlp.variable(self.nx + self.nu, N)
        xnp1 = self.nlp.variable(self.nx, 1)
        X = cs.horzcat(W[0 : self.nx, :], xnp1)
        U = W[self.nx : self.nx + self.nu, :]
        t = cs.linspace(self.T0, self.Tf, N + 1).T

        Xl = X[:, 0:-1]
        tl = t[:, 0:-1]
        Xr = X[:, 1:]
        tr = t[:, 1:]
        t0 = t[:, 0]
        tf = t[:, -1]
        X0 = X[:, 0]
        Xf = X[:, -1]

        # objective
        Ll = self._L.map(N, "serial")(tl, Xl, U).T
        Lr = self._L.map(N, "serial")(tr, Xr, U).T
        lagrange = 0.5 * (tr - tl) @ (Ll + Lr)
        mayer = self._E(t0, tf, X0, Xf)
        obj = lagrange + mayer

        # e0
        e0 = self._e0(t0, X0)
        e0_lb = self._e0_lb
        e0_ub = self._e0_ub

        # ef
        if isinstance(self._ef, cs.Function):
            ef = self._ef(tf, Xf)
            ef_lb = self._ef_lb
            ef_ub = self._ef_ub
        else:
            ef = cs.MX()
            ef_lb = cs.MX()
            ef_ub = cs.MX()

        # defect
        if method == "trap":
            D = Xr - self.trap().map(N, "serial")(tl, tr, Xl, U)
        elif method == "rk":
            D = self.rk4().map(N, "serial")(tl, tr, Xl, Xr, U)
        else:
            raise ValueError("Incorrect shooting method")
        D_lb = cs.DM.zeros(D.shape)
        D_ub = cs.DM.zeros(D.shape)

        # path
        if isinstance(self._h, cs.Function):
            H = self._h.map(N, "serial")(tl, Xl, U)
            H_lb = cs.repmat(self._h_lb, 1, N)
            H_ub = cs.repmat(self._h_ub, 1, N)
        else:
            H = cs.MX()
            H_lb = cs.DM()
            H_ub = cs.DM()

        A = cs.vcat([D, H])
        A_lb = cs.vcat([D_lb, H_lb])
        A_ub = cs.vcat([D_ub, H_ub])
        constraint = cs.vertcat(
            cs.vertcat(A[:, 0], e0), cs.vertcat(cs.vec(A[:, 1:-1]), A[:, -1], ef)
        )
        lb = cs.vertcat(
            cs.vertcat(A_lb[:, 0], e0_lb),
            cs.vertcat(cs.vec(A_lb[:, 1:-1]), A_lb[:, -1], ef_lb),
        )
        ub = cs.vertcat(
            cs.vertcat(A_ub[:, 0], e0_ub),
            cs.vertcat(cs.vec(A_ub[:, 1:-1]), A_ub[:, -1], ef_ub),
        )

        self.nlp.minimize(obj)
        self.ineqeq = ((lb - ub).full().flatten() == 0).tolist()
        self.nlp.subject_to(self.nlp.bounded(lb, constraint, ub))

        dis_ocp = {
            "X": X,
            "U": U,
            "t": t,
            "obj": obj,
            "constraint": constraint,
            "lb": lb,
            "ub": ub,
            "lagrange": lagrange,
            "mayer": mayer,
            "e0": e0,
            "e0_lb": e0_lb,
            "e0_ub": e0_ub,
        }
        if isinstance(self._ef, cs.Function):
            dis_ocp["ef"] = ef
            dis_ocp["ef_lb"] = ef_lb
            dis_ocp["ef_ub"] = ef_ub

        if isinstance(self._h, cs.Function):
            dis_ocp["h"] = H
            dis_ocp["h_lb"] = H_lb
            dis_ocp["h_ub"] = H_ub

        self._dis_ocp = dis_ocp

    def set_solver(self, name="fatrop", options: dict | None = None):
        if name == "fatrop" and options is None:
            options = {}
            options["expand"] = True
            options["fatrop"] = {"mu_init": 0.1}
            options["structure_detection"] = "auto"
            options["debug"] = False
            options["equality"] = self.ineqeq
        if options is None:
            options = {}

        self.nlp.solver(name, options)

    def solve(
        self,
        Xg,
        Ug,
    ):
        if Xg.shape != self._dis_ocp["X"].shape:
            raise ValueError("Xg wrong shape")
        if Ug.shape != self._dis_ocp["U"].shape:
            raise ValueError("Ug wrong shape")
        self.nlp.set_initial(self._dis_ocp["X"], Xg)
        self.nlp.set_initial(self._dis_ocp["U"], Ug)
        sol = self.nlp.solve()
        return sol

    def trap(self):
        xl = cs.SX.sym("xl", self.nx, 1)
        xr = cs.SX.sym("xr", self.nx, 1)
        u = cs.SX.sym("u", self.nu, 1)
        tl = cs.SX.sym("tl", 1, 1)
        tr = cs.SX.sym("tr", 1, 1)
        kl = self._f(tl, xl, u)
        kr = self._f(tr, xr, u)
        defect = xr - xl - 0.5 * (tr - tl) * (kl + kr)

        p = cs.vertcat(xl, tl, tr, u)
        residual_fun = cs.Function(
            "residual", [xr, p], [defect], ["xr", "p"], ["defect"]
        )
        xr_solver = cs.rootfinder("xr_solver", "newton", residual_fun)

        x_guess = xl + (tr - tl) * kl

        sol = xr_solver(x_guess, p)
        xr = sol

        interval_integrator = cs.Function(
            "shooting_interval",
            [tl, tr, xl, u],
            [xr],
        )

        return interval_integrator

    def rk4(self):
        xl = cs.SX.sym("xl", self.nx, 1)
        xr = cs.SX.sym("xr", self.nx, 1)
        u = cs.SX.sym("u", self.nu, 1)
        tl = cs.SX.sym("tl", 1, 1)
        tr = cs.SX.sym("tr", 1, 1)
        dt = tr - tl
        k1 = self._f(tl, xl, u)
        k2 = self._f(tl + dt / 2, xl + dt / 2 * k1, u)
        k3 = self._f(tl + dt / 2, xl + dt / 2 * k2, u)
        k4 = self._f(tr, xl + dt * k3, u)
        defect = xr - xl - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return cs.Function("defect", [tl, tr, xl, xr, u], [defect])

    def fw_euler(self):
        xl = cs.SX.sym("xl", self.nx, 1)
        xr = cs.SX.sym("xr", self.nx, 1)
        u = cs.SX.sym("u", self.nu, 1)
        tl = cs.SX.sym("tl", 1, 1)
        tr = cs.SX.sym("tr", 1, 1)
        dt = tr - tl
        k1 = self._f(tl, xl, u)
        defect = xr - xl - dt * (k1)
        return cs.Function("defect", [tl, tr, xl, xr, u], [defect])
