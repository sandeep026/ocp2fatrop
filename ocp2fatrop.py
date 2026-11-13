
from dataclasses import dataclass, field

import casadi as cs


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

    def __post_init__(self):
        self.t = cs.SX.sym("t")
        self.x = cs.SX.sym("x", self.nx)
        self.u = cs.SX.sym("u", self.nu)
        self.t0 = cs.SX.sym("t0")
        self.x0 = cs.SX.sym("x0", self.nx)
        self.tf = cs.SX.sym("tf")
        self.xf = cs.SX.sym("xf", self.nx)
        options = {}
        options["expand"] = True
        options["fatrop"] = {"mu_init": 0.1}
        options["structure_detection"] = "auto"
        options["debug"] = True
        self.nlp.solver("fatrop", options)
        #self.nlp.solver('ipopt')

    @classmethod
    def check_type(cls, names: list["str"], obj, req_type):
        for name in names:
            var = getattr(obj, name)
            if not isinstance(var, cs.SX):
                raise TypeError(
                    f"{name} must be of type SX but received type {type(var)}"
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

        for name in ["L", "f", "h"]:
            if name in exps:
                setattr(
                    self, f'_{name}', cs.Function(name, self.get_path_var(), [exps[name]])
                )

        for name in ["e0", "ef", "E"]:
            if name in exps:
                if name == "e0":
                    setattr(
                        self, f'_{name}', cs.Function(name, self.get_e0_var(), [exps[name]])
                    )
                elif name == "ef":
                    setattr(
                        self, f'_{name}', cs.Function(name, self.get_ef_var(), [exps[name]])
                    )
                else:
                    setattr(
                        self, f'_{name}', cs.Function(name, self.get_event_var(), [exps[name]])
                    )

        for name in ["e0_lb", "ef_lb", "e0_ub", "ef_ub", "h_lb", "h_ub"]:
            if name in exps:
                setattr(self, f'_{name}', cs.DM(exps[name]))

    def transcribe(self, N: int = 50):

        if not isinstance(self._e0,cs.Function):
            raise ValueError("event constraint at t0, e0, e0_lb, e0_ub mandatory")

        if not isinstance(self._f,cs.Function):
            raise ValueError("dynamics f is mandatory")

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
        if isinstance(self._L,cs.Function):
            Ll=self._L.map(N, "serial")(tl, Xl, U).T
            Lr=self._L.map(N, "serial")(tr, Xr, U).T
            lagrange = 0.5*(tr - tl) / N@(Ll+Lr) 
        else:
            lagrange = 0
        if isinstance(self._E,cs.Function):
            mayer = self._E(t0, tf, X0, Xf)
        else:
            mayer = 0
        obj = lagrange + mayer

        # e0
        e0 = self._e0(t0, X0)
        e0_lb = self._e0_lb
        e0_ub = self._e0_ub

        # ef
        if isinstance(self._ef,cs.Function):
            ef = self._ef(tf, Xf)
            ef_lb = self._ef_lb
            ef_ub = self._ef_ub
        else:
            ef = cs.MX()
            ef_lb = cs.MX()
            ef_ub = cs.MX()


        # defect
        D = Xr-self.trap().map(N,'serial')(tl,tr,Xl,U)
        #D = self.rk4().map(N, "serial")(tl, tr, Xl, Xr, U)
        D_lb = cs.DM.zeros(D.shape)
        D_ub = cs.DM.zeros(D.shape)

        # path
        if isinstance(self._h,cs.Function):
            H = self._h.map(N, "serial")(tl, Xl, U)
            H_lb = cs.repmat(self._h_lb, 1, N)
            H_ub = cs.repmat(self._h_ub, 1, N)

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

        if False:
        
           self.nlp.subject_to(self.nlp.bounded(lb,constraint,ub))

        else:

            id_eq=cs.np.where((lb-ub).full()==0)[0].tolist()
            for i in range(lb.numel()):
                if i in id_eq:
                    self.nlp.subject_to(constraint[i]==0)
                else:
                    self.nlp.subject_to(self.nlp.bounded(lb[i],constraint[i],ub[i]))


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
        if  isinstance(self._ef,cs.Function):
            dis_ocp["ef"] = ef
            dis_ocp["ef_lb"] = ef_lb
            dis_ocp["ef_ub"] = ef_ub

        if  isinstance(self._h,cs.Function):
            dis_ocp["h"] = H
            dis_ocp["h_lb"] = H_lb
            dis_ocp["h_ub"] = H_ub

        self._dis_ocp = dis_ocp

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
        defect = xr - xl - 1 * (tr - tl) * (kl + kr)

        p = cs.vertcat(xl, tl, tr, u)
        residual_fun = cs.Function('residual', [xr, p], [defect], ['xr', 'p'], ['defect'])
        xr_solver = cs.rootfinder('xr_solver', 'newton', residual_fun)

        x_guess = xl + (tr-tl) * kl
    
        sol = xr_solver(x_guess, p)
        xr = sol
    
        interval_integrator = cs.Function(
            'shooting_interval',
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
        dt=tr-tl
        k1 = self._f(tl,xl, u)
        k2 = self._f(tl+dt/2,xl + dt / 2 * k1, u)
        k3 = self._f(tl+dt/2,xl + dt / 2 * k2, u)
        k4 = self._f(tr,xl + dt * k3, u)
        defect = xr- xl - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return cs.Function("defect", [tl, tr, xl, xr, u], [defect])
