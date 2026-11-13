import casadi as cs
import matplotlib.pyplot as plt
from ocp2fatrop import Ocp2Fatrop


def cart():
    ocp = Ocp2Fatrop(nx=2, nu=1, T0=0, Tf=2)
    t, x, u = ocp.t, ocp.x, ocp.u
    exprs = {}
    exprs["L"] = u**2
    exprs["f"] = cs.vertcat(x[1], -x[1] + u)
    exprs["e0"] = ocp.x0 - cs.DM([0, 0])
    exprs["e0_lb"] = [0, 0]
    exprs["e0_ub"] = [0, 0]
    final_expr = (
        ocp.xf[0]
        - 0.5 * (cs.exp(1) ** 2 - 2) * ocp.xf[1]
        - (-cs.exp(4) + 8 * cs.exp(2) - 23) / 16
    )
    exprs["ef"] = final_expr
    exprs["ef_lb"] = [0]
    exprs["ef_ub"] = [0]
    ocp.set_expressions(exprs)
    N = 100
    ocp.transcribe(N)
    Xg = cs.DM.ones(2, N + 1) * 1.0
    Ug = cs.DM.zeros(1, N)
    sol = ocp.solve(Xg, Ug)
    X = sol.value(ocp._dis_ocp["X"])
    t_grid = sol.value(ocp._dis_ocp["t"])
    U = sol.value(ocp._dis_ocp["U"])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, X[0, :], label="$x_1$")
    plt.plot(t_grid, X[1, :], label="$x_2$")
    plt.title("State Trajectories")
    plt.xlabel("Time ($t$)")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t_grid[0:-1], U.T, label="$u$", color="r")
    plt.title("Control Input ($u$)")
    plt.xlabel("Time ($t$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cart()
