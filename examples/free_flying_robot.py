import casadi as cs
import matplotlib.pyplot as plt

from ocp2fatrop import Ocp2Fatrop, print_timings


def robot():
    ocp = Ocp2Fatrop(nx=6, nu=4, T0=0, Tf=12)
    t, x, u = ocp.t, ocp.x, ocp.u
    exprs = {}
    exprs["L"] = cs.sum1(u)
    exprs["E"] = 0
    alpha = 0.2
    beta = 0.2
    thrust = u[0] - u[1] + u[2] - u[3]
    exprs["f"] = cs.vertcat(
        x[3],  # y_dot_1 = y_4
        x[4],  # y_dot_2 = y_5
        x[5],  # y_dot_3 = y_6
        thrust * cs.cos(x[2]),  # y_dot_4
        thrust * cs.sin(x[2]),  # y_dot_5
        alpha * (u[0] - u[1]) - beta * (u[2] - u[3]),  # y_dot_6
    )
    y_initial = cs.DM([-10, -10, cs.pi / 2, 0, 0, 0])
    exprs["e0"] = ocp.x0 - y_initial
    exprs["e0_lb"] = [0] * 6
    exprs["e0_ub"] = [0] * 6
    y_final = cs.DM.zeros(6, 1)
    exprs["ef"] = ocp.xf - y_final
    exprs["ef_lb"] = [0] * 6
    exprs["ef_ub"] = [0] * 6
    exprs["h"] = cs.vertcat(u, u[0] + u[1], u[2] + u[3])
    inf = float("inf")
    exprs["h_lb"] = [0, 0, 0, 0, -inf, -inf]
    exprs["h_ub"] = [1, 1, 1, 1, 1, 1]
    ocp.set_expressions(exprs)
    N = 1000
    ocp.transcribe(N, method="rk")
    Xg = cs.vertcat(
        cs.linspace(-10, 0, N + 1).T,  # y1 (px)
        cs.linspace(-10, 0, N + 1).T,  # y2 (py)
        cs.linspace(cs.pi / 2, 0, N + 1).T,  # y3 (theta)
        cs.DM.zeros(1, N + 1),  # y4 (vx)
        cs.DM.zeros(1, N + 1),  # y5 (vy)
        cs.DM.zeros(1, N + 1),  # y6 (omega)
    )
    Ug = cs.vertcat(
        cs.DM.ones(1, N) * 0.1,
        cs.DM.ones(1, N) * 0.1,
        cs.DM.ones(1, N) * 0.9,
        cs.DM.ones(1, N) * 0.9,
    )
    ocp.set_solver("fatrop")
    #    ocp.set_solver("uno",{'uno.preset':'ipopt','expand':True})
    #    ocp.set_solver("ipopt",{'expand':True,'equality':ocp.ineqeq})

    sol = ocp.solve(Xg, Ug)

    print_timings()

    X = sol.value(ocp._dis_ocp["X"])
    t_grid = sol.value(ocp._dis_ocp["t"])
    U = sol.value(ocp._dis_ocp["U"])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X[0, :], X[1, :], "-b", lw=2)
    plt.title("Free-Flying Robot Trajectory ($y_1$ vs $y_2$)")
    plt.xlabel("$y_1$ (Position X)")
    plt.ylabel("$y_2$ (Position Y)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t_grid[0:-1], U.T, "-o", ms=3)
    plt.legend([f"u{i + 1}" for i in range(4)])
    plt.title("Control Inputs ($u_1, u_2, u_3, u_4$)")
    plt.xlabel("Time ($t$)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    robot()
