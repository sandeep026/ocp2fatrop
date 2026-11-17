import casadi as cs

from ocp2fatrop import Ocp2Fatrop


# depends on agents and obstacle package
def animate(sol: cs.OptiSol, ocp: Ocp2Fatrop):
    tsol = sol.value(ocp._dis_ocp["t"])
    xsol = sol.value(ocp._dis_ocp["X"][0, :])
    ysol = sol.value(ocp._dis_ocp["X"][1, :])
    thetasol = sol.value(ocp._dis_ocp["X"][2, :])

    time_cfg = Time(t0=tsol[0], tf=tsol[-1], intervals=2 * tsol.size)

    from scipy.interpolate import CubicSpline

    def path(time_cfg):
        tip = np.linspace(time_cfg.t0, time_cfg.tf, time_cfg.intervals + 1)
        x = CubicSpline(tsol, xsol)(tip)
        y = CubicSpline(tsol, ysol)(tip)
        theta = CubicSpline(tsol, thetasol - 3.14 / 2)(tip)
        return x, y, theta

    agent1 = Agent(
        name="car",
        vertices=shapes["car"],
        path=path,
        color="skyblue",
        edge_color="navy",
    )

    def make_circle(radius: float, num_vertices: int = 32) -> list[list[float]]:
        # Start at top (pi/2) and go clockwise (-3*pi/2) to match your roomba winding order
        angles = np.linspace(np.pi / 2, -3 * np.pi / 2, num_vertices, endpoint=False)
        x = np.round(radius * np.cos(angles), 3)
        y = np.round(radius * np.sin(angles), 3)
        return np.column_stack((x, y)).tolist()

    # Add any radii you want dynamically
    for r in [6, 2, 1.5]:
        shapes[f"circle_{r}"] = make_circle(radius=r, num_vertices=32)

    # Instantiate a static obstacle
    obstacle1 = Obstacle(
        name="ob1",
        vertices=shapes["circle_6"],
        center=(2.0, 8.0),
        theta=45 * np.pi / 180,
        color="dimgray",
        edge_color="black",
    )

    # Instantiate a static obstacle
    obstacle2 = Obstacle(
        name="ob2",
        vertices=shapes["circle_2"],
        center=(12.0, 8.0),
        theta=45 * np.pi / 180,
        color="dimgray",
        edge_color="black",
    )

    # Instantiate a static obstacle
    obstacle3 = Obstacle(
        name="ob3",
        vertices=shapes["circle_1.5"],
        center=(22.0, 10.0),
        theta=45 * np.pi / 180,
        color="dimgray",
        edge_color="black",
    )

    ao = AgentsAndObstacles(
        agents=[agent1], obstacles=[obstacle1, obstacle2, obstacle3], time=time_cfg
    )
    ao.visualize(filename="demo")
