import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from ocp2fatrop import Ocp2Fatrop

def get_interpolants(T0=0,Tf=5,N=30):
    # reference state

    waypoints = np.array([
        (0, 0),     # Start
        (5, 3),     # Waypoint 1
        (10, 8),    # Waypoint 2
        (15, 10),   # Waypoint 3
        (20, 12),   # Waypoint 4
        (25, 8),    # Waypoint 5
    ])
    
    distances = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        distances[i] = distances[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])    
    spline_x = CubicSpline(distances, waypoints[:, 0])
    spline_y = CubicSpline(distances, waypoints[:, 1])
    total_length = distances[-1]
    desired_speed = 5.0
    time_grid = np.linspace(T0, Tf, N + 1)
    path_distances = (desired_speed * time_grid) % total_length
    x_ref_vals = spline_x(path_distances)
    y_ref_vals = spline_y(path_distances)
    dx_ds = spline_x.derivative()(path_distances)
    dy_ds = spline_y.derivative()(path_distances)
    theta_ref_vals = np.arctan2(dy_ds, dx_ds)
    v_ref_vals = np.full(N + 1, desired_speed)
    interp_x = cs.interpolant('x_ref', 'linear', [time_grid], x_ref_vals)
    interp_y = cs.interpolant('y_ref', 'linear', [time_grid], y_ref_vals)
    interp_theta = cs.interpolant('theta_ref', 'linear', [time_grid], theta_ref_vals)
    interp_v = cs.interpolant('v_ref', 'linear', [time_grid], v_ref_vals)
    return interp_x,interp_y,interp_theta,interp_v

def run_car_ocp_fatrop():
    # taken from https://github.com/meco-group/fatrop/blob/main/examples/ocp/car_reference_tracking_example.py

    T0 = 0.0
    Tf = 5.0
    N = 50

    interp_x,interp_y,interp_theta,interp_v=get_interpolants(T0=T0,Tf=Tf)

    ocp = Ocp2Fatrop(nx=4, nu=2, T0=T0, Tf=Tf)
    t, x, u = ocp.t, ocp.x, ocp.u    
    exprs = {}
    xr = interp_x(t)
    yr = interp_y(t)
    thetar = interp_theta(t)
    vr = interp_v(t)   
    pos_err = (x[0] - xr)**2 + (x[1] - yr)**2
    heading_err = cs.sin(x[2] - thetar)**2 + (1 - cs.cos(x[2] - thetar))**2
    v_err = (x[3] - vr)**2    
    Q_pos = 10.0
    Q_heading = 10.0
    Q_velocity = 0.1
    R_acc = 1.0
    R_steer = 0.0
    lag=(Q_pos * pos_err + 
                  Q_heading * heading_err + 
                  Q_velocity * v_err + 
                  R_acc * u[0]**2 + 
                  R_steer * u[1]**2)
    exprs["L"] = lag
    exprs["E"] = 0.0
    v_curr = x[3]
    theta_curr = x[2]
    a_ctrl, delta_ctrl = u[0], u[1]
    L_wheel=2.7
    exprs["f"] = cs.vertcat(
        v_curr * cs.cos(theta_curr),              # x_dot
        v_curr * cs.sin(theta_curr),              # y_dot
        v_curr * cs.tan(delta_ctrl) / L_wheel,    # theta_dot
        a_ctrl                                    # v_dot
    )
    y_initial = cs.DM([0.5, -0.5, np.pi/6, 1.0])
    exprs["e0"] = cs.vertcat(ocp.x0 - y_initial, ocp.t0 - T0)
    exprs["e0_lb"] = [0.0] * 5
    exprs["e0_ub"] = [0.0] * 5
    exprs["ef"] = cs.vertcat(ocp.tf - Tf)
    exprs["ef_lb"] = [0.0]
    exprs["ef_ub"] = [0.0]
    obstacles = [
        (2, 8, 6.0),   
        (12, 8, 2.0),
        (22, 10, 1.5)
    ]
    
    inf = cs.DM.inf()
    obs_exprs = []
    obs_lb = []
    obs_ub = []
    
    for obs in obstacles:
        ox, oy, orad = obs
        dist_sq = (x[0] - ox)**2 + (x[1] - oy)**2
        obs_exprs.append(orad**2 - dist_sq)
        obs_lb.append(-inf)
        obs_ub.append(0.0)
    
    h_list = [
        x[3],       # velocity bound indicator
        u[0],       # acceleration control
        u[1]        # steering control
    ] + obs_exprs
    
    exprs["h"] = cs.vertcat(*h_list)
    exprs["h_lb"] = [-5.0, -3.0, -np.pi/4] + obs_lb
    exprs["h_ub"] = [20.0, 3.0, np.pi/4] + obs_ub    
    ocp.set_expressions(exprs)
    ocp.transcribe(N)
    
    tg=cs.linspace(0,5,N+1).T
    Xg = cs.vertcat(
        interp_x(tg),
        interp_y(tg),
        interp_theta(tg),
        interp_v(tg),
    )
    Ug = cs.vertcat(
        cs.DM.zeros(1, N),
        cs.DM.zeros(1, N)
    )
    
    ocp.set_solver()
    sol = ocp.solve(Xg, Ug)

    plot(sol,ocp,interp_x,interp_y,interp_theta,interp_v,obstacles)

def plot(sol,ocp,interp_x,interp_y,interp_theta,interp_v,obstacles):
    X = sol.value(ocp._dis_ocp['X'])
    t_grid = sol.value(ocp._dis_ocp['t'])
    U = sol.value(ocp._dis_ocp['U'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].plot(X[0, :], X[1, :], '-b', linewidth=2, label='Actual Trajectory')
    axes[0, 0].plot(interp_x(t_grid), interp_y(t_grid), 'k--', linewidth=2, alpha=0.7, label='Reference Trajectory')
    axes[0, 0].plot(X[0, 0], X[1, 0], 'go', markersize=10, label='Start')
    axes[0, 0].plot(X[0, -1], X[1, -1], 'ro', markersize=10, label='End')
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)
        axes[0, 0].add_patch(circle)
        circle_safe = plt.Circle((obs[0], obs[1]), obs[2] + 0.5, color='orange', alpha=0.2, linestyle='--', fill=False)
        axes[0, 0].add_patch(circle_safe)
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].set_title('Vehicle Trajectory (Converged)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    axes[0, 1].plot(t_grid, X[3, :], '-b', linewidth=2, label='Actual Velocity')
    axes[0, 1].plot(t_grid, interp_v(t_grid), 'k--', linewidth=2, alpha=0.7, label='Reference Velocity')
    axes[0, 1].axhline(y=20.0, color='gray', linestyle='--', label='Max speed')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].set_title('Velocity Profile (Converged)')
    axes[0, 1].set_ylim(0, 22)
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(t_grid, X[2, :] * 180/np.pi, '-b', linewidth=2, label='Actual Heading')
    axes[0, 2].plot(t_grid, interp_theta(t_grid) * 180/np.pi, 'k--', linewidth=2, alpha=0.7, label='Reference Heading')
    axes[0, 2].set_xlabel('Time [s]')
    axes[0, 2].set_ylabel('Heading [deg]')
    axes[0, 2].set_title('Heading Angle (Converged)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    time_control = t_grid[:-1]
    axes[1, 0].plot(time_control, U[0, :], '-r', linewidth=2)
    axes[1, 0].axhline(y=3.0, color='gray', linestyle='--')
    axes[1, 0].axhline(y=-3.0, color='gray', linestyle='--')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Acceleration [m/s²]')
    axes[1, 0].set_title('Acceleration Input (Converged)')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time_control, U[1, :] * 180/np.pi, '-r', linewidth=2)
    axes[1, 1].axhline(y=45.0, color='gray', linestyle='--')
    axes[1, 1].axhline(y=-45.0, color='gray', linestyle='--')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Steering Angle [deg]')
    axes[1, 1].set_title('Steering Input (Converged)')
    axes[1, 1].grid(True)
    
    pos_errors = np.sqrt((X[0, :] - interp_x(t_grid))**2 + (X[1, :] - interp_y(t_grid))**2)
    heading_errors = np.abs(X[2, :] - interp_theta(t_grid)) * 180/np.pi
    heading_errors = np.minimum(heading_errors, 360 - heading_errors)
    
    axes[1, 2].plot(t_grid, pos_errors, '-g', linewidth=2, label='Position Error')
    axes[1, 2].plot(t_grid, heading_errors, '-r', linewidth=2, label='Heading Error')
    axes[1, 2].set_xlabel('Time [s]')
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].set_title('Tracking Errors (Converged)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_car_ocp_fatrop()