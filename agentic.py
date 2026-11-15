import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

def run_n_agent_optimization(n_agents=4):
    if n_agents % 2 != 0:
        raise ValueError("Please provide an even number of agents for direct swapping!")

    N_steps = 50          
    T = 6.0               
    dt = T / N_steps

    opti = ca.Opti()

    # Decision variables: States [x, y, theta] and Controls [v, omega] for each agent
    X = [opti.variable(3, N_steps + 1) for _ in range(n_agents)]
    U = [opti.variable(2, N_steps) for _ in range(n_agents)]

    # Unicycle Kinematics & RK4 Integration
    def unicycle_dynamics(state, control):
        return ca.vertcat(control[0] * ca.cos(state[2]), 
                          control[0] * ca.sin(state[2]), 
                          control[1])

    def rk4_step(x, u, dt):
        k1 = unicycle_dynamics(x, u)
        k2 = unicycle_dynamics(x + 0.5 * dt * k1, u)
        k3 = unicycle_dynamics(x + 0.5 * dt * k2, u)
        k4 = unicycle_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Workspace & Physical Specs (Box: [0, 10] x [0, 10])
    x_min, x_max = 0.0, 10.0
    y_min, y_max = 0.0, 10.0
    robot_radius = 0.35        
    safe_dist_robots = 2.0 * robot_radius + 0.4  # Inter-robot clearance buffer

    # Generate start and goal positions symmetrically on a circle centered at (5, 5)
    center_x, center_y = 5.0, 5.0
    radius = 3.5
    
    start_pos = []
    goal_pos = []
    
    for i in range(n_agents):
        angle_start = i * (2.0 * np.pi / n_agents)
        angle_goal = angle_start + np.pi  # Swap with diametrically opposite agent
        
        start_pos.append(np.array([center_x + radius * np.cos(angle_start), 
                                   center_y + radius * np.sin(angle_start)]))
        goal_pos.append(np.array([center_x + radius * np.cos(angle_goal), 
                                  center_y + radius * np.sin(angle_goal)]))

    # Objective: Minimum Control Effort (Smoothness)
    obj = 0
    for i in range(n_agents):
        for k in range(N_steps):
            obj += ca.sumsqr(U[i][:, k]) * dt
    opti.minimize(obj)

    # Constraints Setup
    for i in range(n_agents):
        # Initial condition (Position)
        opti.subject_to(X[i][0, 0] == start_pos[i][0])
        opti.subject_to(X[i][1, 0] == start_pos[i][1])

        # Final position constraint (Swap)
        opti.subject_to(X[i][0, -1] == goal_pos[i][0])
        opti.subject_to(X[i][1, -1] == goal_pos[i][1])

        for k in range(N_steps):
            # Dynamics
            opti.subject_to(X[i][:, k+1] == rk4_step(X[i][:, k], U[i][:, k], dt))
            
            # Workspace bounds (accounting for robot radius)
            opti.subject_to(X[i][0, k] >= x_min + robot_radius)
            opti.subject_to(X[i][0, k] <= x_max - robot_radius)
            opti.subject_to(X[i][1, k] >= y_min + robot_radius)
            opti.subject_to(X[i][1, k] <= y_max - robot_radius)

            # Control Limits
            opti.subject_to(U[i][0, k] >= -2.0)  # Linear velocity min
            opti.subject_to(U[i][0, k] <=  2.0)  # Linear velocity max
            opti.subject_to(U[i][1, k] >= -3.0)  # Angular velocity min
            opti.subject_to(U[i][1, k] <=  3.0)  # Angular velocity max

    # Inter-Robot Collision Avoidance across all time steps for all unique pairs
    for k in range(N_steps + 1):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                opti.subject_to(
                    (X[i][0, k] - X[j][0, k])**2 + (X[i][1, k] - X[j][1, k])**2 >= safe_dist_robots**2
                )

    # Solver configuration
    opts = {
        'ipopt.print_level': 5, 
        'ipopt.hessian_approximation': 'limited-memory', 
        'ipopt.max_iter': 2000, 
        'ipopt.tol': 1e-4
    }
    opti.solver('ipopt', opts)

    # --- GENERIC ASYMMETRIC WARM START INITIALIZATION ---
    for i in range(n_agents):
        s_x, s_y = start_pos[i][0], start_pos[i][1]
        g_x, g_y = goal_pos[i][0], goal_pos[i][1]
        
        alphas = np.linspace(0, 1, N_steps + 1)
        g_x_traj = (1 - alphas) * s_x + alphas * g_x
        g_y_traj = (1 - alphas) * s_y + alphas * g_y
        
        # Add alternating tangential offset to break symmetry
        offset_sign = 1.0 if i % 2 == 0 else -1.0
        for k in range(N_steps + 1):
            bump = 0.8 * np.sin(alphas[k] * np.pi) * offset_sign
            g_x_traj[k] -= bump * np.sin(i * (2.0 * np.pi / n_agents))
            g_y_traj[k] += bump * np.cos(i * (2.0 * np.pi / n_agents))

        # Compute heading angle theta dynamically
        g_theta_traj = np.zeros(N_steps + 1)
        for k in range(N_steps):
            dx = g_x_traj[k+1] - g_x_traj[k]
            dy = g_y_traj[k+1] - g_y_traj[k]
            g_theta_traj[k] = np.arctan2(dy, dx)
        g_theta_traj[-1] = g_theta_traj[-2]

        opti.set_initial(X[i][0, :], g_x_traj)
        opti.set_initial(X[i][1, :], g_y_traj)
        opti.set_initial(X[i][2, :], g_theta_traj)

    print(f"Solving {n_agents}-agent multi-robot optimization...")
    sol = opti.solve()
    print("Optimization solved successfully!")
    
    trajectories = [sol.value(X[i]) for i in range(n_agents)]
    return trajectories, N_steps, T, n_agents


# ==========================================
# GENERIC VISUALIZER
# ==========================================
def visualize_n_agents(trajectories, N_steps, T, n_agents):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.set_title(f"{n_agents}-Agent Circular Swap Simulation", fontsize=12, fontweight='bold')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Draw Workspace Box [0, 10] x [0, 10]
    workspace = Rectangle((0, 0), 10, 10, edgecolor='black', facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(workspace)

    # Generate distinct colors using a colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_agents)]
    
    robot_circles = []
    robot_trails = []

    for i, color in enumerate(colors):
        trail, = ax.plot([], [], color=color, alpha=0.4, linewidth=2, linestyle='-')
        robot_trails.append(trail)
        
        c = Circle((0, 0), 0.35, color=color, alpha=0.8, label=f'R{i+1}')
        ax.add_patch(c)
        robot_circles.append(c)

    ax.legend(loc='upper right', fontsize=8, ncol=2)

    def init():
        for c in robot_circles:
            c.set_center((0, 0))
        for t in robot_trails:
            t.set_data([], [])
        return robot_circles + robot_trails

    def update(frame):
        for i, c in enumerate(robot_circles):
            x = trajectories[i][0, frame]
            y = trajectories[i][1, frame]
            c.set_center((x, y))
            
            hx = trajectories[i][0, :frame+1]
            hy = trajectories[i][1, :frame+1]
            robot_trails[i].set_data(hx, hy)
            
        return robot_circles + robot_trails

    anim = animation.FuncAnimation(
        fig, update, frames=N_steps+1, init_func=init, blit=True, interval=(T*1000)/N_steps
    )
    
    plt.show()


if __name__ == "__main__":
    try:
        # Try changing n_agents to 4, 6, or 8!
        n_robots = 20
        trajectories, N_steps, T, n_agents = run_n_agent_optimization(n_agents=n_robots)
        visualize_n_agents(trajectories, N_steps, T, n_agents)
    except Exception as e:
        print("Simulation failed:", e)