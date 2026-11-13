## ocp2fatrop

Fatrop is a structure exploiting discrete OCP solver for fast numerical optimal control. Inspired by IPOPT, it implements an primal dual IP algorithm and uses Ricatti recursion to solve the linear system at each Newton
iteration efficiently. If the KKT solve time is currently a bottleneck in your setup, switching to FATROP can speed up things significantly. However, this solver requires a special ordering of variables and constraints
to work. It also requires the transcription must be multiple shooting.

## Problem

The `Ocp2Fatrop` class supports continuous-time **Bolza-form Optimal Control Problems (OCPs)** defined by the following mathematical program:

$$\min_{x(t), u(t), t_0, t_f} \int_{t_0}^{t_f} L(t, x(t), u(t)) \, dt + E(t_0, t_f, x(t_0), x(t_f))$$

**Subject to:**

1. **System Dynamics:**

$$\dot{x}(t) = f(t, x(t), u(t)) \quad \forall t \in [t_0, t_f]$$


2. **Path Constraints:**

$$h_{\text{lb}} \le h(t, x(t), u(t)) \le h_{\text{ub}}$$


3. **Initial Boundary / Event Constraints:**

$$e_{0,\text{lb}} \le e_0(t_0, x(t_0)) \le e_{0,\text{ub}}$$


4. **Final Boundary / Event Constraints:**

$$e_{f,\text{lb}} \le e_f(t_f, x(t_f)) \le e_{f,\text{ub}}$$

### Description of Variables and Functions

The code maps these mathematical elements into CasADi symbolic variables and Python attributes:

#### Dimensions & Horizon

* **`nx`**: Dimension of the state vector $x(t)$.
* **`nu`**: Dimension of the control input vector $u(t)$.
* **`T0`, `Tf**`: Nominal initial and final time bounds of the horizon.

#### Symbolic Variables

* **`t`**: Time symbol (`cs.SX`).
* **`x`**: State vector symbol of size `nx` (`cs.SX`).
* **`u`**: Control vector symbol of size `nu` (`cs.SX`).
* **`t0`, `tf**`: Initial and final time symbols.
* **`x0`, `xf**`: Initial and final state vector symbols.

#### Objective & Dynamics Expressions (`set_expressions`)

* **`_L`**: Lagrange term (running cost integrand $L$).
* **`_E`**: Mayer term (terminal cost function $E$).
* **`_f`**: Continuous-time ODE dynamics function $f$.

#### Constraints & Bounds (`set_expressions`)

* **`_h`, `_h_lb`, `_h_ub**`: Path constraint function $h$ along with its lower ($h_{\text{lb}}$) and upper ($h_{\text{ub}}$) numerical bounds.
* **`_e0`, `_e0_lb`, `_e0_ub**`: Initial event/boundary condition function $e_0$ with its bounds.
* **`_ef`, `_ef_lb`, `_ef_ub**`: Final event/boundary condition function $e_f$ with its bounds.

### Decision vector

$$
W=\begin{bmatrix}
x_1\\
u_1\\
x_2\\
u_2\\
\vdots \\
x_{n}\\
u_{n}\\
x_{n+1}
\end{bmatrix}
$$

### Constraint vector

$$
G=\begin{bmatrix}
r_1\\
h_1\\
e_1\\
r_2\\
h_2\\
\vdots \\
r_{n}\\
h_{n}\\
e_{n+1}
\end{bmatrix}
$$

$$
G_{lb}=\begin{bmatrix}
r_{(lb,1)}\\
h_{(lb,1)}\\
e_{(lb,1)}\\
r_{(lb,2)}\\
h_{(lb,2)}\\
\vdots \\
r_{(lb,n)}\\
h_{(lb,n)}\\
e_{(lb,n+1)}
\end{bmatrix}
$$

$$
G_{ub}=\begin{bmatrix}
r_{(ub,1)}\\
h_{(ub,1)}\\
e_{(ub,1)}\\
r_{(ub,2)}\\
h_{(ub,2)}\\
\vdots \\
r_{(ub,n)}\\
h_{(ub,n)}\\
e_{(ub,n+1)}
\end{bmatrix}
$$
