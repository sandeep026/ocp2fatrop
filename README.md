## ocp2fatrop

Fatrop is a structure exploiting discrete OCP solver for fast numerical optimal control. Inspired by IPOPT, it implements an primal dual IP algorithm and uses Ricatti recursion to solve the linear system at each Newton
iteration efficiently. If the KKT solve time is currently a bottleneck in your setup, switching to FATROP can speed up things significantly. However, this solver requires a special ordering of variables and constraints
to work. It also requires the transcription must be multiple shooting.

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
