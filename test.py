
def test():

    trajopt = Ocp2Fatrop(nx=3, nu=2, T0=0, Tf=1)

    exprs = {}
    exprs["L"] = 0
    exprs["E"] = trajopt.xf[2]
    exprs["e0"] = trajopt.x0 - cs.DM([0] * 3)
    exprs["e0_lb"] = [0] * 3
    exprs["e0_ub"] = [0] * 3
    exprs["ef"] = trajopt.xf[0] - cs.DM([1])
    exprs["ef_lb"] = [0]
    exprs["ef_ub"] = [0]
    exprs["f"] = cs.vertcat(trajopt.x[1]*trajopt.u[1], (trajopt.u[0] - trajopt.x[1])*trajopt.u[1],trajopt.u[1])
    exprs["h"] = cs.vertcat(
        trajopt.x[1] - (1 - 0.5 * cs.sin(2 * cs.pi * trajopt.x[0])), trajopt.u, trajopt.x[2]
    )
    inf = cs.DM.inf()
    exprs["h_lb"] = [-inf, 0,0, 0]
    exprs["h_ub"] = [0, 1,inf, inf]
    trajopt.set_expressions(exprs)
    N=5
    dis_ocp = trajopt.transcribe(N)
    Xg = cs.vertcat(cs.linspace(0, 1, N + 1).T, cs.DM.zeros(1, N + 1),cs.linspace(0,5,N+1).T)
    Ug = cs.vertcat(cs.DM.zeros(1, N),cs.DM.ones(1, N))
    sol=trajopt.solve(Xg, Ug)
    import matplotlib.pyplot as plt
    X=sol.value(trajopt._dis_ocp['X'][0:2,:])
    t=sol.value(trajopt._dis_ocp['X'][-1,:])#cs.np.linspace(0,1,N+1)
    U=sol.value(trajopt._dis_ocp['U'][0,:])
    plt.plot(t,X[0,:],'-o', label='pos')
    plt.plot(t,X[1,:],'-o', label='vel')
    plt.plot(t[0:-1],U,'-o',label='throttle')
    plt.legend()
    plt.figure()
    plt.spy(cs.jacobian(trajopt.nlp.g,trajopt.nlp.x).sparsity())
    plt.show()
test()

