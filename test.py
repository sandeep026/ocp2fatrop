from ocp2fatrop import Ocp2Fatrop
import casadi as cs

def test():

    ocp = Ocp2Fatrop(nx=3, nu=2, T0=0, Tf=1)

    exprs = {}
    exprs["L"] = 0
    exprs["E"] = ocp.xf[2]
    exprs["e0"] = ocp.x0 - cs.DM([0] * 3)
    exprs["e0_lb"] = [0] * 3
    exprs["e0_ub"] = [0] * 3
    exprs["ef"] = ocp.xf[0] - cs.DM([1])
    exprs["ef_lb"] = [0]
    exprs["ef_ub"] = [0]
    exprs["f"] = cs.vertcat(ocp.x[1]*ocp.u[1], (ocp.u[0] - ocp.x[1])*ocp.u[1],ocp.u[1])
    exprs["h"] = cs.vertcat(
        ocp.x[1] - (1 - 0.5 * cs.sin(2 * cs.pi * ocp.x[0])), ocp.u, ocp.x[2]
    )
    inf = float('inf')
    exprs["h_lb"] = [-inf, 0,0, 0]
    exprs["h_ub"] = [0, 1,inf, inf]
    ocp.set_expressions(exprs)
    N=5
    dis_ocp = ocp.transcribe(N)
    Xg = cs.vertcat(cs.linspace(0, 1, N + 1).T, cs.DM.zeros(1, N + 1),cs.linspace(0,5,N+1).T)
    Ug = cs.vertcat(cs.DM.zeros(1, N),cs.DM.ones(1, N))
    ocp.set_solver()
    sol=ocp.solve(Xg, Ug)
    import matplotlib.pyplot as plt
    X=sol.value(ocp._dis_ocp['X'][0:2,:])
    t=sol.value(ocp._dis_ocp['X'][-1,:])#cs.np.linspace(0,1,N+1)
    U=sol.value(ocp._dis_ocp['U'][0,:])
    plt.plot(t,X[0,:],'-o', label='pos')
    plt.plot(t,X[1,:],'-o', label='vel')
    plt.plot(t[0:-1],U,'-o',label='throttle')
    plt.legend()
    plt.figure()
    plt.spy(cs.jacobian(ocp.nlp.g,ocp.nlp.x).sparsity())
    plt.show()
test()

