# Test Robust positive invariance on high dimensional nonlinear system
# you can test different values of m, w_i and q_i and see the result in sim.png and states.png
from lib.enclosing_method_toolbox import *

print("platooning example")
m = 5 # number of robots
w_i = 10**(-4) # disturbance amplitude 
q_i = 10**(4) # Scale factor of the matrix Q

###############################################
# system definition x_dot = f(x)

n = 2*m-1  # dimension of the problem
Xi, Wi = [], [] # list of symbolic state variable
for i in range(1,n+1):
    Xi.append(sym.symbols("x"+str(i),real=True)) # xi
for i in range(1,m+1):
    Wi.append(sym.symbols("w"+str(i),real=True))

Fx = []
for i in range(n):
    if i<m-1:
        Fx.append([Xi[m+i]-Xi[m-1+i]])
    elif i == n-1:
        s = 0
        for k in range(m-1):
            s = s + Xi[k]
        Fx.append([sym.atan(-s+Xi[m-1]-2*Xi[i])+Wi[i-(m-1)]])
    else:
        Fx.append([sym.atan(Xi[i-(m-1)] + Xi[i+1] - 2 * Xi[i]) + Wi[i-(m-1)]])
Fx = sym.Matrix(Fx)
print("f(x) = ", Fx)
# Wi_box = 0.0001 * IntervalVector(m,[-1, 1])
Wi_box = w_i * IntervalVector(m,[-1, 1])
meth = PositiveInvEllipseEnclosureMethod(Fx, Xi, Wi, Wi_box)

##############################################
# stability analysis
A = meth.sys_d.Jfx(np.zeros((n,1)),np.zeros((m,1)))
# Q = 10000*scipy.linalg.solve_continuous_lyapunov(A.T, -np.eye(n))
Q = q_i*scipy.linalg.solve_continuous_lyapunov(A.T, -np.eye(n))
Q = np.array(Q).astype(int) # rounding
Q = np.array(Q).astype(float)
print("Testing Q =")
print(Q)
t0 = time.time()
Qout, res = meth.test_positive_invariance_by_enclosure(Q, delta=0.01)
t1 = time.time()
print("computational time is ",t1-t0)

###############################################
# simulation of the robots
d0, v0, dt, tend = 20, 10.0, 0.5, 20
L = m*d0 # length of the circle
r = L / (2 * np.pi)
Ai, Vi = np.array([[i*d0 + 5*random.random() for i in range(m)]]).T, np.zeros((m,1))

X = np.zeros((n,1))
for i in range(n):
    if i<m-1:
        X[i,0] = Ai[i+1,0]-Ai[i,0]-d0
    else:
        X[i,0] = Vi[i-(m-1),0]-v0

fig1 = figure()
fig1.suptitle("Platooning on the circle",fontsize = 16)
ax1 = fig1.add_subplot(111, aspect='equal')

T = np.arange(0, tend, dt)
# Col = ["r", "b", "g", "k", "m"]
for t in T:
    if t == 0:
        X_list = X
    else:
        X_list = np.concatenate((X_list, X), axis=1)
    X = X + dt * meth.sys_d.f_np(X,np.zeros((m,1)))
    Vi = X[m-1:n, :] + v0 * np.ones((m, 1))
    Ai = Ai + dt * Vi
    Pxi = r * np.cos(2 * np.pi * Ai.astype(float) / L)
    Pyi = r * np.sin(2 * np.pi * Ai.astype(float) / L)
    Thi = np.pi / 2 * np.ones((m, 1)) + 2 * np.pi / L * Ai

    # plot the vehicles
    ax1.clear()
    ax1.set_xlim(-1.1*r, 1.1*r)
    ax1.set_ylim(-1.1*r, 1.1*r)
    ax1.set_ylabel("position y")
    ax1.set_xlabel("position x")
    cir = Circle((0,0),r,color="k",fill=False)
    ax1.add_patch(cir)
    for i in range(m):
        pose = np.array([[Pxi[i, 0], Pyi[i, 0], Thi[i, 0]]]).T
        draw_tank(ax1, pose, col="k")
    fig1.savefig("platooning_result/sim.png")
    pause(0.01)

fig2 =  figure()
fig2.suptitle("State evolution",fontsize = 16)
ax2 = fig2.add_subplot(111)
ax2.set_ylabel("state x")
ax2.set_xlabel("t")
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for i in range(n):
    ax2.plot(T, X_list[i, :], "k")
fig2.savefig("platooning_result/states.png")
show()
