import sympy as sym
import numpy as np
import scipy
import random
from scipy.linalg import sqrtm
from math import sqrt
from codac import IntervalVector, IntervalMatrix, Interval, sqr, sqrt, cos, sin, sign, abs
from matplotlib.pyplot import figure, show, quiver, FormatStrFormatter, pause, Circle
from matplotlib.patches import Ellipse
import time


def draw_field(ax, f, xmin, xmax, ymin, ymax, a):
    Mx = np.arange(xmin, xmax, a)
    My = np.arange(ymin, ymax, a)
    X1, X2 = np.meshgrid(Mx, My)
    VX, VY = f(X1, X2)
    R = np.sqrt(VX ** 2 + VY ** 2)
    quiver(Mx, My, VX / R, VY / R, width=0.002)


def random_in_interval_vector(P):  # select random vector in the interval vector P
    n = len(P)
    res = np.zeros((n, 1))
    for i in range(n):
        res[i, 0] = random.uniform(P[i].lb(), P[i].ub())
    return res


def enclose_ellipse_by_box(Q_: np.ndarray):  # enclose the ellipse E(Q) with a box
    # |xi|<||Gamma_i| (i_th column ot line bcs symetric)
    # xi = Gamma_i*y with |y|<1, y = Gamma_inv*x in the unit circle
    Gamma = sqrtm(np.linalg.inv(Q_))
    box_ = []
    for i in range(len(Q_)):
        m = np.linalg.norm(Gamma[:, [i]])
        box_.append([-m, m])
    return IntervalVector(box_)


def to_interval_matrix(M: np.array):  # transform np.array into Matrix of intervals
    rows, cols = len(M), len(M[0])
    R = IntervalMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            R[i][j] = M[i, j] * Interval(1, 1)
    return R


def draw_ellipse(ax_, Q_, mu_, s=1, prec=20, linewidths=3, col="r"):  # in 2 dimension
    n = (4 * s) / (2 * prec)
    X = np.arange(-s, s, n)
    Y = np.arange(-s, s, n)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z1 = np.zeros_like(X_mesh)
    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x = np.array([[X_mesh[i, j], Y_mesh[i, j]]]).T
            z1 = (x - mu_).T @ Q_ @ (x - mu_)
            Z1[i, j] = z1[0, 0]
    ax_.contour(X_mesh, Y_mesh, Z1, [1], colors=col, linewidths=linewidths)


class SymSystem:  # Class function containing the symbolic expression of the system x=f(x,w)
    # has many tools functions (h_enclosure, contractor function)
    def __init__(self, Fx_: sym.Matrix, Xi_, Wi_=None, Wi_box=None, t=None):
        self.Wi = Wi_  # exogenous input symbolic expression Wi_=[a1,a2]
        self.Wi_box = Wi_box  # Intervalvector of the exogenous input
        self.t = t
        if self.Wi is None:
            self.m = 0
        else:
            self.m = len(self.Wi)  # dimension of the exogenous input
        if self.t is None:
            self.time_presence = False
        else:
            self.time_presence = True

        self.Fx = Fx_  # symbolic expression of f(x) in x_dot = f(x,w)
        self.Xi = Xi_  # list of state symbols Xi_=[x1,x2,...]
        self.n, _ = Fx_.shape  # state dimension

    def f(self, x1_, x2_):  # x_dot = f(x,0) (for the vector field in D2 inputs are mesh-grid)
        f1_, f2_, x1, x2 = x1_, x2_, self.Xi[0], self.Xi[1]
        for i in range(len(x1_)):
            for j in range(len(x1_[0])):
                f_ = self.Fx.subs([(x1, x1_[i, j]), (x2, x2_[i, j])])
                for k in range(self.m):
                    f_ = f_.subs([(self.Wi[k], 0)])
                if f_[0, 0] == sym.zoo:
                    f1_[i, j] = 0  # remove zoo
                else:
                    f1_[i, j] = f_[0, 0]
                if f_[1, 0] == sym.zoo:
                    f2_[i, j] = 0
                else:
                    f2_[i, j] = f_[1, 0]
        return f1_, f2_

    def f_np(self, X_: np.array, W_=np.zeros((1, 1)), ti=0):
        # return the array f(X_,W_) knowing the symbolic expression Fx_ with the symbol list Xi_[x1,x2,...]
        f_ = self.Fx
        if np.linalg.norm(W_) == 0:
            W_ = np.zeros((self.m, 1))  # default value
        for i in range(len(X_)):
            xi = self.Xi[i]
            f_ = f_.subs([(xi, X_[i, 0])])
        for j in range(self.m):
            wi = self.Wi[j]
            f_ = f_.subs([(wi, W_[j, 0])])
        if self.time_presence:
            f_ = f_.subs([(self.t, ti)])
        return np.array(f_).astype(np.float64)

    def replace_symbolic_by_values(self, f_, X_: np.array, W_: np.array, t=0):
        # replace variables x and w in f_ by values in X_ and W_
        for i in range(self.n):
            xi = self.Xi[i]
            f_ = f_.subs([(xi, X_[i, 0])])
        for i in range(self.m):
            wi = self.Wi[i]
            f_ = f_.subs([(wi, W_[i, 0])])
        if self.time_presence:
            f_ = f_.subs([(self.t, t)])
        return np.array(f_).astype(np.float64)

    def h_ellipse(self, Q, delta):  # encapsulate the image of h(x,w,delta) by an ellipse
        Gamma = np.linalg.inv(sqrtm(Q))  # gamma_x
        X_box = enclose_ellipse_by_box(Q)  # tightest axis-aligned box
        In = to_interval_matrix(np.eye(self.n))  # Identity matrix
        Ax = np.eye(self.n) + delta * self.Jfx(np.zeros((self.n, 1)),
                                               np.zeros((self.m, 1)))  # Jacobian matrix at the origin

        Gamma_inv_ = to_interval_matrix(np.linalg.inv(Gamma))
        Ax_inv_ = to_interval_matrix(np.linalg.inv(Ax))

        Jfx_box = self.Jfx_box(X_box, self.Wi_box)  # interval of the Jacobians
        b_box = Gamma_inv_ * (Ax_inv_ * (In + delta * Jfx_box) - In) * X_box
        if self.Wi is not None:
            Jfw_box = self.Jfw_box(X_box, self.Wi_box)
            b_box = b_box + delta * Gamma_inv_ * Ax_inv_ * Jfw_box * self.Wi_box

        b_box_norm = 0
        for i in range(self.n):
            b_box_norm = b_box_norm + sqr(b_box[i])
        b_box_norm = sqrt(b_box_norm)
        rho = b_box_norm.ub()

        Ax_inv = np.linalg.inv(Ax)
        Qout = 1 / (1 + rho) ** 2 * Ax_inv.T @ Q @ Ax_inv
        return Qout

    def Jfx(self, X_: np.array, W_: np.array, t=0):  # jacobian df/dx(X_,W_,delta)
        f_ = self.Fx.jacobian(self.Xi)  # symbolic expression
        return self.replace_symbolic_by_values(f_, X_, W_, t)

    def jacobian_box_base(self, X_: IntervalVector, W_: IntervalVector, f_):
        # base function for Jhx_box and Jhd_box, compute the jacobian_box from f_
        Jh_box = IntervalMatrix(f_.rows, f_.cols)
        for i in range(f_.rows):
            for j in range(f_.cols):  # for every element of the jacobian
                fij = str(f_[i, j])
                fij = fij.replace("Abs", "abs")
                for k in range(self.n, 0, -1):  # decreasing order
                    fij = fij.replace("x" + str(k), "X_[" + str(k - 1) + "]")  # replace state by interval
                for k in range(self.m, 0, -1):
                    fij = fij.replace("w" + str(k), "W_[" + str(k - 1) + "]")  # replace external influence by interval
                a = eval(fij)
                if isinstance(a, (int, float)):
                    a = Interval(a, a)
                Jh_box[i][j] = a
        return Jh_box

    def Jfx_box(self, X_: IntervalVector, W_: IntervalVector):
        # jacobian_box of df/dx(X_,W_,t)
        f = self.Fx.jacobian(self.Xi)
        return self.jacobian_box_base(X_, W_, f)

    def Jfw_box(self, X_: IntervalVector, W_: IntervalVector):
        # jacobian_box of df/dw(X_,W_,t)
        f = self.Fx.jacobian(self.Wi)
        return self.jacobian_box_base(X_, W_, f)


class PositiveInvEllipseEnclosureMethod:
    # Test if an ellipse is positive invariant by the enclosure approach
    def __init__(self, Fxd, Xi, Wi=None, Wi_box=None):
        self.sys_d = SymSystem(Fxd, Xi, Wi, Wi_box)  # with external influence

    def test_positive_invariance_by_enclosure(self, Q0, delta):
        Qout = self.sys_d.h_ellipse(Q0, delta)
        print("Qout ="),print(Qout)
        res, L = test_positive_definite_by_cholesky(IntervalMatrix(to_interval_matrix(Qout-Q0)))
        if res:
            print("Cholesky decomposition done")
            print("ellipsoid Qout in ellipsoid Q, so the ellipse is positive invariant")
            return Qout, True
        else:
            print("Cholesky decomposition failed, no able to conclude")
            return Qout, False


def draw_result(ax, Q, Q_out, s, sys: SymSystem, Wi_box_=None):
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid()
    # draw f(x)
    draw_field(ax, sys.f, -s, s, -s, s,
               (4 * s) / (2 * 20))

    # draw ellipse
    draw_ellipse(ax, Q, np.zeros((2, 1)), s=s, prec=40, linewidths=3, col="r")
    draw_ellipse(ax, Q_out, np.zeros((2, 1)), s=s, prec=40, linewidths=3, col="g")

    # draw some trajectories
    X0 = np.linalg.inv(sqrtm(Q)) @ np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
    dt_, tend_ = 0.1, 10
    for i in range(4):  # for every point
        xi = np.zeros((2, int(tend_ / dt_)))
        xi[:, [0]] = X0[:, [i]]
        # compute the trajectory with euler
        for k in range(0, int(tend_ / dt_) - 1):
            if Wi_box_ is None:
                xi[:, [k + 1]] = xi[:, [k]] + dt_ * sys.f_np(xi[:, [k]])
            else:
                wi = random_in_interval_vector(Wi_box_)
                xi[:, [k + 1]] = xi[:, [k]] + dt_ * sys.f_np(xi[:, [k]], wi)
        ax.plot(xi[0, :], xi[1, :], "b")
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    return


def tran2H(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])


def rot2H(a):
    return np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])


def plot2D(ax, M, col='black', w=1):
    ax.plot(M[0, :], M[1, :], col, linewidth=w)


def add1(M):
    M = np.array(M)
    return np.vstack((M, np.ones(M.shape[1])))


def draw_tank(ax, x, col='darkblue', r=1, w=2):
    mx, my, th = list(x[0:3, 0])
    M = r * np.array(
        [[1, -1, 0, 0, -1, -1, 0, 0, -1, 1, 0, 0, 3, 3, 0], [-2, -2, -2, -1, -1, 1, 1, 2, 2, 2, 2, 1, 0.5, -0.5, -1]])
    M = add1(M)
    plot2D(ax, tran2H(mx, my) @ rot2H(th) @ M, col, w)


def test_positive_definite_by_cholesky(A: IntervalMatrix):  # A = L@L.T, square symetric matrix
    n, m = A.shape()
    L = IntervalMatrix(n, m)  # initialisation with same size
    if n != m:
        print("A is not square")
        return False, L

    for j in range(n):  # for every column
        S = Interval(0, 0)

        # diagonal element
        for k in range(0, j):
            S += sqr(L[j][k])
        U = A[j][j] - S
        if U.lb()<0:
            return False, L
        L[j][j] = sqrt(U)

        # the rest of the column
        for i in range(j + 1, n):
            S = Interval(0, 0)
            for k in range(0, j):
                S += L[j][k] * L[i][k]
            L[i][j] = (A[i][j] - S) / L[j][j]
            L[j][i] = Interval(0, 0)
    return True, L
