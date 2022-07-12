# Test Robust positive invariance on your own system 
# set the system with n, m, Fx and Wi_box
# test the positive invariance of Q
from lib.article_toolbox import *

###############################################
# system definition x_dot = f(x)
n = 1  # dimension of the state
m = 1  # dimension of the disturbance

Xi, Wi = [], [] # list of symbolic state variable
for i in range(1,n+1):
    Xi.append(sym.symbols("x"+str(i),real=True)) # xi
for i in range(1,m+1):
    Wi.append(sym.symbols("w"+str(i),real=True))
  
Fx = sym.Matrix([[-Xi[0]+Wi[0]]]) # symbolic expression of f(x)
print("f(x) = ", Fx)
# Wi_box = 0.0001 * IntervalVector(m,[-1, 1])
Wi_box = 0.001 * IntervalVector(m,[-1, 1]) # Disturbance interval
meth = PositiveInvEllipseEnclosureMethod(Fx, Xi, Wi, Wi_box)

##############################################
# stability analysis
Q = np.eye(1)
print("Testing Q =")
print(Q)
Qout, res = meth.test_positive_invariance_by_enclosure(Q, delta=0.01)