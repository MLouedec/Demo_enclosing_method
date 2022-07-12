# Test Robust positive invariance on 2d nonlinear system
from lib.enclosing_method_toolbox import *

print("Pendulum example")
###############################################
# system definition x_dot = f(x)
n = 2  # dimension of the problem [th th_dot]
x1, x2, w1 = sym.symbols("x1 x2 w1",real=True)
Fx = sym.simplify(sym.Matrix([[x2],[-sym.sin(x1) - 2 * x2]])) + sym.Matrix([[0], [w1]]) # f(x)
print("f(x) = ", Fx)

Xi,Wi  = [x1, x2],[w1]
Wi_box = 1 * IntervalVector([[-0.1, 0.1]]) # [w]

##############################################
# stability analysis
Q1 = np.array([[6,2],[2,2]])
Q2 = np.array([[6,-2],[-2,2]])
Q3 = 10*Q1

print("Testing Q1")
meth = PositiveInvEllipseEnclosureMethod(Fx, Xi, Wi, Wi_box)
Q1_out, _ = meth.test_positive_invariance_by_enclosure(Q1, delta=0.1)
fig1 = figure()
ax1 = fig1.add_subplot(111, aspect='equal')
draw_result(ax1,Q1,Q1_out,1,meth.sys_d,Wi_box)
ax1.set_title("Q1")
fig1.savefig('pendulum_result/Q1.png')

print("Testing Q2")
Q2_out, _ = meth.test_positive_invariance_by_enclosure(Q2, delta=0.1)
fig2 = figure()
ax2 = fig2.add_subplot(111, aspect='equal')
draw_result(ax2,Q2,Q2_out,1,meth.sys_d,Wi_box)
ax2.set_title("Q2")
fig1.savefig('pendulum_result/Q2.png')

print("Testing Q3")
Q3_out, _ = meth.test_positive_invariance_by_enclosure(Q3, delta=0.1)
fig3 = figure()
ax3 = fig3.add_subplot(111, aspect='equal')
draw_result(ax3,Q3,Q3_out,0.3,meth.sys_d,Wi_box)
ax3.set_title("Q3")
fig1.savefig('pendulum_result/Q3.png')

print("The test is done, look at pendulum_result/Q1.png, /Q2.png and /Q3.png")
show()
