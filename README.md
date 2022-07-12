# Demo_enclosing_method
Demo of the enclosing method to verify robust positive invairance for ellipsoids. This folder is composed of 2 examples, one blank model and the toolbox header.

## pendulum_example.py
1st example - the stability of a simple pendulum with friction \n
"
x1_dot = x2
x2_dot = -sin(x1)-2*x2+w1
"
Three state ellipsoids are tested: Q1 (is robust positive invariant), Q2 (is not positive invariant) and Q3 (is too small to be verified positive invariant)
The noise amplitude can be tuned with "Wi_box"
results are visible in pendulum_result/

## platooning_example.py
2nd example - the stability of a n-dimensional system
The cars are controlled to maintain equal distances. Three parameters can be tuned:
- the number of cars "m"
- the disturbance amplitude "w_i"
- the scale factor of the ellipsoid Q "q_i"
results are visible in platooning_result/

## blank model
Follow the instructions in blank_model.py to use the enlosing method on your system

## lib/enclosing_method_toolbox.py
Tools developped for the enclosing method. The main class is PositiveInvEllipseEnclosureMethod
