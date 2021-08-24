from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol, diff, simplify, integrate, lambdify
import nlopt

degree = 3
order = degree + 1
t = Symbol('t')
T = Symbol('T')


Ax = []
X = 0
for i in range(order):
	axi = Symbol('ax{}'.format(i))
	Ax.append(axi)
	X += axi*t**i

opt_vars = [T] + Ax

t_opt_vars = [t] + opt_vars

X_func = lambdify([t_opt_vars], X)

grad_x_func = [ambdify([t_opt_vars], diff(X_func, var)) for var in opt_vars]

dX_func = lambdify([opt_vars], diff(X, t))
d2X_func = lambdify([opt_vars], diff(X, t, 2))


def position(c, t):
  return c[0] + c[1] * t + c[2] * t**2

def velocity(c, t):
  return c[1] + 2 * c[2] * t

def acceleration(c, t):
  return 2 * c[2]

#Define the objective function: minimize T (x[0])
def objective(opt_x, grad):
  if grad.size > 0:
  	grad[:] = 0
  	grad[0] = 1
  return opt_x[0]

#Define the initial position constraint: position(ti) = pi 
def initialPositionConstraint(opt_x, grad, ts, ps):
  if grad.size > 0:
    for i in range(grad.shape[0]):
    	grad[i] = grad_x_func_at_0[i](opt_x)

   t_opt_x = np.zeros((order+2)) 
   t_opt_x[0] = ts
   t_opt_x[1:] = opt_x 
  return X_func(t_opt_x) - ps
  
#Define the final position constraint: position(T) = pf
def finalPositionConstraint(x, grad, position, pf):
  if grad.size > 0:
    grad[0] = x[2] + 2 * x[3] * x[0]
    grad[1] = 1
    grad[2] = x[0]
    grad[3] = x[0]**2
  return position(x[1:], x[0]) - pf
 
#Define the velocity constraint: velocity(tj) < vmax for every j 
def velocityConstraint(x, grad, velocity, vmax, i):
  if grad.size > 0:
    grad[0] = 2 * x[3] * i / steps
    grad[1] = 0
    grad[2] = 1
    grad[3] = 2 * (x[0] - ti) * i / steps
  return velocity(x[1:], (x[0]-ti)*i/steps) - vmax

#Define the acceleration constraint: acceleration(tj) < amax for every j
def accelerationConstraint(x, grad, acceleration, amax, i):
  if grad.size > 0:
    grad[0] = 0
    grad[1] = 0
    grad[2] = 0
    grad[3] = 2
  return acceleration(x[1:], (x[0]-ti)*i/steps) - amax

#Define the algorithm used: SQP
opt = nlopt.opt(nlopt.LD_SLSQP, 4)

#Define the lower bounds of the optimization parameters
opt.set_lower_bounds([0, -float('inf'), -float('inf'), -float('inf')])

#Set the objective function
opt.set_min_objective(objective)

#Set the equality constraints: postion constraints
opt.add_equality_constraint(lambda x, grad: initialPositionConstraint(x, grad, position, pi), 1e-8)
opt.add_equality_constraint(lambda x, grad: finalPositionConstraint(x, grad, position, pf), 1e-8)

#Set the inequality constraints: velocity and accelereation
for i in range(steps):
  opt.add_inequality_constraint(lambda x, grad: velocityConstraint(x, grad, velocity, vmax, i), 1e-8)
  opt.add_inequality_constraint(lambda x, grad: accelerationConstraint(x, grad, acceleration, amax, i), 1e-8)

#Set tolerance
opt.set_xtol_rel(1e-8)

#Begin optimization with the initial values
x = opt.optimize([1e-4, 1e-4, 1e-4, 1e-4])

print("Optimal Value at ", x[0], x[1], x[2], x[3])
print("Minimum Time = ", x[0])
print("Result = ", opt.last_optimize_result())

ft = []
t = np.linspace(ti, x[0], steps)
for i in range(len(t)):
  ft.append(x[1] + x[2] * t[i] + x[3] * t[i]**2)

print("point A: ", t[0], ", ", ft[0])
print("point B: ", t[-1], ", ", ft[-1])

# plt.xlim(0, x[0])
# plt.ylim(-1000, 200)
plt.scatter(t, ft)
plt.show()

