from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol, diff, simplify, integrate, lambdify
import nlopt

def B(x, k, i, t, B_0):
   if k == 0:
      return B_0[i]
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t, B_0)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t, B_0)
   return c1 + c2

k = 3
n =  6 # n = 4 when k = 3
m = k+n+1
print("m = ", m)
print("n = ", n)
# t = np.zeros((m+1,))
# t[3:9] = np.linspace(0, 1, 6)
# t[0:k] = 0
# t[-k:] = 1

t = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]) # k=3, n=6
#t = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])             # k=1, n=6 or k=3, n=4

xx = np.linspace(t[0], t[-1], 50*len(t))
xx = xx[:-1]
# P = np.array([ 0, 1, 2, 3, 5, 1, 2])


B_0 = []
for i in range(m):
    B_0.append(sym.Symbol('B{}_0'.format(i)))

P = []
for i in range(n+1):
    P.append(sym.Symbol('P[{}]'.format(i)))

T = Symbol('T')
knot_vec = []
for i in range(t.shape[0]):
    knot_vec.append(t[i]*T)
print("Knot vect = ", knot_vec)



x = Symbol('t')
S = 0
for i in range(m-k):
    Bi_k = B(x, k, i, knot_vec, B_0)
    S += Bi_k*P[i]

# print("S = ", sym.simplify(S))
# print()

S_list = []
for j in range(m):
    Sj = S
    list1 = np.zeros((m))
    list1[j] = 1
    for i, B in enumerate(list1):
        Sj = Sj.subs( [(B_0[i], B)])

    Sj = sym.simplify(Sj)
    if Sj != 0:
        S_list.append(Sj)
    # print("S{} = ".format(j), Sj)

print("------------------------------------------")

obj = 0
for S_ in S_list:
    #print(S_)
    d3S_ = diff(S_, x, 3)
    d3S_P2 = pow(d3S_, 2)
    obj += d3S_P2
# print(simplify(obj))

f = lambdify([[T] + [val for val in P]], obj, "numpy")


print("------------------------------------------")





S0 = S_list[0]

S0_x_0 = S0.subs(x, 0)
print("starting position condition: S0(0) == 0")
print("{}==ps".format(S0_x_0))


dS0 = diff(S0, x)
dS0_x_0 = dS0.subs(x, 0)
print("starting velocity condition: dS0(0) == 0")
print("{}==0".format(dS0_x_0))


Sm = S_list[-1]
Sm_x_T = Sm.subs(x, T)
print("end position condition: Sm(T) == 5")
print("{}==5".format(Sm_x_T))

dSm = diff(Sm, x)
dSm_x_T = dSm.subs(x, T)
print("end velocity condition: dSm(T) == 0")
print("{}==0".format(dSm_x_T))

V_list = []
for S_ in S_list:
    V_list.append(diff(S_, x))


#----------------------------------------------------------------------
ps = 2
vs = 0

pf = 50
vf = 0

vmax = 5
steps = 100


opt_var_list = [T] + [cp for cp in P]
starting_position = lambdify([opt_var_list], S0_x_0)
starting_position_gradients = [lambdify( [opt_var_list], diff(S0_x_0, var)) for var in opt_var_list]

starting_velocity = lambdify([opt_var_list], dS0_x_0)
starting_velocity_gradients = [lambdify( [opt_var_list], diff(dS0_x_0, var)) for var in opt_var_list]

target_position = lambdify([opt_var_list], Sm_x_T)
target_position_gradients = [lambdify( [opt_var_list], diff(Sm_x_T, var)) for var in opt_var_list]

target_velocity = lambdify([opt_var_list], dSm_x_T)
target_velocity_gradients = [lambdify( [opt_var_list], diff(dSm_x_T, var)) for var in opt_var_list]

time_pluse_opt_var_list = [x] + opt_var_list

velocities_list = []
velocities_gradient_list = []
for V_ in V_list:
    velocities_list.append(lambdify([time_pluse_opt_var_list], V_))
    velocities_gradient_list.append([lambdify( [time_pluse_opt_var_list], diff(V_, var)) for var in opt_var_list])

def objective(x, grad):
  if grad.size > 0:
    grad[:] = 0
    grad[0] = 1
  return x[0]

def initialPositionConstraint(x, grad, ps):
  if grad.size > 0:
    for i in range(x.shape[0]):
        grad[i] = starting_position_gradients[i](x)
  return starting_position(x) - ps

def initialVelocityConstraint(x, grad, vs):
  if grad.size > 0:
    for i in range(x.shape[0]):
        grad[i] = starting_velocity_gradients[i](x)
  return starting_velocity(x) - vs

def finalPositionConstraint(x, grad, pf):
  if grad.size > 0:
    for i in range(x.shape[0]):
        grad[i] = target_position_gradients[i](x)
  return target_position(x) - pf


def finalVelocityConstraint(x, grad, vs):
  if grad.size > 0:
    for i in range(x.shape[0]):
        grad[i] = target_velocity_gradients[i](x)
  return target_velocity(x) - vf


def velocityConstraint(x, grad, vmax, index):
    last_i = 0
    for i, knot in enumerate(t):
        if (index/steps) >= knot:
            last_i = i
        else:
            #print("for index={}, knot_index = {}, knot_value = {}, step_value = {}".format(index, last_i, t[last_i], index/steps))
            break
    v_index = last_i - 3
    X_t_optx = np.zeros((n+3))
    X_t_optx[0] = index*x[0]/steps
    X_t_optx[1:] = x
    if grad.size > 0:
        for i in range(x.shape[0]):
            grad[i] = (velocities_gradient_list[v_index])[i](X_t_optx)
    
    return velocities_list[v_index](X_t_optx) - vmax



opt = nlopt.opt(nlopt.LD_SLSQP, n+2)
##### REMEBER TO CHANGE THE LOWER BOUND OF T
lower_bounds_list = []
for i in range(n+2):
    lower_bounds_list.append(-float('inf'))
lower_bounds_list[0] = 0
opt.set_lower_bounds(lower_bounds_list)

opt.set_min_objective(objective)

#Set the equality constraints: postion constraints
opt.add_equality_constraint(lambda x, grad: initialPositionConstraint(x, grad, ps), 1e-8)
opt.add_equality_constraint(lambda x, grad: initialVelocityConstraint(x, grad, vs), 1e-8)
opt.add_equality_constraint(lambda x, grad: finalPositionConstraint(x, grad, pf), 1e-8)
opt.add_equality_constraint(lambda x, grad: finalVelocityConstraint(x, grad, vf), 1e-8)

#Set the inequality constraints: velocity and accelereation
for i in range(steps):
    opt.add_inequality_constraint(lambda x, grad: velocityConstraint(x, grad, vmax, i), 1e-8)
    # opt.add_inequality_constraint(lambda x, grad: accelerationConstraint(x, grad, acceleration, amax, i), 1e-8)

#Set tolerance
opt.set_xtol_rel(1e-8)

#Begin optimization with the initial values
x = opt.optimize(4*np.ones((n+2)))

print("Optimal Value at ", x)
print("Minimum Time = ", x[0])
print("Result = ", opt.last_optimize_result())