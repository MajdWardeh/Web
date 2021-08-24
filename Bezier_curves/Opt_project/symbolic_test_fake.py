from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol, diff, simplify, integrate, lambdify
import nlopt

def B1(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B1(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B1(x, k-1, i+1, t)
   return c1 + c2

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

# obj = 0
# for S_ in S_list:
#     #print(S_)
#     d3S_ = diff(S_, x, 3)
#     d3S_P2 = pow(d3S_, 2)
#     obj += d3S_P2
# # print(simplify(obj))

# f = lambdify([[T] + [val for val in P]], obj, "numpy")

S0 = S_list[0]

S0_x_0 = S0.subs(x, 0)
# print("starting position condition: S0(0) == 0")
# print("{}==ps".format(S0_x_0))


dS0 = diff(S0, x)
dS0_x_0 = dS0.subs(x, 0)
# print("starting velocity condition: dS0(0) == 0")
# print("{}==0".format(dS0_x_0))


Sm = S_list[-1]
Sm_x_T = Sm.subs(x, T)
# print("end position condition: Sm(T) == 5")
# print("{}==5".format(Sm_x_T))

dSm = diff(Sm, x)
dSm_x_T = dSm.subs(x, T)
# print("end velocity condition: dSm(T) == 0")
# print("{}==0".format(dSm_x_T))

V_list = []
for S_ in S_list:
    V_list.append(diff(S_, x))


#----------------------------------------------------------------------
ps = -5
vs = 0

pf = 5
vf = 0

vmax = 0.1
steps = 1000


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

def objective(x_opt, grad):
  if grad.size > 0:
    grad[:] = 0
    grad[0] = 1
  return x_opt[0]

def initialPositionConstraint(x_opt, grad, ps):
  if grad.size > 0:
    for i in range(x_opt.shape[0]):
        grad[i] = starting_position_gradients[i](x_opt)
  return starting_position(x_opt) - ps

def initialVelocityConstraint(x_opt, grad, vs):
  if grad.size > 0:
    for i in range(x_opt.shape[0]):
        grad[i] = starting_velocity_gradients[i](x_opt)
  return starting_velocity(x_opt) - vs

def finalPositionConstraint(x_opt, grad, pf):
  if grad.size > 0:
    for i in range(x_opt.shape[0]):
        grad[i] = target_position_gradients[i](x_opt)
  return target_position(x_opt) - pf

def finalVelocityConstraint(x_opt, grad, vs):
  if grad.size > 0:
    for i in range(x_opt.shape[0]):
        grad[i] = target_velocity_gradients[i](x_opt)
  return target_velocity(x_opt) - vf


def velocityConstraint(x_opt, grad, vmax, tp):
    last_i = 0
    assert (tp >= 0) and (tp <= 1)
    for i, knot in enumerate(t):
        if tp >= knot:
            last_i = i
        else:
            #print("for index={}, knot_index = {}, knot_value = {}, step_value = {}".format(index, last_i, t[last_i], index/steps))
            break
    v_index = last_i - k
    x_t = np.zeros((n+3))
    x_t[0] = tp*x_opt[0]
    x_t[1:] = x_opt
    if grad.size > 0:
        for i in range(x_opt.shape[0]):
            grad[i] = (velocities_gradient_list[v_index])[i](x_t)
    
    return velocities_list[v_index](x_t) - vmax

def negative_velocityConstraint(x_opt, grad, vmax, tp):
    last_i = 0
    assert (tp >= 0) and (tp <= 1)
    for i, knot in enumerate(t):
        if tp >= knot:
            last_i = i
        else:
            #print("for index={}, knot_index = {}, knot_value = {}, step_value = {}".format(index, last_i, t[last_i], index/steps))
            break
    v_index = last_i - k
    x_t = np.zeros((n+3))
    x_t[0] = tp*x_opt[0]
    x_t[1:] = x_opt
    if grad.size > 0:
        for i in range(x_opt.shape[0]):
            grad[i] = -1*(velocities_gradient_list[v_index])[i](x_t)
    
    return -1*velocities_list[v_index](x_t) - vmax




def velocityAtS0(x_opt, grad, vmax, tp):
    v0 = velocities_list[0]
    x_t = np.zeros((n+3))
    x_t[0] = tp*x_opt[0]
    x_t[1:] = x_opt
    if grad.size > 0:
        for i in range(x_opt.shape[0]):
            grad[i] = (velocities_gradient_list[0])[i](x_t)
    return v0(x_t)-vmax

def negative_velocityAtS0(x_opt, grad, vmax, tp):
    v0 = velocities_list[0]
    x_t = np.zeros((n+3))
    x_t[0] = tp*x_opt[0]
    x_t[1:] = x_opt
    if grad.size > 0:
        for i in range(x_opt.shape[0]):
            grad[i] = -1*(velocities_gradient_list[0])[i](x_t)
    return -v0(x_t)-vmax


opt = nlopt.opt(nlopt.LD_SLSQP, n+2)
##### REMEBER TO CHANGE THE LOWER BOUND OF T
lower_bounds_list = []
for i in range(n+2):
    lower_bounds_list.append(-float('inf'))
lower_bounds_list[0] =  8.8142  #4.13762#1e-4
opt.set_lower_bounds(lower_bounds_list)

opt.set_min_objective(objective)

opt.add_equality_constraint(lambda x_, grad: initialPositionConstraint(x_, grad, ps), 1e-8)
opt.add_equality_constraint(lambda x_, grad: initialVelocityConstraint(x_, grad, vs), 1e-8)
opt.add_equality_constraint(lambda x_, grad: finalPositionConstraint(x_, grad, pf), 1e-8)
opt.add_equality_constraint(lambda x_, grad: finalVelocityConstraint(x_, grad, vf), 1e-8)

# for tp_ in np.linspace(0, 0.24, 100):
#     opt.add_inequality_constraint(lambda x, grad: velocityAtS0(x, grad, vmax, tp_), 1e-6)
#     opt.add_inequality_constraint(lambda x, grad: negative_velocityAtS0(x, grad, vmax, tp_), 1e-6)

for tp_ in np.linspace(0, 0.99, steps):
    opt.add_inequality_constraint(lambda x, grad: velocityConstraint(x, grad, vmax, tp_), 1e-8)
    opt.add_inequality_constraint(lambda x, grad: negative_velocityConstraint(x, grad, vmax, tp_), 1e-8)

# for i in range(steps):
#     opt.add_inequality_constraint(lambda x, grad: velocityConstraint(x, grad, vmax, i), 1e-8)
    # opt.add_inequality_constraint(lambda x, grad: accelerationConstraint(x, grad, acceleration, amax, i), 1e-8)

#Set tolerance
opt.set_xtol_rel(1e-8)

#Begin optimization with the initial values
x_opt = opt.optimize(10*np.ones((n+2)))

print("Optimal Value at ")
for xi in x_opt[1:]:
    print(xi, end = ", ")
print()
print("Minimum Time = ", x_opt[0])
print("Result = ", opt.last_optimize_result())

P = x_opt[1:]

# fig, ax = plt.subplots()
# for i in range(n+1):
#     Bi_k = np.array([B1(x, k, i, t) for x in xx])
#     ax.plot(xx, Bi_k)
# ax.grid(True)
# plt.show()


knot_vec_T = lambdify(T, knot_vec)
tt = knot_vec_T(x_opt[0])


xx = np.linspace(tt[0], tt[-1], 50*len(tt))
xx = xx[:-1]

curve = np.zeros((len(xx),))
for i in range(n+1):
    Bi_k = np.array([B1(x1, k, i, tt) for x1 in xx])
    # print(" i = {}, pi = {}".format(i, P[i]))
    curve[:] += Bi_k*P[i]

fig, ax = plt.subplots()
ax.plot(xx, curve)
ax.set_title('Minimum-time 1D trajectory using B-spline')
plt.xlabel("Time")
plt.ylabel("X position")
ax.grid(True)
plt.show()



#printing S0 and dS0 only
t1 = np.linspace(0, 0.25*x_opt[0], 100)
S0_func = lambdify([time_pluse_opt_var_list], S0)
dS0_func = lambdify([time_pluse_opt_var_list], diff(S0, x))
t_x = np.zeros((n+3))
t_x[1:] = x_opt 
S0_values = []
dS0_values = []
for t1_value in t1:
    t_x[0] = t1_value
    S0_values.append(S0_func(t_x))
    dS0_values.append(dS0_func(t_x))

t_x[0] = 0.1*x_opt[0]
print("the vel at 0.1T = ", dS0_func(t_x))
V0 = diff(S0, x)
V0_grad = []
for var in opt_var_list:
    V0_grad.append(lambdify([time_pluse_opt_var_list], diff(V0, var)) )

for i,_ in enumerate(V0_grad):
    print("V0_grad[i] = ", V0_grad[i](t_x))

grad = np.zeros((x_opt.shape[0],))
v_func = velocityAtS0(x_opt, grad, vmax, 0.1)
print("v_func = ", v_func)
print(grad)

print("testing velocityConstraint ----------------------")
grad = np.zeros((x_opt.shape[0],))
v_func = velocityConstraint(x_opt, grad, vmax, 0.1)
print("v_func = ", v_func)
print(grad)


fig, ax = plt.subplots()
ax.plot(t1, S0_values)
ax.plot(t1, dS0_values)
ax.grid(True)
plt.show()

