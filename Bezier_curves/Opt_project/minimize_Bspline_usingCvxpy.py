
import numpy as np
import matplotlib
matplotlib.use("TkAgg")     # for macOS
import matplotlib.pyplot as plt
import cvxpy as cp

def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2


k = 3
n =  6 # n = 4 when k = 3
m = k+n+1

P = cp.Variable(n+1)

obj = cp.Minimize(451584.0*(0.571428571428571*P[0] - P[1] + 0.523809523809524*P[2] - 0.0952380952380952*P[3])**2 + 50176.0*(0.428571428571429*P[1] - P[2] + 0.857142857142857*P[3] - 0.285714285714286*P[4])**2 + 50176.0*(0.285714285714286*P[2] - 0.857142857142857*P[3] + P[4] - 0.428571428571429*P[5])**2 + 451584.0*(0.0952380952380952*P[3] - 0.523809523809524*P[4] + P[5] - 0.571428571428571*P[6])**2)

constraints = [P[0]==0, P[1]==0, P[6]==5, P[5]==5]

print(constraints)

prob = cp.Problem(obj, constraints)
prob.solve() 
print("status:", prob.status)
print("optimal value", prob.value)

t = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1])
xx = np.linspace(t[0], t[-1], 50*len(t))
xx = xx[:-1]

fig, ax = plt.subplots()
for i in range(n+1):
	Bi_k = np.array([B(x, k, i, t) for x in xx])
	ax.plot(xx, Bi_k)
ax.grid(True)
ax.set_title('B-spline functions')
plt.xlabel("Time")
# plt.ylabel("X positio)
plt.show()

curve = np.zeros((len(xx),))
for i in range(n+1):
	Bi_k = np.array([B(x, k, i, t) for x in xx])
	print(" i = {}, pi = {}".format(i, P[i].value))
	curve[:] += Bi_k*P[i].value

fig, ax = plt.subplots()
ax.plot(xx, curve)
ax.set_title('Minimum-jerk 1D trajectory')
plt.xlabel("time")
plt.ylabel("X position")
# ax.plot(xx, P[:, 1], '*')
ax.grid(True)
plt.show()