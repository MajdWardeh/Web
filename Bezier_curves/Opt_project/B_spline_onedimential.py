from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt

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


def bspline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))

k = 2
n =  7 # n = 4 when k = 3
m = k+n+1
# t = np.zeros((m+1,))
# t[3:9] = np.linspace(0, 1, 6)
# t[0:k] = 0
# t[-k:] = 1
tm = 1
t = tm*np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1])
xx = np.linspace(t[0], t[-1], 50*len(t))
xx = xx[:-1]
P = np.array([ 0, 1, 2, 3, 5, 1, 2, 4])
#P = np.array([ 0, 0, 1, 0, 5, 0, 0, 0])


for i in range(n+1):
	fig, ax = plt.subplots()
	Bi_k = np.array([B(x, k, i, t) for x in xx])
	ax.plot(xx, Bi_k)
	ax.grid(True)
	plt.show()



curve = np.zeros((len(xx),))
for i in range(n+1):
	Bi_k = np.array([B(x, k, i, t) for x in xx])
	print(" i = {}, pi = {}".format(i, P[i]))
	curve[:] += Bi_k*P[i]


# print(curve)


fig, ax = plt.subplots()
ax.plot(xx, curve)
# ax.plot(xx, P[:, 1], '*')
ax.grid(True)
plt.show()



# spl = BSpline(t, c, k)
# # # print(spl)
# # print(spl(2.5))

# # bspline(2.5, t, c, k)

# fig, ax = plt.subplots()
# xx = np.linspace(1.5, 4.5, 50)
# ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
# ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
# ax.grid(True)
# ax.legend(loc='best')
# plt.show()
