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

n = 6
m = k+n+1
# t = np.zeros((m+1,))
# t[3:9] = np.linspace(0, 1, 6)
# t[0:k] = 0
# t[-k:] = 1

k = 3
n =  6 # n = 4 when k = 3
m = k+n+1
print("m = ", m)
print("n = ", n)

T = 4.13762
t = T*np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1])

P1 = [ -5.0, -5.0, 10.000000000000007, 9.971352319664783, 5.860410191559371, 5.0, 5.0]
P2 = [0.0, 3.4480166666666663, 10.000000000000034, 9.985288266985929, 7.874154579467177, 7.0, 7.0]

xx = np.linspace(t[0], t[-1], 100*len(t))
xx = xx[:-1]
P = np.zeros((n+1, 2))
for i in range(P.shape[0]):
	P[i, 0] = P1[i]
	P[i, 1] = P2[i]

print(P)

fig, ax = plt.subplots()
for i in range(n+1):
	Bi_k = np.array([B(x, k, i, t) for x in xx])
	ax.plot(xx, Bi_k)
ax.grid(True)
plt.show()



curve = np.zeros((len(xx), 2))
for i in range(n+1):
	Bi_k = np.array([B(x, k, i, t) for x in xx])
	curve[:, 0] += Bi_k*P[i, 0]
	curve[:, 1] += Bi_k*P[i, 1]

# print(curve)


fig, ax = plt.subplots()
ax.plot(curve[:, 0], curve[:, 1])
ax.plot(P[:, 0], P[:, 1], '*')
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
