from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol, diff, simplify, integrate, lambdify


n = 2
P = []
for i in range(n+1):
	P.append(sym.Symbol('P[{}]'.format(i)))

t = Symbol('t')

y = 0
for i,p in enumerate(P):
	y += p*pow(t, i)

print(y)


f = lambdify([[p for p in P] + [t]], y, 'numpy')

y_value = y.subs([(P[0], 1), (P[1], 2), (P[2], 3), (t, 5)])
print(y_value)

args = [1, 2, 3, 5]
f_value = f(args)
print(f_value)