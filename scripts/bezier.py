import numpy as np
from numpy import linalg as la
from scipy.special import binom
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, expand
import matplotlib.pyplot as plt

def fact(x):
    if x == 1 or x == 0:
        return 1
    mult = 1
    for i in range(2, x+1):
        mult *= i
    return mult

def calc_coeff(k, n):
    return fact(n)/(fact(k)*fact(n-k))

def Bk_n(k, n, t):
    return binom(n, k)*Pow(1-t, n-k)*Pow(t, k)

def Bk_4(k, t):
    return Bk_n(k, 4, t)


def Plot(p, t, acc=100): 
# plot a Bezier curve and its derivative from 0 to 1 with 'acc' accuracy
    dp = diff(p, t)

    t_space = np.linspace(0, 1, acc)
    p_eval = [p.subs(t, ti) for ti in t_space]
    dp_eval = [dp.subs(t, ti) for ti in t_space]

    plt.plot(t_space, p_eval, 'r')
    plt.plot(t_space, dp_eval, 'b')
    plt.show()


if __name__ == "__main__":
    n = 4
    t = Symbol('t')
    # C = [3, 2.5, 1, 4, 5]
    # p = 0
    for k in range(0, n+1):
        # p += C[k]*Bk_4(k, t)
        print(expand(Bk_n(k, 2, t)))
    
    # 4th order Bernstein Coefficients to monomial coeffs
    # B = np.array([[1, -4, 6, -4, 1], [0, 4, -12, 12, -4], [0, 0, 6, -12, 6], [0, 0, 0, 4, -4], [0, 0, 0, 0, 1]], dtype=np.float64)
    # print(B)
    # print(la.inv(B))

    B = np.array([[1, -2, 1], [0, 2, -2], [0, 0, 1]]) 
    print(la.inv(B))
    