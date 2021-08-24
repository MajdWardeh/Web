import numpy as np
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify
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
    return calc_coeff(k, n)*Pow(1-t, n-k)*Pow(t, k)

def Bk_4(k, t):
    return Bk_n(k, 4, t)


def Plot(p, t, ts, acc=100): 
# plot a Bezier curve and its derivative from 0 to 1 with 'acc' accuracy
    dp = diff(p, t)
    t_space = np.linspace(0, 1, acc)
    p_eval = [p.subs(t, ti) for ti in t_space]
    # dp_eval = [dp.subs(t, ti) for ti in t_space]

    plt.plot(ts + t_space, p_eval, 'r')
    # plt.plot(ts + t_space, dp_eval, 'b')


def find_P2_control_points(P1, n, t, T=0.8):
    C2 = [0] * (n+1)
    C2[0] = P1.subs(t, T)
    C2[1] = (diff(P1, t)).subs(t, T)/n + C2[0]
    C2[2] = (diff(P1, t, 2)).subs(t, T)/(n*(n-1)) + 2*C2[1] - C2[0]
    # C2[3] = (diff(P1, t, 3)).subs(t, T)/(n*(n-1)*(n-2)) + 3*C2[2] - 3*C2[1] + C2[0]
    # C2[4] = C2[3]
    C2[3] = 1
    return C2

if __name__ == "__main__":
    n = 4
    t = Symbol('t')
    acc = 100
    t_space = np.linspace(0, 1, acc)
    C1 = [3, 3.5, 1, 4, 3]
    x = [0, 0.25, 0.5, 0.75, 1]
    # # plotting the control points of C1
    # for i in range(n+1):
    #     plt.plot(x[i], C1[i], 'r*')

    p1 = 0
    for k in range(0, n+1):
        p1 += C1[k]*Bk_4(k, t)

    # at time t = 0.8, we want to concatenate P1 with another Bezier curve P2.
    T = 0.25
    plt.plot(T, p1.subs(t, T), 'ro', label='X, concatenation point')

    p1 = 0
    for k in range(0, n+1):
        p1 += C1[k]*Bk_4(k, t)
    plt.plot(t_space, [p1.subs(t, ti) for ti in t_space],'r', label='P1')
    
    C2 = find_P2_control_points(p1, n, t, T)
    print(C2)
    p2 = 0
    for k in range(0, n+1):
        p2 += C2[k]*Bk_4(k, t)
    plt.plot(t_space + T, [p2.subs(t, ti) for ti in t_space],'g', label='P2')

    C3 = C2.copy()
    C3[2] = 0.5
    p3 = 0
    for k in range(0, n+1):
        p3 += C3[k]*Bk_4(k, t)
    plt.plot(t_space + T, [p3.subs(t, ti) for ti in t_space],'b', label='P3')
    plt.xlabel('time')
    plt.ylabel('distance')
    plt.legend(loc="lower left")
    plt.show()