import numpy as np
from math import pow

def bezier4thOrder(cp, t):
    # t must be in the range [0, 1]
    coeff_arr = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    assert cp.shape[1] == 5, 'assertion failed, the provided control points (cp) are not for a 4th order Bezier.'
    P = np.zeros_like(cp, dtype=np.float64)
    for k in range(5):
        P[:, k] = cp[:, k] * coeff_arr[k] * pow(1-t, 4-k) * pow(t, k)
    return np.sum(P, axis=1)
    



