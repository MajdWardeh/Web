import scipy.special
import numpy as np
from numpy import linalg as la

# the two give the same results 
x = scipy.special.binom(4, 4)
print(x)

A = np.array([[1, 1, 1, 1], [0, 1/3, 2/3, 3/3], [0, 0, 1/3, 3/3],  [0, 0, 0, 1]])
print(A)
print(a.inv(A))