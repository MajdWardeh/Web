import numpy as np
from math import atan2, pi

x = np.array([[ 0.9625503,   2.61321006,  0.9625503],
    [-0.29832397, -0.29832396, -2.1443235],
    [ 3.53245473,  2.81458616,  3.53245473]])

px = x[0, 1]
pz = x[2, 1]
print(px, pz)

theta = atan2(px, pz)
print(theta*180/pi)