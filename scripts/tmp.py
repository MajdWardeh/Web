import numpy as np
from scipy.spatial.transform import Rotation
from math import degrees, pi

x = np.random.randint(5, size=(3, 5))
for p in x.T:
    print(p.shape)