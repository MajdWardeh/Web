import numpy as np


ar1 = np.random.randint(1, 5, (10,))

ar2 = ar1[1:] - ar1[0:-1]
print(ar1)
print(ar2)