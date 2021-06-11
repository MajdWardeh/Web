import numpy as np

print(np.arange(10, -1, step=-1, dtype=np.int))

ar1 = np.random.randint(1, 10, (10,))
take = 5
idx = 5
print(ar1)
print(ar1[idx-take+1:idx+1])