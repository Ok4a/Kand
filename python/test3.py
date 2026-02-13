import numpy as np

rng = np.random.default_rng(0)

A = [[1,2],[3,4],[5,6]]


print(A)

print(np.mean(A, axis = 0))