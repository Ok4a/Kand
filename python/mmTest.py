from linearSolver import runAll
import numpy as np
from scipy.io import mmread
from time import perf_counter

# A = mmread("matrixData/bcsstk17.mtx.gz").toarray()
A = mmread("matrixData/s1rmq4m1.mtx.gz").toarray()
size = np.shape(A)[0]
print(size)
b = np.ones((size,1))
M_inv = np.diag(1/np.diag(A))

runAll(A, b, M_inv)