import linearSolver as ls
import Precondition as pre
import numpy as np
from scipy.io import mmread
from time import perf_counter

A = mmread("matrixData/bcsstk14.mtx.gz").toarray()
# A = mmread("matrixData/s1rmq4m1.mtx.gz").toarray()
size = np.shape(A)[0]
b = np.ones((size,1))
M_inv = pre.Jacobi(A)
print(type(M_inv))

# ls.CGS(A,b,M_inv,verbose=True)