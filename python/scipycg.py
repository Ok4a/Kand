import numpy as np
from scipy.sparse import csc_array
import scipy.sparse.linalg as sp
import linearSolver as ls
from scipy.io import mmread
import Precondition as ps


A = mmread("matrixData/bcsstk14.mtx.gz").toarray()
size = np.shape(A)[0]
b = np.ones((size, 1))
M_inv = ps.Jacobi(A)
x, exit_code = sp.bicgstab(A, b)
print(exit_code)
print(np.allclose(A.dot(x), b))

x = ls.BiCGSTAB(A,b)
print(np.allclose(A.dot(x), b))
