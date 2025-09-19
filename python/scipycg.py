import numpy as np
import scipy.sparse.linalg as sp
import linearSolver as ls
from scipy.io import mmread
import Precondition as ps


A = mmread("matrixData/nos1.mtx.gz").toarray()
size = np.shape(A)[0]
b = np.ones((size, 1))
M_inv = ps.Jacobi(A)
x = np.zeros((size,1))
x, exit_code = sp.bicgstab(A, b, M = np.eye(size),maxiter = 1000000)
print(exit_code)
print(np.allclose(A.dot(x), b))
print(np.linalg.norm(A.dot(x)-b))
print()

x = ls.CG(A, b, M_inv=M_inv, verbose = True)
