import numpy as np
import scipy.sparse.linalg as sp
import linearSolver as ls
from scipy.io import mmread
import Precondition as pc


A = mmread("matrixData/conf5.0-00l4x4-1000.mtx.gz").toarray()
# size = 100
# A = np.random.random((size,size))
size = np.shape(A)[0]
print(size)

b = np.conj(A.T) @ np.ones((size, 1))

A = np.conj(A.T)@A
# A += np.eye(size, dtype=complex)*0.5
M_inv = pc.Jacobi(A)
x, exit_code = sp.cg(A, b, M = M_inv, rtol = np.pow(1/10,10))
# print(exit_code)
# print(np.allclose(A.dot(x), b))
# print(np.linalg.norm(A.dot(x) - b))
# print()

# print(np.linalg.cond(A))
# print(np.linalg.cond(M_inv@A))

ls.CG(A, b, verbose = True, tol = np.pow(1/10,5))
ls.CG(A, b, M_inv = M_inv, verbose = True, tol = np.pow(1/10,5))
# ls.BiCGSTAB(A, b, verbose = True)
# ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = True)

