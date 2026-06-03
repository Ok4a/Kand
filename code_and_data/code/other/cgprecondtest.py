import numpy as np
import scipy.sparse.linalg as sp
import linearSolver as ls
from scipy.io import mmread
from scipy.sparse.linalg._interface import MatrixLinearOperator
import Precondition as pc



A = mmread("matrixData/bcspwr01.mtx.gz")
size = np.shape(A)[0]
print(size)

b = np.ones((size, 1))
b_tilde = np.conj(A.T) @ b

A_tilde = np.conj(A.T) @ A
# M_inv = pc.Jacobi(A, normalEq=True)
# M_inv = pc.Jacobi(A)
M_inv = np.eye(size)
# M_inv, _ = pc.shuffle(size)

# print(np.linalg.cond(A_tilde.toarray()))    
# print(np.linalg.cond(M_inv@A_tilde.toarray()))

# ls.CG(A_tilde, b_tilde, M_inv=M_inv, verbose = True, tol = 1e-5)
# ls.CG(A, b,M_inv=M_inv, verbose = True, tol = 1e-5, normal_eq=True)

# ls.CG(A, b,M_inv=M_inv, verbose = True, tol = 1e-5)

ls.CG(A,b,M_inv=M_inv, verbose=True, normal_eq=True)
ls.CGNew(A_tilde,b_tilde,M_inv=M_inv,verbose=True)