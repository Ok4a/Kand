import linearSolver as ls
import Precondition as prec
import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg._interface import MatrixLinearOperator
from scipy.sparse.linalg._interface import LinearOperator
from time import perf_counter
from util import randAb

A = mmread("matrixData/bcspwr01.mtx.gz")
size = np.shape(A)[0]
print(size)

# A, _ = randAb(20)
# size = np.shape(A)[0]
# print(size)

b = np.ones((size, 1))


shift = prec.shift_precondition(A)
# te = prec.Jacobi_class(A)
# M_inv = LinearOperator(shape=(size,size), matvec = shift.mv)

# M_inv = shift.Linear()




rng = np.random.default_rng(1)
for i in range(10):
    # M_inv = prec.shift_precondition(A, rng = rng).Linear()
    M_inv = np.linalg.inv(prec.shift(A,size, rng))
    # print(np.linalg.det(M_inv))
    # M_inv = prec.Jacobi(A, normalEq=True)
    ls.CG(A, b, M_inv=M_inv, tol = 1e-5, verbose = True, normal_eq = True)


