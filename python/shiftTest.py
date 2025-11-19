import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
from scipy.sparse.linalg._interface import MatrixLinearOperator
from scipy.sparse.linalg._interface import LinearOperator
from time import perf_counter
import matplotlib.pyplot as plt

# A = mmread("matrixData/bcsstk01.mtx.gz")
A = mmread("matrixData/bcspwr03.mtx.gz")
size = np.shape(A)[0]

b = np.ones((size, 1))

_, r_norm_nonPre , nonPreK = ls.BiCGSTAB(A, b, verbose = True)

smallestK = 10000
smallestSeed = -1

numOffDiag = 2
numPrecond = 2
k_list = []


for i in range(1000):
    print(i, end = '\r')
    
    rng = np.random.default_rng(i)
    # M_inv = prec.shift( size, numOffDiag=numOffDiag, rng=rng)
    M_inv = prec.multiShift(size, numShift = numPrecond, numOffDiag = numOffDiag, rng = rng)
    # if min(np.linalg.eigvals(M_inv.toarray())) <= 0:
    #     print()
    #     print(i,np.linalg.eigvals(M_inv.toarray()))
    #     # break
    # print((np.linalg.det(M_inv)))
    _, _, k = ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = False)
    k_list.append(k)
    if k < smallestK:
        smallestSeed = i
        smallestK = k
    # if k < nonPreK:
    #     print(i)
    #     break

print()
print(smallestK, smallestSeed)

rng = np.random.default_rng(smallestSeed)
M_inv = prec.multiShift(size, numShift = numPrecond, numOffDiag = numOffDiag, rng = rng)
# print(M_inv)
# print(np.linalg.eigvals(M_inv.toarray()))
# print(np.linalg.cond(A.toarray()),np.linalg.cond(M_inv@A))

_, r_norm_pre, k = ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = True)
plt.figure(1)
plt.hist(k_list, bins=25)
plt.title(f"Min: {np.min(k_list)}, mean: {np.mean(k_list)}, max: {np.max(k_list)}")

plt.figure(2)
plt.plot((range(len(r_norm_nonPre[0]))), np.log(r_norm_nonPre[0]), label = "Non Precondtioned r")
plt.plot((range(len(r_norm_nonPre[1]))), np.log(r_norm_nonPre[1]), label = "Non Precondtioned s")
plt.plot((range(len(r_norm_pre[0]))), np.log(r_norm_pre[0]), label = "Precondtioned r")
plt.plot((range(len(r_norm_pre[1]))), np.log(r_norm_pre[1]), label = "Precondtioned s")
plt.legend()
plt.title("Log Residual norm")
plt.ylabel("log residual norm")
plt.xlabel("Iteration Count")
plt.show()