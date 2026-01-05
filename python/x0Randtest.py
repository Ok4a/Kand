import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
from scipy.sparse.linalg._interface import MatrixLinearOperator
from scipy.sparse.linalg._interface import LinearOperator
from time import perf_counter
import matplotlib.pyplot as plt

A = mmread("matrixData/steam3.mtx.gz")
# A = mmread("matrixData/bcspwr03.mtx.gz")
# A = mmread("matrixData/bcspwr01.mtx.gz")
size = np.shape(A)[0]

b = np.ones((size, 1))
# b = np.random.uniform(-20,20, size=(size,1))
# b = util.adj(A)@b


smallestK = np.inf
largestK = 0
smallestSeed = -1
largestSeed = -1

numOffDiag = 3
numPrecond = 1
M_inv = prec.randMShift(size, numShift = numPrecond, numOffDiag = numOffDiag, rng = 391)
k_list = []

count = 0

bound = 10
_, r_norm_nonPre , nonPreK, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose = True)
maxIter = nonPreK*10

for i in range(500):
    print("Best seed", smallestSeed, i, end = '\r')
    # M_inv = prec.shift( size, numOffDiag = numOffDiag, rng = rng)
    # M_inv = prec.multiShift(size, numShift = numPrecond, numOffDiag = numOffDiag, rng = 359)
    # if min(np.linalg.eigvals(M_inv.toarray())) <= 0:
    #     print()
    #     print(i,np.linalg.eigvals(M_inv.toarray()))
    #     # break
    # print((np.linalg.det(M_inv)))

    rng = np.random.default_rng(i)

    x0 = rng.normal(size=(size,1))
    # x0 = rng.uniform(low=-bound, high=bound, size=(size,1))

    _, _, k, flag = ls.BiCGSTAB(A, b,x0=x0, M_inv = M_inv, verbose = False, maxIter = maxIter)
    if flag == 0:
        count +=1
        k_list.append(k)
        if k < smallestK:
            smallestSeed = i
            smallestK = k
            # maxIter = k
        if k > largestK:
            largestK = k
            largestSeed = i
    # if k < nonPreK:
    #     print(i)
    #     break
print()
print(count)

rng = np.random.default_rng(smallestSeed)
# x0 = rng.uniform(low=-bound, high=bound, size=(size,1))
x0 = rng.normal(size=(size,1))
# M_inv = prec.multiShift(size, numShift = numPrecond, numOffDiag = numOffDiag, rng = rng)

# rng = np.random.default_rng(largestSeed)
# M_inv_bad = prec.multiShift(size, numShift = numPrecond, numOffDiag = numOffDiag, rng = rng)
# print(M_inv)
# print(np.linalg.eigvals(M_inv.toarray()))
# print("Condition numbers")
# print("A:", np.linalg.cond(A.toarray()))
# print("M_inv@A:", np.linalg.cond(M_inv@A.toarray()))
# print("M_inv_bad@A:", np.linalg.cond(M_inv_bad@A.toarray()))
# print()

print("Small Seed", smallestSeed)
_, r_norm_pre, k, flag = ls.BiCGSTAB(A, b,x0=x0, M_inv = None, verbose = True, maxIter = maxIter)

rng = np.random.default_rng(largestSeed)
x0 = rng.normal(size=(size,1))
# x0 = rng.uniform(low=-bound, high=bound, size=(size,1))
print("Large Seed", largestSeed)
_, r_norm_pre_bad, k, flag = ls.BiCGSTAB(A, b,x0=x0, M_inv = M_inv, verbose = True, maxIter = maxIter)

print(np.linalg.norm(x0),np.sum(x0))

plt.figure(1)

plt.hist(k_list, bins = 25)
plt.title(f"Min: {np.min(k_list)}, mean: {np.mean(k_list)}, max: {np.max(k_list)}")

fig =plt.figure(2)
ax = fig.add_subplot()
ax.set_yscale("log")
plt.plot((range(len(r_norm_nonPre[0]))), (r_norm_nonPre[0]), label = "x0=0 Precondtioned r")
plt.plot((range(len(r_norm_nonPre[1]))), (r_norm_nonPre[1]), label = "x0=0 Precondtioned s")

plt.plot((range(len(r_norm_pre[0]))), (r_norm_pre[0]), label = "x0=best Precondtioned r")
plt.plot((range(len(r_norm_pre[1]))), (r_norm_pre[1]), label = "x0=best Precondtioned s")

plt.plot((range(len(r_norm_pre_bad[0]))), (r_norm_pre_bad[0]), label = "0=Worst Precondtioned r")
plt.plot((range(len(r_norm_pre_bad[1]))), (r_norm_pre_bad[1]), label = "0=Worst Precondtioned s")
plt.legend()
plt.title("Log Residual norm")
plt.ylabel("log residual norm")
plt.xlabel("Iteration Count")
plt.show()