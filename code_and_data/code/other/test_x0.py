import numpy as np
from scipy.io import mmread
import linearSolver as ls
import matplotlib.pyplot as plt
import util

A = mmread("matrixData/steam3.mtx.gz")
print("load")
size = np.shape(A)[0]

b = np.ones((size, 1))
# print(np.linalg.matrix_rank(A.toarray()))
verbose = True

print("x0: 0")
_,res0,k0, flag0 = ls.BiCGSTAB(A,b,verbose=verbose)

print("x0: 1")
x0 = np.ones((size, 1))
_,res1,k1, flag1 = ls.BiCGSTAB(A,b,x0=x0,verbose=verbose)

print("x0: A@1")
x0 = A@b
_,resA,kA, flagA = ls.BiCGSTAB(A,b,x0=x0,verbose=verbose)

print("x0: adj(A)@1")
x0 = util.adj(A)@b
_,resadjA,kadjA, flagadjA = ls.BiCGSTAB(A,b,x0=x0,verbose=verbose)

print("x0: trans(A)@1")
x0 = np.transpose(A)@b
_,resTA,kTA, flagTA = ls.BiCGSTAB(A,b,x0=x0,verbose=verbose)

results = np.array([["0",k0,flag0],["1",k1,flag1],["A",kA,flagA],["adjA",kadjA,flagadjA],["TA",kTA,flagTA]])
print(results)

fig = plt.figure(1)
ax = fig.add_subplot()
ax.set_yscale("log")
plt.plot((range(len(res0[0]))), (res0[0]), label = "res0 r")
# plt.plot((range(len(res0[1]))), (res0[1]), label = "res0 s")

plt.plot((range(len(res1[0]))), (res1[0]), label = "res1 r")
# plt.plot((range(len(res1[1]))), (res1[1]), label = "res1 s")

plt.plot((range(len(resA[0]))), (resA[0]), label = "resA r")
# plt.plot((range(len(resA[1]))), (resA[1]), label = "resA s")

plt.plot((range(len(resadjA[0]))), (resadjA[0]), label = "resadjA r")
# plt.plot((range(len(resadjA[1]))), (resadjA[1]), label = "resadjA s")

plt.plot((range(len(resTA[0]))), (resTA[0]), label = "resTA r")
# plt.plot((range(len(resTA[1]))), (resTA[1]), label = "resTA  s")
plt.legend()
plt.title("Residual norm per iteration")
plt.ylabel("residual norm")
plt.xlabel("Iteration Count")


plt.show()