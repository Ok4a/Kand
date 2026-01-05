import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt

A = mmread("matrixData/sherman5.mtx.gz")
size = np.shape(A)[0]

b = np.ones((size, 1))

_, r_norm_non , startK, flag = ls.BiCGSTAB(A, b, verbose = True)

rng = np.random.default_rng()
start_range = 0.0001
numCoef = 4
upperCoefList = rng.uniform(low = -start_range, high = start_range, size = numCoef)
# coefList = rng.normal(scale=start_range, size = numCoef)


lowerCoefList = rng.uniform(low = -start_range, high = start_range, size = numCoef)
# coefList = rng.normal(scale=start_range, size = numCoef)


print(upperCoefList, lowerCoefList)

tol = 1e-10
M_inv = prec.parShift(size, upperCoefList, lowerCoefList)
_, _ , bestK, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = True, tol = tol)

k_list = [bestK]
ii_list = [0]

lr = 1

for ii in range(1, 250):
    print(bestK, ii, end = '\r')
    index = rng.choice(numCoef) # choose index to change
    UoL = rng.uniform()
    # change = rng.normal(scale=lr) # how much the coef will be changed
    change = rng.uniform(low = -lr, high = lr)

    tempUpperCoef = upperCoefList.copy()
    tempLowerCoef = lowerCoefList.copy()
    if UoL < 0.5:
        tempUpperCoef[index] += change
    else:
        tempLowerCoef[index] += change
    M_inv = prec.parShift(size, tempUpperCoef,tempLowerCoef)

    _, _ , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = tol, maxIter = bestK + 1)
    if k <= bestK and flag == 0: # is it better and did it converge

        lowerCoefList = tempLowerCoef.copy()
        upperCoefList = tempUpperCoef.copy()
        # _, _ , k, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose=True)
        bestK = k
        k_list.append(k)
        ii_list.append(ii)
        continue

    # tempUpperCoef[index] -= 2 * change # try the other direction
    if UoL < 0.5:
        tempUpperCoef[index] -= 2 * change
    else:
        tempLowerCoef[index] -= 2 * change
    M_inv = prec.parShift(size,tempUpperCoef,tempLowerCoef)

    _, _ , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = tol, maxIter = bestK + 1)
    if k <= bestK and flag == 0: # is it better and did it converge
        upperCoefList = tempUpperCoef.copy()
        lowerCoefList = tempLowerCoef.copy()
        # _, _ , k, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose=True)
        bestK = k
        k_list.append(k)
        ii_list.append(ii)
        continue
    # coefList[index] += change # go back one iteration

print(bestK, ii)
print(upperCoefList)
print(lowerCoefList)
print(k_list)

M_inv = prec.parShift(size, upperCoefList, lowerCoefList)
_, r_norm_best , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = True)

plt.figure(1)
plt.plot(ii_list, k_list)
plt.ylabel("BiCGStab iteration count")
plt.xlabel("Learning iteration")
plt.figure(2)
plt.plot(k_list)
plt.ylabel("BiCGStab iteration count")
plt.xlabel("")


fig = plt.figure(3)
ax = fig.add_subplot()
ax.set_yscale("log")
plt.plot((range(len(r_norm_non[0]))), (r_norm_non[0]), label = "Non r")
plt.plot((range(len(r_norm_non[1]))), (r_norm_non[1]), label = "Non s")

plt.plot((range(len(r_norm_best[0]))), (r_norm_best[0]), label = "Best r")
plt.plot((range(len(r_norm_best[1]))), (r_norm_best[1]), label = "Best s")
plt.legend()
plt.title("Log Residual norm")
plt.ylabel("log residual norm")
plt.xlabel("Iteration Count")
plt.show()


def ta(h:str)->str:
    pass