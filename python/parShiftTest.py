import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt

A = mmread("matrixData/steam2.mtx.gz")
size = np.shape(A)[0]

b = np.ones((size, 1))

_, r_norm_non , startK, flag = ls.BiCGSTAB(A, b, verbose = True)

numCoef = 10
rng = np.random.default_rng(0)
# coefList = rng.normal(scale=0.05,size=numCoef)
start_range = 0.1
# coefList = rng.uniform(low = -start_range, high = start_range, size = numCoef)
# coefList = rng.normal(scale = start_range, size = numCoef)
coefList = np.zeros(numCoef)
print(coefList)
print(np.linalg.norm(coefList))

tol = 1e-10
M_inv = prec.parShift(size, coefList)
_, _ , bestK, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = True, tol = tol)

k_list = [bestK]
ii_list = [0]
last_change = 0
step_range = 0.1
rng = np.random.default_rng()

iteration_count = 500
# step_size_change = np.linspace(1,0, iteration_count,endpoint=False, retstep=True)
# step_size_change = np.geomspace(-0.001,-1, iteration_count)+1
# step_size_change = np.linspace(1,0.001,iteration_count)
for ii in range(1, iteration_count):
    print(bestK, ii, end = '\r')
    index = rng.choice(numCoef) # choose index to change
    # change = rng.normal(scale=lr) # how much the coef will be changed
    # change = rng.uniform(low = -step_range*step_size_change[ii], high = step_range*step_size_change[ii])
    change = rng.uniform(low = -step_range, high = step_range)

    tempCoef = coefList.copy()
    tempCoef[index] += change
    M_inv = prec.parShift(size, tempCoef)

    _, _ , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = tol, maxIter = bestK + 1)
    if k <= bestK and flag == 0: # is it better and did it converge
        
        coefList = tempCoef.copy()
        # _, _ , k, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose=True)
        bestK = k
        k_list.append(k)
        ii_list.append(ii)
        last_change = ii
        continue

    tempCoef[index] -= 2 * change # try the other direction
    M_inv = prec.parShift(size,tempCoef)

    _, _ , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = tol, maxIter = bestK + 1)
    if k <= bestK and flag == 0: # is it better and did it converge
        coefList = tempCoef.copy()
        # _, _ , k, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose=True)
        bestK = k
        k_list.append(k)
        ii_list.append(ii)
        last_change = ii
        continue
    # coefList[index] += change # go back one iteration

print(bestK, ii)
print(coefList)
print(np.linalg.norm(coefList))
print("num changes:", len(k_list)-1, "Last Change:", last_change)
M_inv = prec.parShift(size, coefList)
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