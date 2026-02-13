import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt
import time
import genLaplace

def percentageShiftTest(A, seed, step_range, linspace_end, iteration_count, numCoef, file):
    size = np.shape(A)[0]

    b = np.ones((size, 1))

    _, r_norm_non , startK, flag = ls.BiCGSTAB(A, b, verbose = True)

    # numCoef = 8
    rng = np.random.default_rng(seed)
    coefList = rng.normal(scale=0.05, size=numCoef)
    start_range = 0.1
    # coefList = rng.uniform(low = -start_range, high = start_range, size = numCoef)
    # coefList = rng.normal(scale = start_range, size = numCoef)
    # coefList = np.zeros(numCoef)
    print(coefList)
    print(np.linalg.norm(coefList))

    tol = 1e-10
    M_inv = prec.parShift(size, coefList)
    _, _ , bestK, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, verbose = True, tol = tol)

    k_list = [bestK]
    ii_list = [0]
    last_change = 0

    saveData = [(0, bestK)]
    file.write(f'{0}: {bestK}\n')


    pm = [1, -1] # tries both directions
    change_scale = [1,0.5,0.25] # scales of changes


    # iteration_count = 100
    # step_size_change = np.linspace(1,0, iteration_count, endpoint = False, retstep = True)
    # step_size_change = np.geomspace(-0.001, -1, iteration_count) + 1
    if linspace_end is not None:
        step_size_change = np.linspace(1, linspace_end, iteration_count-1)
    for ii in range(1, iteration_count):
        print(bestK, ii, end = '\r')
        index = rng.choice(numCoef) # choose index to change
        # change = rng.normal(scale=lr) # how much the coef will be changed
        if linspace_end is None:
            change = rng.uniform(low = -step_range, high = step_range)
        else:
            change = rng.uniform(low = -step_range, high = step_range) * step_size_change[ii-1]

        found_better = False
        for gsji in pm:
            for scale in change_scale:

                tempCoef = coefList.copy()
                tempCoef[index] += change*gsji*scale
                M_inv = prec.parShift(size, tempCoef)

                _, _ , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = tol, max_iter = bestK + 1)
                if k <= bestK and flag == 0: # is it better and did it converge
                    
                    coefList = tempCoef.copy()
                    # _, _ , k, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose=True)
                    bestK = k
                    k_list.append(k)
                    ii_list.append(ii)
                    last_change = ii
                    file.write(f'{ii}: {k}\n')
                    saveData.append((ii,k))
                    found_better = True
                    break

            if found_better:
                break

    # np.savetxt('test.txt', saveData, fmt='%d')
    print()
    return bestK

temp = []
numSeeds = 5
step_range = 0.1
# matrix = "matrixData/Sherman5.mtx.gz"
matrix = "Laplace"
iteration_count = 500
numCoef = 10
# A = mmread(matrix)
A = genLaplace.gen2dLaplace(30, seed = 0)


with open('laplaceTest.txt', mode = 'a') as txt_file:
    txt_file.write(str(time.time()) + '\n')
    txt_file.write(matrix)

    txt_file.write(f'\n')
    for seed in range(numSeeds):
        print('Seed:', seed)
        txt_file.write(f'\nSeed: {seed}\n')
        temp.append(percentageShiftTest(A=A, seed=seed, step_range=step_range, linspace_end=None, iteration_count=iteration_count,numCoef=numCoef, file=txt_file))

    txt_file.write(str(np.mean(temp)))
    # np.savetxt('test.txt', temp,fmt='%d')
