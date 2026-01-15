import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt
import winsound


def linspaceTest(A_name, seed, step_range, linspace_end,iteration_count, file):
    A = mmread(A_name)
    size = np.shape(A)[0]

    b = np.ones((size, 1))

    _, r_norm_non , startK, flag = ls.BiCGSTAB(A, b, verbose = True)

    numCoef = 8
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
    # step_range = 0.1
    # rng = np.random.default_rng()

    saveData = [(0, bestK)]
    file.write(f'{0}: {bestK}\n')


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
            file.write(f'{ii}: {k}\n')
            saveData.append((ii,k))
            continue

        tempCoef[index]-=2*change # try the other direction
        M_inv = prec.parShift(size,tempCoef)

        _, _ , k, flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = tol, maxIter = bestK + 1)
        if k <= bestK and flag == 0: # is it better and did it converge
            coefList = tempCoef.copy()
            # _, _ , k, flag = ls.BiCGSTAB(A, b,M_inv=M_inv, verbose=True)
            bestK = k
            k_list.append(k)
            ii_list.append(ii)
            last_change = ii
            file.write(f'{ii}: {k}\n')
            saveData.append((ii,k))
            continue
        # coefList[index] += change # go back one iteration

    # np.savetxt('test.txt', saveData, fmt='%d')
    print()
    return saveData

temp = []
numSeeds = 1
step_range = 0.1
matrix = "matrixData/steam2.mtx.gz"
iteration_count= 500

with open('test.txt', mode='a') as txt_file:

    txt_file.write(f'Linspace\n')
    for seed in range(numSeeds):
        print('Seed:', seed)
        txt_file.write(f'\nSeed: {seed}\n')
        temp.append(linspaceTest(A_name=matrix,seed=seed,step_range=step_range,linspace_end=0.001,iteration_count=iteration_count,file=txt_file))

    txt_file.write(f'\n')
    txt_file.write(f'No scale\n')
    for seed in range(numSeeds):
        print('Seed:', seed)
        txt_file.write(f'\nSeed: {seed}\n')
        temp.append(linspaceTest(A_name=matrix, seed=seed, step_range=step_range, linspace_end=None,iteration_count=iteration_count, file=txt_file))

    # np.savetxt('test.txt', temp,fmt='%d')
# winsound.Beep(500,500)
