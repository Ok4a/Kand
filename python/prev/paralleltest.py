from multiprocessing import Pool
import genLaplace
import linearSolver
import numpy as np
from time import perf_counter


def f(A,M_inv):
    size = np.shape(A)[0]
    b = np.ones((size, 1))
    _,_,k,_ = linearSolver.BiCGSTAB(A, b)
    return k


if __name__ == '__main__':

    data = genLaplace.genLaplaceData(30, seed_list=range(500))
    start = perf_counter()
    with Pool(1) as p:
        k_list = p.starmap(f, ((d, 1) for d in data))
    print(perf_counter()-start)
    print(k_list)