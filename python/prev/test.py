import numpy as np
from scipy.sparse.linalg import eigs
import linearSolver as ls
from time import perf_counter


A,_ = ls.randAb(100, normal=True)


start = perf_counter()
print(np.linalg.cond(A))
print("Run time:", perf_counter() - start)