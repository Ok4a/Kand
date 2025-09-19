import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import cg
import python.linearSolver as linearSolver
# P = np.array([[4, 0, 1, 0],
#               [0, 5, 0, 0],
#               [1, 0, 3, 2],
#               [0, 0, 2, 4]])
# A = csc_array(P)
# b = np.array([-1, -0.5, -1, 2])

size = 100
A, b = linearSolver.GenAb(size)
# b = np.array([-1, -0.5, -1, 2])
b = np.ones((size,1))
x, exit_code = cg(A, b, atol=1e-5)
print(exit_code)    # 0 indicates successful convergence
print(np.allclose(A.dot(x), b))