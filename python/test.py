import numpy as np
from scipy.io import mmread
import linearSolver as ls
import matplotlib.pyplot as plt


# A = np.reshape(range(3*3), shape=(3,3))
# r = np.ones(shape=(3,1))
# U = np.concatenate((r,A@r), axis=1)
# print(A@U, A@A@r)


# A = np.eye(5)
# A += np.diag(np.linspace(0.1,0.9,4), k = 1)
# print(A@A@A@A@A@A@A@A@A@A)

A = mmread("matrixData/bcspwr03.mtx.gz")
size = np.shape(A)[0]

b = np.ones((size, 1))
_, r_norm_nonPre , nonPreK = ls.BiCGSTAB(A, b, verbose = True)

plt.plot((range(len(r_norm_nonPre[0]))), np.log(r_norm_nonPre[0]), label = "Non Precondtioned r")
plt.plot((range(len(r_norm_nonPre[1]))), np.log(r_norm_nonPre[1]), label = "Non Precondtioned s")
plt.show()
np.savetxt("r_norm_lap_r.txt", r_norm_nonPre[0])
np.savetxt("r_norm_lap_s.txt", r_norm_nonPre[1])