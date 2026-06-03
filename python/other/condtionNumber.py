import numpy as np
from scipy.io import mmread
import matplotlib.pyplot as plt


matrix_list = ['eris1176','fidap004','orsirr_1','sherman5']

for matrix_name in matrix_list:
    matrix = mmread(f'matrixMarket/{matrix_name}.mtx.gz')
    eigs = np.linalg.eigvals(matrix.toarray())
    print(matrix_name, max(abs(eigs))/min(abs(eigs)))
    # print(f'{matrix_name} condition number: {np.linalg.cond(matrix.toarray(),1)}, {np.max(np.sum(np.abs(matrix), axis=0))}, {np.max(np.sum(np.abs(np.linalg.inv(matrix.toarray())), axis=0))}')



# matrix = mmread('matrixMarket/eris1176.mtx.gz').toarray()
# print(np.max(np.sum(np.abs(matrix), axis=0)))
# print(np.linalg.cond(matrix, 1))


# matrix[matrix == 0] = np.nan
# plt.imshow(matrix)

# plt.show()