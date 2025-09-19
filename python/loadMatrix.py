from scipy.io import mmread
import numpy as np

m = mmread("matrixData/bcsstk14.mtx.gz")
size = np.shape(m)[0]
print(size)