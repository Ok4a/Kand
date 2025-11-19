import numpy as np

subsize = 5
I = np.eye(subsize)
zero = np.zeros((subsize,subsize))
K = np.block([[zero,I],[I,zero]])
# print(np.linalg.inv(K))


print(K@K)