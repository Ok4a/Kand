import numpy as np
from scipy.io import mmread
import linearSolver as ls
import matplotlib.pyplot as plt
import util
from itertools import product

I = np.eye(2)
a = np.arange(1,5).reshape((2,2))
zero = np.zeros((2,2))

a1 = np.concatenate([zero,a.T])
a2 = np.concatenate([a,zero])
A = np.concatenate([a1,a2], axis=1)

I1 = np.concatenate([zero,I])
I2 = np.concatenate([I,zero])
AntiI = np.concatenate([I1,I2], axis=1)

print(A)

print(AntiI@A)
# rng = np.random.default_rng()
# size = 10000

# A = np.array(

#         [-0.01463557,
#         -0.01678746, -0.02122048,
#         -0.00676193, -0.01237617, -0.01208201,
#         -0.02033992, -0.01883074, -0.01200618, -0.01339265,
#         -0.01600863, -0.00575484, -0.00860635, -0.01501886, -0.0107969])

# B = []
# for i in range(len(A)):
#     for j in range(i+1,len(A)):
#         B.append(abs(A[i]-A[j]))

# print(np.mean(B))
# print(np.var(A))


# data = [x[0]*x[1] for x in rng.normal(loc=1, scale=0.3, size=(size,2))]
# print(f'Mean: {np.mean(data)}, var: {np.var(data)}, std: {np.std(data)}')
# plt.hist(data, bins=500)
# plt.show()
# # A = np.reshape(range(3*3), shape=(3,3))
# r = np.ones(shape=(3,1))
# U = np.concatenate((r,A@r), axis=1)
# print(A@U, A@A@r)


# A = np.eye(5)
# A += np.diag(np.linspace(0.1,0.9,4), k = 1)
# # print(A@A@A@A@A@A@A@A@A@A)

# # A = mmread("matrixData/bcspwr03.mtx.gz")
# # A = mmread("matrixData/bcspwr02.mtx.gz")
# # A =util.adj(A)@A
# # A = mmread("matrixData/bcsstk26.mtx.gz")
# A = mmread("matrixData/ck656.mtx.gz")
# # A = np.diag(np.random.uniform(10, size=50))
# size = np.shape(A)[0]

# b = np.ones((size, 1))
# # b = np.random.uniform(-20,20, size=(size,1))

# # b = util.adj(A)@b
# # _, r_norm_nonPre , nonPreK = ls.BiCGSTAB(A, b, verbose = True)

# # plt.plot((range(len(r_norm_nonPre[0]))), np.log(r_norm_nonPre[0]), label = "Non Precondtioned r")
# # plt.plot((range(len(r_norm_nonPre[1]))), np.log(r_norm_nonPre[1]), label = "Non Precondtioned s")
# # plt.show()
# # np.savetxt("r_norm_lap_r.txt", r_norm_nonPre[0])
# # np.savetxt("r_norm_lap_s.txt", r_norm_nonPre[1])
# # print((np.linalg.eigvals(A.toarray())))
# # print(np.linalg.cond(A.toarray()))
# # x0 = util.adj(A)@b

# # _,r_norm,k, flag = ls.BiCGSTAB(A,b,x0=None,verbose=True)
# # plt.plot((range(len(r_norm))), np.log(r_norm), label = "Non Precondtioned r")
# # plt.show()

# plt.plot(np.geomspace(-1.001,-1)+1)
# plt.show()