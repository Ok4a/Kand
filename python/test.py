import numpy as np
rng = np.random.default_rng(seed=1)
size = 20
A_temp = rng.integers(0,10,size=(size,size))
A = A_temp + A_temp.T
b = rng.integers(0,10, (size,1))
x = [np.zeros((size,1))]

print()

# A = np.matrix([[3,2],[2,6]])
# b = np.matrix([[2],[-8]])
# x = [np.zeros((2,1))]

r = [b-np.matmul(A,x[0])]
p = [r[0]]

k = 0



alpha = []
beta = []
while True:
    alpha.append((np.matmul(r[k].T,r[k])/np.matmul(p[k].T,np.matmul(A,p[k])))[0,0])
    x.append(x[k]+alpha[k]*p[k])
    r.append(r[k]-alpha[k]*np.matmul(A,p[k]))
    if np.linalg.norm(r[k+1]) < np.power(1/10,10) or np.isnan(np.linalg.norm(r[k+1])):
        break
    beta.append((np.matmul(r[k+1].T,r[k+1])/np.matmul(r[k].T,r[k]))[0,0])
    p.append(r[k+1]+beta[k]*p[k])
    k +=1
    

print("k =",k)
print(np.append(np.matmul(A,x[-1]),b, axis=1))
print(np.matmul(A,x[-1])-b)

