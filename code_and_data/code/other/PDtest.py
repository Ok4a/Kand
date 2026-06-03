import numpy as np
BREAKtrue = False
rng = np.random.default_rng()
for j in range(3,100):
    print(j, end='\r')
    size = j
    bound = 1


    for i in range(100):
        offdiag = rng.uniform(low=-bound, high= bound,size=(size,size))
        offdiag = (offdiag+offdiag.T)/2

        A = np.eye(size)
        A += np.triu(offdiag, k=1) + np.tril(offdiag, k = -1)
        minEig = min(np.linalg.eigvals(A))
        
        if minEig < 0:
            print(size)
            print(i)
            BREAKtrue = True

            print(A)
            print(np.linalg.det(A))
            print(np.sort(np.linalg.eigvals(A)))
            break
    if BREAKtrue:
        break
    
print()
print(np.linalg.eigvals(A[0:2,0:2]))