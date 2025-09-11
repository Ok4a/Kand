import numpy as np
from time import perf_counter
rng = np.random.default_rng()


def CG(A, b):
    if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
        raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point
    
    r_current = b - np.matmul(A, x) # residual
    if np.linalg.norm(r_current) < np.power(1/10, 10): # check if initial point works
        return x
    rr_cur_prod = np.matmul(r_current.T, r_current)

    p = r_current

    k = 1
    while True:
        Ap_prod = np.matmul(A, p)

        alpha = (rr_cur_prod / np.matmul(p.T, Ap_prod))[0,0]

        x = x + alpha * p # next point

        r_next = r_current - alpha * Ap_prod

        if np.linalg.norm(r_next) < np.power(1/10, 10): # check if residual is zero
            print("k =",k,np.linalg.norm(r_next))
            return x, r_next
        
        rr_next_prod = np.matmul(r_next.T, r_next)
        beta = (rr_next_prod / rr_cur_prod)[0, 0]
        p = r_next + beta * p

        rr_cur_prod = rr_next_prod
        r_current = r_next

        k += 1

def PrecondCG(A, b, M_inv):
    if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
        raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point
    
    r_current = b - np.matmul(A,x) # residual
    if np.linalg.norm(r_current) < np.power(1/10, 10): # check if initial point works
        return x
    
    p = np.matmul(M_inv,r_current)
    rMr_cur_prod = np.matmul(r_current.T, p)

    k = 1
    while True:
        Ap_prod = np.matmul(A, p)

        alpha = (rMr_cur_prod / np.matmul(p.T, Ap_prod))[0,0]
        
        x = x + alpha * p
        r_next = r_current - alpha * Ap_prod    

        if np.linalg.norm(r_next) < np.power(1/10, 10): # check if residual is zero
            print("k =",k,np.linalg.norm(r_next))
            return x, r_next
        
        Mr = np.matmul(M_inv, r_next)
        
        rMr_next_prod = np.matmul(r_next.T, Mr)

        beta = (rMr_next_prod / rMr_cur_prod)[0, 0]

        p = Mr + beta * p

        rMr_cur_prod = rMr_next_prod

        r_current = r_next

        k += 1


def randAb(size, l, u):
    A = rng.random((size, size)) * (u - l) + l
    A = np.matmul(A, A.T)

    b = rng.random((size,1)) * (u - l) + l

    return A, b


if __name__ == "__main__":
    size = 1000
    A, b = randAb(size, 1, -1)
    M_inv = (np.diag(1/np.diag(A)))

    condA = np.linalg.cond(A)
    condMA = np.linalg.cond(np.matmul(M_inv, A))
    print(size, condA, condMA)
    print()

    start = perf_counter()
    x,r = CG(A, b)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.matmul(A, x) - b))
    print()

    start = perf_counter()
    x,r = PrecondCG(A, b, M_inv)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.matmul(A, x) - b))