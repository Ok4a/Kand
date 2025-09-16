import numpy as np
from time import perf_counter
from scipy.sparse import csgraph
rng = np.random.default_rng()

def NewCG(A, b, precond = None, tol = np.pow(1/10, 10)):
    if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
        raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
    
    if type(precond) == type(None):
        isPrecond = False
    else:
        isPrecond = True
    
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point
    
    r = b - A.dot(x) # residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x
    
    if isPrecond:
        p = np.dot(precond, r).copy()
        rr_cur_prod = np.dot(r.T, p)
    else:
        rr_cur_prod = np.dot(r.T, r)
        p = r.copy()
        
    k = 0
    while True:
        Ap_prod = A.dot(p) # saves one matrix vector prod
        alpha = rr_cur_prod / np.dot(p.T, Ap_prod)

        x += alpha * p # next point

        r -= alpha * Ap_prod # next residual

        if np.linalg.norm(r) < tol: # check if residual is zero
            print("k =", k, np.linalg.norm(r))
            return x
        
        # if preconditioning
        if isPrecond:
            Mr = np.dot(precond, r) # saves multiplication
            rr_next_prod = np.dot(r.T, Mr)
            beta = rr_next_prod / rr_cur_prod
            p = Mr + beta * p # next search direction
            
        else:
            rr_next_prod = np.dot(r.T, r)
            beta = rr_next_prod / rr_cur_prod
            p = r + beta * p # next search direction

        rr_cur_prod = rr_next_prod

        k += 1


def randAb(size, l, u):

    A = rng.random((size, size)) * (u - l) + l
    b = rng.random((size,1)) * (u - l) + l
    return A, b

def GenAb(size):
    A = np.diag(rng.integers(10, size = size)) - np.eye(size, k = -1) - np.eye(size, k = 1)

    b = rng.random((size,1)) *10

    return A, b

    

if __name__ == "__main__":
    size = 1000
    # A, b = randAb(size, -1, 1)
    A, b = GenAb(size)
    A = np.matmul(A, A.T)

    M_inv = np.diag(1/np.diag(A))

    print(size)
    print()


    start = perf_counter()
    x = NewCG(A, b)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b))
    print()

    start = perf_counter()
    x = NewCG(A, b, M_inv)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b))
    print()

    # start = perf_counter()
    # x = CGS(A, b)
    # print("time", perf_counter() - start)
    # print("sol norm", np.linalg.norm(np.dot(A, x) - b))
    # print()