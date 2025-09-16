import numpy as np
from time import perf_counter
rng = np.random.default_rng()


def CG(A, b, tol = np.pow(1/10, 10)):
    if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
        raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point
    
    r = b - A.dot(x) # residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x
    rr_cur_prod = np.dot(r.T, r)
    p = r.copy()

    k = 0
    while True:
        Ap_prod = A.dot(p)
        alpha = rr_cur_prod / np.dot(p.T, Ap_prod)

        x += alpha * p # next point

        r -= alpha * Ap_prod

        if np.linalg.norm(r) < tol: # check if residual is zero
            print("k =", k, np.linalg.norm(r))
            return x
        
        rr_next_prod = np.dot(r.T, r)
        
        beta = rr_next_prod / rr_cur_prod
        p = r + beta * p

        rr_cur_prod = rr_next_prod

        k += 1
def PrecondCG(A, b, M_inv, tol = np.pow(1/10, 10)):
    if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
        raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point
    
    r = b - np.dot(A,x) # residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x
    
    p = np.dot(M_inv,r).copy()
    rMr_cur_prod = np.dot(r.T, p)

    k = 0
    while True:
        Ap_prod = np.dot(A, p)

        alpha = (rMr_cur_prod / np.dot(p.T, Ap_prod))[0,0]
        
        x += alpha * p
        r -= alpha * Ap_prod    

        if np.linalg.norm(r) < tol: # check if residual is zero
            print("k =",k,np.linalg.norm(r))
            return x
        
        Mr = np.dot(M_inv, r)
        
        rMr_next_prod = np.dot(r.T, Mr)

        beta = (rMr_next_prod / rMr_cur_prod)[0, 0]

        p = Mr + beta * p

        rMr_cur_prod = rMr_next_prod

        k += 1

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
        p = np.dot(M_inv, r).copy()
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
            Mr = np.dot(M_inv, r)
            rr_next_prod = np.dot(r.T, Mr)
            beta = rr_next_prod / rr_cur_prod
            p = Mr + beta * p # next search direction
            
        else:
            rr_next_prod = np.dot(r.T, r)
            beta = rr_next_prod / rr_cur_prod
            p = r + beta * p # next search direction
            

        rr_cur_prod = rr_next_prod

        k += 1


def CGS(A,b,tol= np.pow(1/10,10)):
    
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point

    r = b - A.dot(x)
    q = np.zeros((size, 1))
    p = np.zeros((size, 1))
    rho_old = 1
    r_tilde = r.copy()
    k = 0
    while True:
        rho_new = r_tilde.T * r
        beta = rho_new/rho_old
        u = r +beta*q
        p = u + beta * (q + beta * p)
        v = A.dot(p)
        sigma = r_tilde * v
        alpha = rho_new/sigma
        
        q = u - alpha * v
        r = r - alpha * A.dot(u + q)
        if np.linalg.norm(r) < tol or k > size*5: # check if residual is zero
            print("k =", k, np.linalg.norm(r))
            return x
        x=x+alpha*(u+q)

        r_tilde = r
        rho_old = rho_new
        k+=1

def randAb(size, l, u, int = False):
    if int:
        A = rng.integers(l,u,size=(size, size))
        b = rng.integers(l,u,size=(size, 1))
    else:
        A = rng.random((size, size)) * (u - l) + l
        b = rng.random((size,1)) * (u - l) + l
    return A, b


if __name__ == "__main__":
    size = 1000
    A, b = randAb(size, -1, 1)
    A = np.matmul(A, A.T)
    M_inv = np.diag(1/np.diag(A))

    print(A)
    condA = np.linalg.cond(A)
    condMA = np.linalg.cond(np.dot(M_inv, A))
    print(size, condA, condMA)
    print()

    tol = np.pow(1/10,5)

    start = perf_counter()
    x = NewCG(A, b, tol = tol)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b))
    print()

    start = perf_counter()
    x = NewCG(A, b, M_inv, tol = tol)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b))
    print()

    # start = perf_counter()
    # x = CGS(A, b)
    # print("time", perf_counter() - start)
    # print("sol norm", np.linalg.norm(np.dot(A, x) - b))
    # print()