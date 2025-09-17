import numpy as np
from time import perf_counter
import scipy.sparse.linalg as sp
rng = np.random.default_rng(seed=1)

def CG(A, b, precond = None, tol = np.pow(1/10, 10)):
    if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
        raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))

    isPrecond = False if precond is None else True
    
    size = np.size(b)
    x = np.zeros((size, 1)) # initial starting point
    
    r = b - A.dot(x) # residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x
    
    p = (precond @ r).copy() if isPrecond else r.copy()
    
    rho_prev = np.dot(r.T, p)

    k = 0
    while True:
        Ap_prod = A.dot(p) # saves one matrix vector prod
        alpha = rho_prev / (p.T @ Ap_prod)

        x += alpha * p # next point

        r -= alpha * Ap_prod # next residual

        if np.linalg.norm(r) < tol: # check if residual is zero
            print("k =", k, np.linalg.norm(r))
            return x
        
        Mr = precond @ r if isPrecond else r
        rho_next = r.T @ Mr
        beta = rho_next / rho_prev
        p = Mr + beta * p # next search direction

        rho_prev = rho_next

        k += 1


def BiCGSTAB(A, b, M_inv = None, tol = np.pow(1/10, 10)):

    isPrecond = False if M_inv is None else True
    size = np.size(b)
    x = np.zeros((size, 1))
    
    r = b - A.dot(x) 
    r_tilde = r.copy()
    rho_prev = r_tilde.T @ r
    p = r.copy()
    k = 0
    while True:
        p_hat = M_inv.dot(p) if isPrecond else p

        Ap_prod = A.dot(p_hat)

        alpha = rho_prev / (r_tilde.T @ Ap_prod)

        h = x + alpha * p_hat
        s = r - alpha * Ap_prod

        if np.linalg.norm(s) < tol:
            print("k =", k, np.linalg.norm(s))
            return h
        
        s_hat = M_inv.dot(s) if isPrecond else s

        t = A.dot(s_hat)

        omega = (t.T @ s) / (t.T @ t)
        x = h + omega * s_hat
        r = s - omega * t

        if np.linalg.norm(r) < tol:
            print("k =", k, np.linalg.norm(r))
            return x
        rho_next = r_tilde.T @ r
        beta = (rho_next / rho_prev) * (alpha  / omega)
        p = r + beta * (p - omega * Ap_prod)

        rho_prev = rho_next

        k+=1


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
    x = CG(A, b)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b), np.allclose(A@x, b))
    print()

    start = perf_counter()
    x = CG(A, b, M_inv)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b), np.allclose(A@x, b))
    print()

    start = perf_counter()
    x = BiCGSTAB(A, b)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b), np.allclose(A@x, b))
    print()

    start = perf_counter()
    x = BiCGSTAB(A, b, M_inv)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b), np.allclose(A@x, b))
    print()
