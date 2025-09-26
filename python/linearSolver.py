import numpy as np
from time import perf_counter
import scipy.sparse.linalg as sp
rng = np.random.default_rng()

def _CG(A, b, x0 = None, M_inv = None, tol = np.pow(1/10, 10), normalEq = False):
    # if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
    #     raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
    
    # check if the function has been given a preconditioner
    isPrecond = False if M_inv is None else True

    A = np.conj(A.T) @ A  if normalEq else A
    b = np.conj(A.T).dot(b) if normalEq else b
    
    # dim of the system
    size = np.size(b)

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype=complex) if x0 is None else x0

    r = b - A.dot(x) # first residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x, r, 0
    
    p = (M_inv @ r).copy() if isPrecond else r.copy() # first search direction
    
    rho_prev = _inner(r, p)
    # iteration counter
    k = 0 
    while True:
        Ap_prod = A.dot(p) # saves one matrix vector prod
        alpha = rho_prev / (_inner(p,Ap_prod))

        x += alpha * p # next point

        r -= alpha * Ap_prod # next residual

        if np.linalg.norm(r) < tol: # stopping criteria
            return x, r, k
        
        # calc to make the next search direction conjugate (A-orthogonal) to the previous
        Mr = M_inv @ r if isPrecond else r
        rho_next = _inner(r, Mr)
        beta = rho_next / rho_prev
        p = Mr + beta * p # next search direction

        rho_prev = rho_next

        # if k % 100 == 0:
        #     print(k,np.linalg.norm(r))
        k += 1

def CG(A, b, x0 = None, M_inv = None, tol = np.pow(1/10, 10), verbose = False, normalEq = False):
    """function to run the CG function, 
    with possibility of printing extra info to the terminal with the verbose bool"""
    if verbose:
        print("Method: CG\nSystem dim:", np.size(b))
        if normalEq:
            print("Normal eq: True")
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r, k = _CG(A, b, x0, M_inv, tol, normalEq)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  np.linalg.norm(r))
        print("Sol norm:", np.linalg.norm(np.dot(A, x) - b),"\nAll close:", np.allclose(A.dot(x), b), "\n")
    return x

def _BiCGSTAB(A, b, x0 = None, M_inv = None, tol = np.pow(1/10, 10)):

    # check if the function has been given a preconditioner
    isPrecond = False if M_inv is None else True
    
    # dim of the system
    size = np.size(b)

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype=complex) if x0 is None else x0
    
    r = b - A.dot(x) # first residual
    
    r_tilde = r.copy()
    rho_prev = _inner(r_tilde, r)
    p = r.copy()

    k = 0 # iteration counter
    while True:
        p_hat = M_inv.dot(p) if isPrecond else p

        Ap_prod = A.dot(p_hat)

        alpha = rho_prev / _inner(r_tilde, Ap_prod)

        h = x + alpha * p_hat
        s = r - alpha * Ap_prod

        if np.linalg.norm(s) < tol: # stopping criteria
            return h, s, k
        
        s_hat = M_inv.dot(s) if isPrecond else s

        t = A.dot(s_hat)

        omega = _inner(t, s) / _inner(t, t)

        x = h + omega * s_hat
        r = s - omega * t

        if np.linalg.norm(r) < tol: # stopping criteria
            return x, r, k
        
        rho_next = _inner(r_tilde, r)
        beta = (rho_next / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * Ap_prod)

        rho_prev = rho_next

        k += 1

def BiCGSTAB(A, b, x0 = None, M_inv = None, tol = np.pow(1/10, 10), verbose = False):
    """function to run the BiCGSTAB function, 
    with possibility of printing extra info to the terminal with the verbose bool"""
    if verbose:
        print("Method: BiCGSTAB\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r, k = _BiCGSTAB(A, b,x0, M_inv, tol)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  np.linalg.norm(r))
        print("Sol norm:", np.linalg.norm(np.dot(A, x) - b),"\nAll close:", np.allclose(A.dot(x), b), "\n")
    return x


def _inner(x,y):
    return np.conj(x.T) @ y


def randAb(size, l, u):
    A = rng.random((size, size)) * (u - l) + l
    b = rng.random((size,1)) * (u - l) + l
    return A, b

def GenAb(size):
    A = np.diag(rng.integers(10, size = size)) - np.eye(size, k = -1) - np.eye(size, k = 1)
    b = rng.random((size,1))
    return A, b

if __name__ == "__main__":
    size = 1000
    # A, b = randAb(size, -1, 1)
    A, b = GenAb(size)
    A = np.matmul(A, A.T)

    M_inv = np.diag(1/np.diag(A))
    if size <= 1000:
        print(np.linalg.cond(A), np.linalg.cond(M_inv@A))
    print(size)
    print()

    print("CG")
    start = perf_counter()
    x = CG(A, b)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b),np.allclose(A.dot(x),b), "\n")

    print("Precond CG")
    start = perf_counter()
    x = CG(A, b, M_inv)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b),np.allclose(A.dot(x),b), "\n")

    print("BiCGSTAB")
    start = perf_counter()
    x = BiCGSTAB(A, b)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b),np.allclose(A.dot(x),b), "\n")

    print("Precond BiCGSTAB")
    start = perf_counter()
    x = BiCGSTAB(A, b, M_inv)
    print("time", perf_counter() - start)
    print("sol norm", np.linalg.norm(np.dot(A, x) - b),np.allclose(A.dot(x),b), "\n")

