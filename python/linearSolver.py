import numpy as np
from time import perf_counter
import scipy.sparse.linalg as sp

from util import *

def _CG(A, b, x0 = None, M_inv = None, tol = 1e-10, normal_eq = False):
    # if not normal_eq:
    #     if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
    #         raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
        
    # check if the function has been given a preconditioner
    isPrecond = False if M_inv is None else True

    
    # dim of the system
    size = np.size(b)

    b = np.conj(A.T) @ b if normal_eq else b
    if normal_eq:
        A_tilde = np.conj(A.T) @ A

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype = complex) if x0 is None else x0

    r = b.copy() if x0 is None else b - A.dot(x) # first residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x, r, 0
    
    p = (M_inv @ r).copy() if isPrecond else r.copy() # first search direction
    
    rho_prev = inner(r, p)
    # iteration counter
    k = 0 
    while k < 2001:
        Ap_prod = A.dot(p) # saves one matrix vector prod
        denorm = inner(Ap_prod, Ap_prod) if normal_eq else inner(p, Ap_prod)
        alpha = rho_prev / denorm

        x += alpha * p # next point

        r -= alpha * A_tilde.dot(p) if normal_eq else alpha * Ap_prod # next residual

        if np.linalg.norm(r) < tol: # stopping criteria
            break
        
        # calc to make the next search direction conjugate (A-orthogonal) to the previous
        Mr = M_inv @ r if isPrecond else r
        rho_next = inner(r, Mr)
        
        beta = rho_next / rho_prev
        p = Mr + beta * p # next search direction

        rho_prev = rho_next

        if (k) % 5000 == 0:
            print(k, np.linalg.norm(r), end="\r")
        k += 1
    else:
        print("Not converged")
    return x, r, k

def CG(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, normal_eq = False):
    """
    function to run the CG function, 
    with possibility of printing extra info to the terminal with the verbose bool
    """
    if verbose:
        print("\nMethod: CG\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r, k = _CG(A=A, b=b, x0=x0, M_inv=M_inv, tol=tol, normal_eq=normal_eq)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  np.linalg.norm(r))
        print("Sol norm:", np.linalg.norm(np.dot(A.toarray(), x) - b),"\nAll close:", np.allclose(A.dot(x), b), "\n")
    return x, r, k







def _CGNew(A, b, x0 = None, M_inv = None, tol = 1e-10, normal_eq = False):


    # A = A.T@A
    # b = A.T@b

    dim = np.size(b)

    x = np.zeros((dim,1)) if x0 is None else x0

    r = b - A @ x 

    k = 1
    while True:
        z = M_inv@r
        rho_new = inner(r,z)
        if k == 1:
            p = z.copy()
        else:
            beta = rho_new/rho_old
            p = z + beta * p
        q = A@p
        alpha = rho_new / (inner(p,q))
        x = x + alpha* p
        r = r - alpha*q
        r_norm =np.linalg.norm(r)
        if r_norm < tol:
            break
        if k %100 == 0:
            print(k, r_norm)
        rho_old = rho_new
        k +=1
    return x, r, k

def CGNew(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, normal_eq = False):
    """
    function to run the CG function, 
    with possibility of printing extra info to the terminal with the verbose bool
    """
    if verbose:
        print("\nMethod: CG New\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r, k = _CGNew(A=A, b=b, x0=x0, M_inv=M_inv, tol=tol, normal_eq=normal_eq)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  np.linalg.norm(r))
        print("Sol norm:", np.linalg.norm(np.dot(A.toarray(), x) - b),"\nAll close:", np.allclose(A.dot(x), b), "\n")
    return x, r, k



# if __name__ == "__main__":
#     size = 1000
#     # A, b = randAb(size, -1, 1)
#     A, b = GenAb(size)
#     A = np.matmul(A, A.T)

#     M_inv = np.diag(1/np.diag(A))
#     if size <= 1000:
#         print(np.linalg.cond(A), np.linalg.cond(M_inv@A))
#     print(size)
#     print()

#     print("CG")
#     start = perf_counter()
#     x = CG(A, b)
#     print("time", perf_counter() - start)
#     print("sol norm", np.linalg.norm(np.dot(A, x) - b),np.allclose(A.dot(x),b), "\n")

#     print("Precond CG")
#     start = perf_counter()
#     x = CG(A, b, M_inv)
#     print("time", perf_counter() - start)
#     print("sol norm", np.linalg.norm(np.dot(A, x) - b),np.allclose(A.dot(x),b), "\n")

