import numpy as np
from time import perf_counter
import scipy.sparse.linalg as sp
from util import *

def _CG(A, b, x0 = None, M_inv = None, tol = 1e-10, normal_eq = False, maxIter = None):
    # if not normal_eq:
    #     if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
    #         raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
        
    # check if the function has been given a preconditioner
    isPrecond = False if M_inv is None else True


    r_norm = []
    
    # dim of the system
    size = np.size(b)

    b = adj(A) @ b if normal_eq else b
    if normal_eq:
        A_tilde = adj(A) @ A

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype = complex) if x0 is None else x0

    r = b.copy() if x0 is None else b - A.dot(x) # first residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x, r, 0
    
    p = (M_inv @ r).copy() if isPrecond else r.copy() # first search direction
    
    rho_prev = inner(r, p)




    maxIter = np.inf if maxIter is None else maxIter
    # iteration counter
    k = 0 
    while k < maxIter:
        Ap_prod = A.dot(p) # saves one matrix vector prod
        denorm = inner(Ap_prod, Ap_prod) if normal_eq else inner(p, Ap_prod)
        alpha = rho_prev / denorm

        x += alpha * p # next point

        r -= alpha * A_tilde.dot(p) if normal_eq else alpha * Ap_prod # next residual
        r_norm.append(np.linalg.norm(r))
        if r_norm[-1] < tol: # stopping criteria
            break
        
        # calc to make the next search direction conjugate (A-orthogonal) to the previous
        Mr = M_inv @ r if isPrecond else r
        rho_next = inner(r, Mr)
        
        beta = rho_next / rho_prev
        p = Mr + beta * p # next search direction

        rho_prev = rho_next

        if (k+1) % 5000 == 0:
            print(k, np.linalg.norm(r), end="\r")
        k += 1
    return x, r_norm, k

def CG(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, normal_eq = False, maxIter = None):
    """
    function to run the CG function, 
    with possibility of printing extra info to the terminal with the verbose bool
    """
    if verbose:
        print("\nMethod: CG\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r_norm, k = _CG(A=A, b=b, x0=x0, M_inv=M_inv, tol=tol, normal_eq=normal_eq, maxIter= maxIter)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  r_norm[-1])
        print("Sol norm:", np.linalg.norm(np.dot(A.toarray(), x) - b),"\nAll close:", "true!!!!!!!!!!!!!" if np.allclose(A.dot(x), b) else "False", "\n")
    return x, r_norm, k


def _BiCGSTAB(A, b, x0 = None, M_inv = None, tol = 1e-10, maxIter = None):

    # check if the function has been given a preconditioner
    isPrecond = False if M_inv is None else True

    
    
    # dim of the system
    size = np.size(b)

    # list to norms of residuals for comparison
    r_norm = []
    s_norm = []

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype = complex) if x0 is None else x0
    
    r = b - A.dot(x) # first residual
    
    r_tilde = r.copy()
    rho_prev = inner(r_tilde, r)
    p = r.copy()

    maxIter = np.inf if maxIter is None else maxIter

    k = 0 # iteration counter
    while k < maxIter:

        p_hat = M_inv.dot(p) if isPrecond else p

        Ap_prod = A.dot(p_hat)

        temp = inner(r_tilde, Ap_prod)
        if temp == 0:
            flag = 2
            break
        alpha = rho_prev / temp
        x = x + alpha * p_hat # h
        r = r - alpha * Ap_prod # residual 2 (s)
        s_norm.append(np.linalg.norm(r))
        if s_norm[-1] < tol: # stopping criteria
            # x = x
            flag = 0
            break

        s_hat = M_inv.dot(r) if isPrecond else r

        t = A.dot(s_hat)

        omega = inner(t, r) / inner(t, t)

        x = x + omega * s_hat
        r = r - omega * t # residual 1 (r)
        r_norm.append(np.linalg.norm(r))
        if r_norm[-1] < tol: # stopping criteria
            flag = 0
            break
            # return x, r_norm, k

        rho_next = inner(r_tilde, r)
        if np.abs(rho_next) == 0 or omega == 0:
            flag = 2
            # print(rho_next)
            break
        beta = (rho_next / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * Ap_prod)

        rho_prev = rho_next

        k += 1
    else:
        flag = 1
    return x, (r_norm, s_norm), k, flag

def BiCGSTAB(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, maxIter = None):
    """function to run the BiCGSTAB function, 
    with possibility of printing extra info to the terminal with the verbose bool"""
    if verbose:
        print("Method: BiCGSTAB\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r_norm, k, flag = _BiCGSTAB(A, b, x0, M_inv = M_inv, tol = tol, maxIter = maxIter)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  r_norm[0][-1],"\nResidual norm:",  r_norm[1][-1])
        print("Sol norm:", np.linalg.norm(np.dot(A.toarray(), x)- b),"\nAll close:", np.allclose(A.dot(x), b))
        print("Flag:", flag, "\n")
    return x, r_norm, k, flag



def _BiCGSTABl(A, b, x0 = None, M_inv = None, tol = 1e-10, maxIter = None):

    size = np.size(b)
    x = np.zeros((size, 1), dtype = complex) if x0 is None else x0

    r_norm=[]
    r = b - A.dot(x) # first residual
    r_hat = r.copy()
    u = np.zeros(shape=(size,1))
    rho_0 = 1
    alpha = 0
    omega_2 = 1

    k = 0

    maxIter = np.inf if maxIter is None else maxIter
    while k < maxIter:
        print(k, end="\r")
        rho_0 = -omega_2*rho_0

        rho_1 = inner(r_hat,r)
        beta = (alpha*rho_1)/rho_0
        rho_0 = rho_1
        u = r-beta*u
        v= A@u
        gamma = inner(v,r_hat)
        alpha = rho_0/gamma
        r = r -alpha*v
        s = A@r
        x += alpha*u

        rho_1=inner(r_hat,s)
        beta=(alpha*rho_1)/rho_0
        rho_0=rho_1
        v = s-beta*v
        w = A@v
        gamma = inner(w, r_hat)
        alpha = rho_0/gamma
        u = r-beta* u
        r = r-alpha*u
        s= s-alpha * w
        t= A@s

        omega_1 = inner(r,s)
        mu = inner(s,s)
        nu = inner(s,t)
        tau = inner(t,t)
        omega_2 = inner(r,t)
        tau = tau -(nu**2)/mu

        omega_2 = (omega_2-(nu*omega_1)/mu)/tau
        omega_1 = (omega_1-nu*omega_2)/mu
        
        x = x+ omega_1*r+omega_2*s+alpha*u
        r = r-omega_1*s-omega_2*t
        r_norm.append(np.linalg.norm(r))
        if r_norm[-1]< tol:
            break
        u = u-omega_1*v-omega_2*w
        k+=2
    return x, r_norm,k

def BiCGSTABl(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, maxIter = None):
    """function to run the BiCGSTAB function, 
    with possibility of printing extra info to the terminal with the verbose bool"""
    if verbose:
        print("Method: BiCGSTAB\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r_norm, k = _BiCGSTABl(A, b, x0, M_inv = M_inv, tol = tol, maxIter = maxIter)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  r_norm[-1])
        print("Sol norm:", np.linalg.norm(np.dot(A.toarray(), x)- b),"\nAll close:", np.allclose(A.dot(x), b), "\n")
    return x, r_norm, k

