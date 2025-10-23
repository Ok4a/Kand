import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg._interface import LinearOperator
import scipy.sparse as spar

def Jacobi(A, invert = True, normalEq = False):
     
    A = np.conj(A.T) @ A  if normalEq else A
    if invert:
        return diags(1/A.diagonal())
    else:
        return diags(np.diag(A))
    

class Jacobi_class():
    def __init__(self, A):
        self.diag = 1/np.diag(A)
        self.size = np.size(A,0)

    def mv(self,v):
        return np.array([self.diag[i] * v[i] for i in range(self.size)])
  

def shuffle(size, seed = None):
    rng = np.random.default_rng(seed = seed)
    row = np.arange(size)
    col = np.arange(size)
    rng.shuffle(col)
    data = np.array([1]*size)

    coo = spar.coo_array((data, (row, col)), shape = (size, size))
    
    return coo, col


class shift_precondition():
    def __init__(self, A, rng=None):
        self.size = np.size(A,0)
        if rng is None or type(rng) == int:
            rng = np.random.default_rng(seed = rng)
        self.alpha = rng.normal(size=(self.size, 1))

    def mv(self, v):
        return np.array([v[i]+ v[(i+1)%self.size]*self.alpha[i] for i in range(self.size)])
    
    def Linear(self):
        return LinearOperator(shape=(self.size,self.size), matvec = self.mv)
    
def shift(A,size, rng = None):
    if rng is None or type(rng) == int:
        rng = np.random.default_rng(seed = rng)
    

    M = np.eye(size)
    M += np.diag(rng.normal(size = size - 1), k = 1)
    M += np.diag(rng.normal(size = 1), k = -size + 1)
    M += np.diag(rng.normal(size = size - 1), k = -1)
    M += np.diag(rng.normal(size = 1), k = size - 1)
    return M
    # return np.eye(size) + np.eye(size, k = 1) + np.eye(size, k = -size + 1)