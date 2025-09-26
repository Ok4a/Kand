import numpy as np
from scipy.sparse import diags
import scipy.sparse 

def Jacobi(A, invert = True, normalEq = False) -> scipy.sparse._dia.dia_matrix:
     
    A = np.conj(A.T) @ A  if normalEq else A
    if invert:
        return diags(1/np.diag(A))
    else:
        return diags(np.diag(A))