import numpy as np
from scipy.sparse import diags
import scipy.sparse 

def Jacobi(A, invert = True) -> scipy.sparse._dia.dia_matrix:
   
    if invert:
        return diags(1/np.diag(A))
    else:
        return diags(np.diag(A))