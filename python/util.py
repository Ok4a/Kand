import numpy as np
rng = np.random.default_rng()
def inner(x,y):
    return np.conj(x.T) @ y


def randAb(size, l = 0, u = 1, normal = False):
    A = rng.random((size, size)) * (u - l) + l
    b = np.ones((size, 1))
    if normal:
        return np.conj(A.T) @ A, np.conj(A.T) @ np.ones((size, 1))
    else:
        return A, b

def GenAb(size):
    A = np.diag(rng.integers(10, size = size)) - np.eye(size, k = -1) - np.eye(size, k = 1)
    b = rng.random((size,1))
    return A, b
