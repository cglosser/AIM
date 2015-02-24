import numpy as np
from numpy.fft import fft, ifft

def symmetric_toeplitz_circulant(toep):
    """Given the characteristic vector of a symmetric Toeplitz matrix,
    construct the characteristic vector of the expanded circulant matrix.
    """
    stamp_vec = np.array([0] + [i for i in toep[:0:-1]])
    return np.hstack((toep, stamp_vec))

def fast_toeplitz_product(toep, vec):
    """Given the first row/column of a symmetric Toeplit matrix, compute
    a fast fft-based matrix-vector product.
    """
    vec = np.hstack((vec, np.zeros(len(vec))))
    auxillary_vec = symmetric_toeplitz_circulant(toep)
    result = ifft(fft(auxillary_vec)*fft(vec))[:len(vec)] #delete "stamp"
    return np.around(result)
