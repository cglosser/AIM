import numpy as np
from numpy.fft import fft, ifft

def symmetric_toeplitz_circulant(toep_vec):
    """Given the characteristic vector of a symmetric Toeplitz matrix,
    construct the characteristic vector of the expanded circulant matrix.
    """
    stamp_vec = np.array([0] + [i for i in toep_vec[:0:-1]])
    return np.hstack((toep_vec, stamp_vec))

def fast_toeplitz_product(toep_vec, vec):
    """Given the first row/column of a symmetric Toeplit matrix, compute
    a fast fft-based matrix-vector product.
    """
    vec = np.hstack((vec, np.zeros(len(vec))))
    auxillary_vec = symmetric_toeplitz_circulant(toep_vec)
    result = np.real(ifft(fft(auxillary_vec)*fft(vec))[:len(vec)/2])
    return result

def full_toeplitz_product(block_cols, toep, rvec):
    result = np.zeros(block_cols**2)
    for row in range(0, block_cols**2, block_cols):
        for block in range(0, block_cols):
            b_start, b_stop = block*block_cols, (block + 1)*block_cols

            result[row:row + block_cols] += fast_toeplitz_product(
                    toep[row, b_start:b_stop], rvec[b_start:b_stop]
            )
    return result
