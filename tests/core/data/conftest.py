import warnings

import numpy as np
import scipy.sparse
import qutip

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


def random_numpy_dense(shape, fortran):
    """Generate a random numpy dense matrix with the given shape."""
    out = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    if fortran:
        out = np.asfortranarray(out)
    return out

def random_tensor_dense(shape):
    """Generate a random numpy dense matrix with the given shape."""
    out = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    out = tf.constant(out)
    return out
