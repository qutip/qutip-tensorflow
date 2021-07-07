"""This file contains the unary operations that will be benchmarked. All the functions
in this file should return the function that will be tested. The tested funtion must
have as inputs:

    A: input 2D array

    dtype: data type of A (np, tf, qt.data.Dense ... )

    rep: number of times that the operations will be repeated.
The function does not need to return anything else. The getters have as input parameters
only dtype. If the getter returns a NotImplementedError it will be omitted in the
benchmarks.
    """
import warnings

import numpy as np
import scipy as sc
import qutip as qt

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

def get_expm(dtype):
    if dtype == np:
        op = sc.linalg.expm
    elif dtype == tf:
        op = tf.linalg.expm
    elif dtype == sc:
        op = sc.sparse.linalg.expm
    elif issubclass(dtype, qt.data.base.Data):
        op = qt.Qobj.expm

    def expm(A, dtype, rep):
        for _ in range(1):
            x = op(A)

        # synchronize GPU
        if dtype == tf:
            _ = x.numpy()

        return x

    return expm


def get_eigenvalues(dtype):
    if dtype == np:
        op = np.linalg.eigvalsh
    elif dtype == tf:
        op = tf.linalg.eigvalsh
    elif dtype == sc:
        # Omit in benchmarks
        raise NotImplementedError
    elif issubclass(dtype, qt.data.base.Data):
        op = qt.Qobj.eigenenergies

    def eigenvalues(A, dtype, rep):
        for _ in range(1):
            x = op(A)

        # synchronize GPU
        if dtype == tf:
            _ = x.numpy()

        return x
    return eigenvalues
