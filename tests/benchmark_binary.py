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

def get_matmul(dtype):
    def matmul(A, B, dtype, rep):
        for _ in range(rep):
            x = A@B

        # synchronize GPU
        if dtype == tf:
            _ = x.numpy()

        return x

    return matmul


def get_add(dtype):
    def add(A, B, dtype, rep):
        for _ in range(rep):
            x = A+B

        # synchronize GPU
        if dtype == tf:
            _ = x.numpy()

        return x
    return add

