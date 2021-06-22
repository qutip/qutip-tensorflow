import time
import pytest
import qutip as qt
import numpy as np
import scipy as sc
import scipy.sparse
from numpy.testing import assert_almost_equal
import warnings
import tensorflow as tf


size_max = 10
size_n = 10
size_list = np.logspace(1,size_max,size_n, base=2, dtype = int).tolist()

def generate_matrix(size, density):
    """Generate a random matrix of size `sizexsize'. Density is either 'dense'
    or 'sparse' and returns a fully dense or a tridiagonal matrix respectively.
    The matrices are Hermitian."""
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        return (np.diag(ofdiag, k=-1)
                + np.diag(diag, k=0)
                + np.diag(ofdiag.conj(), k=1))

    elif density=="dense":
        H = np.random.random((size,size)) + 1j*np.random.random((size,size))
        return H + H.T.conj()


def change_dtype(A, dtype):
    """Changes a numpy matrix to tensorflow, scipy sparse or to a qutip.Data
    specified by dtype"""
    if dtype == np:
        return A
    elif dtype == tf:
        return tf.convert_to_tensor(A)
    elif dtype == sc:
        return sc.sparse.csr_matrix(A)
    elif issubclass(dtype, qt.data.base.Data):
        A = qt.Qobj(A)
        return A.to(dtype)

# Define operations using always these four input parameters.
def get_matmul(dtype):
    def matmul(A, dtype, rep):
        for _ in range(rep):
            x = A@A

        # synchronize GPU
        if dtype == tf:
            _ = x.numpy()

        return x

    return matmul

def get_add(dtype):
    def add(A, dtype, rep):
        for _ in range(rep):
            x = A+A

        # synchronize GPU
        if dtype == tf:
            _ = x.numpy()

        return x
    return add

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
        op = np.linalg.eigvals
    elif dtype == tf:
        op = tf.linalg.eigvals
    elif dtype == sc:
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

@pytest.mark.parametrize("dtype", [np, tf, sc, qt.data.Dense, qt.data.CSR],
                         ids=["numpy",
                              "tensorflow",
                              "scipy(sparse)",
                              "qt.data.Dense",
                              "qt.data.CSR"])
@pytest.mark.parametrize("get_operation",
                         [get_matmul,
                          get_add,
                          get_expm,
                          get_eigenvalues],
                         ids=["matmul",
                             "add",
                             "expm",
                             "eigvals",
                             ])
@pytest.mark.parametrize("density", ["sparse", "dense"])
@pytest.mark.parametrize("size", size_list)
def test_linear_algebra(benchmark, dtype, size, get_operation, density, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = '-'.join(group[:3])
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    A = generate_matrix(size, density)
    A = change_dtype(A, dtype)

    try:
        operation = get_operation(dtype)
        result = benchmark(operation, A, dtype, 100)
    except (NotImplementedError):
        result = None

    return result
