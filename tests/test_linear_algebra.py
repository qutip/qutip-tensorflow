import time
import pytest
import qutip as qt
import numpy as np
import scipy as sc
import scipy.sparse
from numpy.testing import assert_almost_equal
import warnings
from . import benchmark_unary
from . import benchmark_binary


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

# Get functions from unary ops that stater with `get`
unary_ops = [ getattr(benchmark_unary,_) for _ in dir(benchmark_unary) if _[:3]=="get"]
unary_ids = [ _[4:] for _ in dir(benchmark_unary) if _[:3]=="get"]

binary_ops = [ getattr(benchmark_binary,_) for _ in dir(benchmark_binary) if _[:3]=="get"]
binary_ids = [ _[4:] for _ in dir(benchmark_binary) if _[:3]=="get"]


@pytest.fixture(params = np.logspace(1, 10, 10, base=2, dtype=int).tolist())
def size(request): return request.param

@pytest.fixture(params = ["dense", "sparse"])
def density(request): return request.param

@pytest.fixture(scope='function')
def matrix(size, density):
    """Return a random matrix of size `sizexsize'. Density is either 'dense'
    or 'sparse' and returns a fully dense or a tridiagonal matrix respectively.
    The matrices are Hermitian."""
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        return (np.diag(ofdiag, k=-1)
                + np.diag(diag, k=0)
                + np.diag(ofdiag.conj(), k=1))

    elif density == "dense":
        H = np.random.random((size, size)) + 1j*np.random.random((size, size))
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

#Supported dtypes
dtype_list = [np, tf, sc, qt.data.Dense, qt.data.CSR]
dtype_ids = ['numpy', 'tensorflow', 'scipy-CSR', 'qutip-Dense', 'qutip-CSR']
@pytest.fixture(params = dtype_list, ids=dtype_ids)
def dtype(request): return request.param


@pytest.mark.parametrize("get_operation", unary_ops, ids=unary_ids)
def test_linear_algebra_unary(benchmark, matrix, dtype, get_operation, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = '-'.join(group[1:])
    benchmark.extra_info['dtype'] = group[0]

    # Create unitary
    A = matrix
    A = change_dtype(A, dtype)

    # Benchmark operations and skip those that are not implemented.
    try:
        operation = get_operation(dtype)
        result = benchmark(operation, A, dtype, 100)
    except (NotImplementedError):
        result = None

    return result

@pytest.mark.parametrize("get_operation", binary_ops, ids=binary_ids)
def test_linear_algebra_binary(benchmark, matrix, dtype, get_operation, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = '-'.join(group[1:])
    benchmark.extra_info['dtype'] = group[0]

    matrix = change_dtype(matrix, dtype)

    # Benchmark operations and skip those that are not implemented.
    try:
        operation = get_operation(dtype)
        result = benchmark(operation, matrix, matrix, dtype, 100)
    except (NotImplementedError):
        result = None

    return result
