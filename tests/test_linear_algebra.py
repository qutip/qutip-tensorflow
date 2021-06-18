import time
import pytest
import qutip as qt
import numpy as np
import tensorflow as tf
import scipy as sc
from numpy.testing import assert_almost_equal

size_max = 11
size_n = 11
size_list = np.logspace(1,size_max,size_n, base=2, dtype = int).tolist()
operations = ["__matmul__", "__add__"]
operation_ids = ["matmul", "add"]

def generate_matrix(size, density):
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        return np.diag(ofdiag, k=-1) + np.diag(diag, k=0) + np.diag(ofdiag, k=1)

    elif density=="dense":
        return np.random.random((size,size)) + 1j*np.random.random((size,size))


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


@pytest.mark.parametrize("dtype", [np, tf, sc, qt.data.Dense, qt.data.CSR],
                         ids=["numpy",
                              "tensorflow",
                              "scipy(sparse)",
                              "qt.data.Dense",
                              "qt.data.CSR"])
@pytest.mark.parametrize("operation", operations, ids=operation_ids)
@pytest.mark.parametrize("density", ["sparse", "dense"])
@pytest.mark.parametrize("size", size_list)
def test_linear_algebra(benchmark, dtype, size, operation, density, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = '-'.join(group[:3])
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    A = generate_matrix(size, density)
    A = change_dtype(A, dtype)
    operation = getattr(A, operation)

    result = benchmark(operation, A)

    return result



@pytest.mark.parametrize("dtype", [qt.data.Dense], ids=["qt.data.Dense2"])
@pytest.mark.parametrize("operation", operations, ids=operation_ids)
def test_linear_algebra_qtf(dtype, operation):
    """This is an example test to show how to test the new data layers using
    `test_linear_algebra`.

    TODO: qt.data.Dense needs to be replaced by qt.data.Dense_tf
    """
    density = "dense"
    size = 4

    # Create unitary
    A = generate_matrix(size, density)
    operation_np = getattr(A, operation)
    result_np = operation_np(A)

    A = change_dtype(A, dtype)
    operation = getattr(A, operation)
    result = operation(A)

    assert_almost_equal(result_np, result.full())

