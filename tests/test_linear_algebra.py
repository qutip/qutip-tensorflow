import time
import pytest
import qutip as qt
import numpy as np
import tensorflow as tf
import scipy as sc

size_max = 11
size_n = 15
size_list = np.logspace(1,size_max,size_n, base=2, dtype = int).tolist()

def generate_matrix(size, density):
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        return np.diag(ofdiag, k=-1) + np.diag(diag, k=0) + np.diag(ofdiag, k=1)

    elif density=="dense":
        return np.random.random((size,size)) + 1j*np.random.random((size,size))

def change_dtype(A, dtype):
    "chnages a numpy matrix to tensorflow or scipy sparse"
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
                         ids=["numpy", "tensorflow", "scipy(sparse)",
                              "qt.data.Dense", "qt.data.CSR"])
@pytest.mark.parametrize("operation", ["__matmul__", "__add__"],
                         ids=["matmul", "add"])
@pytest.mark.parametrize("density", ["sparse", "dense"],
                         ids=["sparse(tridiagonal)", "dense"])
@pytest.mark.parametrize("size", size_list)
def test_linear_algebra_ex(benchmark, dtype, size, operation, density, request):
    # Group benchmark by operationa, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = group[-2] + '-' + group[1] + '-' + group[0]
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    A = generate_matrix(size, density)
    A = change_dtype(A, dtype)
    operation = getattr(A, operation)

    result = benchmark(operation, A)

    # Ensure results are correct
    assert True

