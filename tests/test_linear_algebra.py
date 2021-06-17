import time
import pytest
import qutip as qt
import numpy as np
import tensorflow as tf

size_max = 11
size_n = 10
size_list = np.logspace(1,size_max,size_n, base=2, dtype = int).tolist()

def generate_matrix(size, density):
    np.random.seed(1)

    if density == "sparse":
        ofdiag = np.random.rand(size-1) + 1j*np.random.rand(size-1)
        diag = np.random.rand(size) + 1j*np.random.rand(size)

        return np.diag(ofdiag, k=-1) + np.diag(diag, k=0) + np.diag(ofdiag, k=1)

    elif density=="dense":
        return np.random.random((size,size)) + 1j*np.random.random((size,size))

@pytest.mark.parametrize("dtype", [qt.data.Dense, qt.data.CSR],
                         ids=["qt.data.Dense","qt.data.CSR"])
@pytest.mark.parametrize("operation", [qt.Qobj.__matmul__, qt.Qobj.__add__],
                         ids=["matmul", "add"])
@pytest.mark.parametrize("density", ["sparse","dense"], ids=[ "density=0.1", "density=1"])
@pytest.mark.parametrize("size", size_list)
def test_linear_algebra_qt(benchmark, operation, size, density, dtype, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = group[-2] + '-' + group[1] + '-' + group[0]
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    A = generate_matrix(size, density)
    A = qt.Qobj(A)
    A = A.to(dtype)

    result = benchmark(operation, A, A)

    # Ensure results are correct
    assert True

def change_dtype(A, dtype):
    "chnages a numpy matrix to tensorflow or scipy sparse"



@pytest.mark.parametrize("dtype", [np, tf], ids=["numpy", "tensorflow"])
@pytest.mark.parametrize("operation", ["matmul", "add"], ids=["matmul", "add"])
@pytest.mark.parametrize("density", ["sparse", "dense"], ids=[ "density=0.1", "density=1"])
@pytest.mark.parametrize("size", size_list)
def test_linear_algebra_ex(benchmark, dtype, size, operation, density, request):
    # Group benchmark by operationa, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = group[-2] + '-' + group[1] + '-' + group[0]
    benchmark.extra_info['dtype'] = group[-1]

    # Get operation
    if operation=="matmul":
        operation = dtype.matmul

    if operation=="add":
        operation = dtype.add

    # Create unitary
    A = generate_matrix(size, density)

    result = benchmark(operation, A, A)

    # Ensure results are correct
    assert True # TODO

