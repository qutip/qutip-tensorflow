import time
import pytest
import qutip as qt
import numpy as np
import tensorflow as tf

size_max = 10
size_n = 3

@pytest.mark.parametrize("dtype", [qt.data.Dense, qt.data.CSR], ids=["qt.data.Dense",
                                                                     "qt.data.CSR"])
@pytest.mark.parametrize("operation", [qt.Qobj.__matmul__, qt.Qobj.__add__], ids=[
    "matmul", "add"])
@pytest.mark.parametrize("density", [0.1,1], ids=[ "density=0.1", "density=1"])
@pytest.mark.parametrize("size", np.logspace(1,size_max,size_n, base=2, dtype = int).tolist())
def test_linear_algebra_qt(benchmark, size, operation, density, dtype, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = group[-2] + '-' + group[1] + '-' + group[0]
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    U = qt.rand_unitary(size, density=density,
                       dtype = dtype)

    result = benchmark(operation, U, U)

    # Ensure results are correct
    assert True

@pytest.mark.parametrize("dtype", [np], ids=["numpy"])
@pytest.mark.parametrize("operation", [np.matmul, np.add], ids=["matmul", "add"])
@pytest.mark.parametrize("density", [0.1,1], ids=[ "density=0.1", "density=1"])
@pytest.mark.parametrize("size", np.logspace(1,size_max,size_n, base=2, dtype = int).tolist())
def test_linear_algebra_np(benchmark, dtype, size, operation, density, request):
    # Group benchmark by operationa, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = group[-2] + '-' + group[1] + '-' + group[0]
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    U = qt.rand_unitary(size, density=density).full()

    result = benchmark(operation, U, U)

    # Ensure results are correct
    assert True # TODO

@pytest.mark.parametrize("dtype", [tf], ids=["tensorflow"])
@pytest.mark.parametrize("operation", [tf.matmul, tf.add], ids=["matmul", "add"])
@pytest.mark.parametrize("density", [0.1,1], ids=[ "density=0.1", "density=1"])
@pytest.mark.parametrize("size", np.logspace(1,size_max,size_n, base=2, dtype = int).tolist())
def test_linear_algebra_tf(benchmark, dtype, size, operation, density, request):
    # Group benchmark by operationa, density and size.
    group = request.node.callspec.id
    group = group.split('-')
    benchmark.group = group[-2] + '-' + group[1] + '-' + group[0]
    benchmark.extra_info['dtype'] = group[-1]

    # Create unitary
    U = qt.rand_unitary(size, density=density).full()

    result = benchmark(operation, U, U)

    # Ensure results are correct
    assert True # TODO
