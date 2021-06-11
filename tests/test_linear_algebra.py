import time
import pytest
import qutip as qt
import numpy as np
# import tensorflow as tf

# Maybe remove np to simplify code.
# Tensorflow shows deprecation warning for imp
@pytest.mark.parametrize(
    ["operation", "dtype"],
    [pytest.param(qt.Qobj.__matmul__, qt.data.Dense, id="qt-Dense-matmul"),
     pytest.param(qt.Qobj.__matmul__, qt.data.CSR, id="qt-CSR-matmul"),
     pytest.param(np.matmul, qt.data.Dense, id="np-Dense-matmul"),
     pytest.param(qt.Qobj.__add__, qt.data.Dense, id="qt-Dense-add"),
     pytest.param(qt.Qobj.__add__, qt.data.CSR, id="qt-CSR-add"),
     pytest.param(np.add, qt.data.Dense, id="np-Dense-add"),
     # Disabled as it caused due to imp deprecation error
     # pytest.param(tf.matmul, qt.data.Dense, id="np-Dense-matmul"),
    ]
)
@pytest.mark.parametrize("density", [0.1,1], ids=[ "density=0.1", "density=1"])
@pytest.mark.parametrize("size", np.logspace(1,2,3, dtype = int).tolist())
def test_linear_algebra(benchmark, size, operation, density, dtype, request):
    # Group benchmark by operationa and density
    group = request.node.callspec.id
    group = group.split('-')
    group = group[-1] + '-' + group[1]
    benchmark.group = group


    # Create unitary
    U = qt.rand_unitary(size, density=density,
                       dtype = dtype)

    # If numpy method convert array to narray (__array__ method does not work
    # well).
    if operation.__class__ == np.ufunc:
        U = U.full()

    result = benchmark(operation, U, U)

    # Ensure results are correct
    assert True
