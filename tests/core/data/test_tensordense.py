import numpy as np
from numpy.testing import assert_almost_equal
import tensorflow as tf
import pytest

from qutip.core import data
from qutip.core.data import dense
from qutip_tensorflow.core.data import DenseTensor

from . import conftest


# Set up some fixtures for automatic parametrisation.

@pytest.fixture(params=[
    pytest.param((1, 5), id='ket'),
    pytest.param((5, 1), id='bra'),
    pytest.param((5, 5), id='square'),
    pytest.param((2, 4), id='wide'),
    pytest.param((4, 2), id='tall'),
])
def shape(request): return request.param


@pytest.fixture(params=[True, False], ids=['Fortran', 'C'])
def fortran(request): return request.param


def _valid_numpy():
    # Arbitrary valid numpy array.
    return conftest.random_numpy_dense((5, 5), False)

def _valid_tensor():
    # Arbitrary valid tensor array.
    return conftest.random_tensor_dense((5, 5))

def _valid_list():
    # Arbitrary valid tensor array.
    return conftest.random_numpy_dense((5, 5), False).tolist()


@pytest.fixture(scope='function')
def numpy_dense(shape, fortran):
    return conftest.random_numpy_dense(shape, fortran)

@pytest.fixture(scope='function')
def tensor_dense(shape):
    return conftest.random_tensor_dense(shape)


@pytest.fixture(scope='function')
def list_dense(shape):
    return conftest.random_numpy_dense(shape, fortran).tolist()

@pytest.fixture(scope='function')
def data_tensor_dense(tensor_dense):
    return DenseTensor(tensor_dense)


class TestClassMethods:
    def test_init_from_list(self, list_dense, shape):
        test = DenseTensor(list_dense, shape)
        assert test.shape == shape
        assert np.all(test.to_array() == np.array(list_dense))

    def test_init_from_ndarray(self, numpy_dense):
        test = DenseTensor(numpy_dense)
        assert test.shape == numpy_dense.shape
        assert np.all(test.to_array() == numpy_dense)

    def test_init_from_tensor(self, tensor_dense):
        test = DenseTensor(tensor_dense)
        assert test.shape == tuple(tensor_dense.shape.as_list())
        assert np.all(test.to_array() == tensor_dense)

        # by default we do not return a copy
        assert test._tf is tensor_dense

    @pytest.mark.parametrize('dtype', ['complex128',
                                       'float64',
                                       'int32', 'int64',
                                       'uint32'])
    def test_init_from_list_other_dtype(self, shape, dtype):
        _numpy_dense = np.random.rand(*shape).astype(dtype, casting='unsafe')
        _list_dense = _numpy_dense.tolist()
        test = DenseTensor(_list_dense)
        assert test.shape == shape
        assert test._tf.dtype == tf.complex128
        assert test._tf.shape == shape
        assert_almost_equal(test.to_array(), _list_dense)

    @pytest.mark.parametrize('dtype', ['complex128',
                                       'float64',
                                       'int32', 'int64',
                                       'uint32'])
    def test_init_from_ndarray_other_dtype(self, shape, dtype):
        _numpy_dense = np.random.rand(*shape).astype(dtype, casting='unsafe')
        test = DenseTensor(_numpy_dense)
        assert test.shape == shape
        assert test._tf.dtype == tf.complex128
        assert test._tf.shape == shape
        assert np.all(test.to_array() == _numpy_dense)

    @pytest.mark.parametrize('dtype', ['complex128',
                                       'float64',
                                       'int32', 'int64',
                                       'uint32'])
    def test_init_from_tensor_other_dtype(self, shape, dtype):
        numpy_dense = np.random.rand(*shape).astype(dtype, casting='unsafe')
        tensor = tf.constant(numpy_dense)
        test = DenseTensor(tensor)
        assert test.shape == shape
        assert test._tf.shape == shape
        assert test._tf.dtype == tf.complex128

        tensor = tf.cast(tensor, dtype=tf.complex128)
        assert np.all(test._tf == tensor)

    @pytest.mark.parametrize(['arg', 'kwargs', 'error'], [
        pytest.param(_valid_tensor(), {'shape': ()}, ValueError,
                     id="numpy-shape 0 tuple"),
        pytest.param(_valid_tensor(), {'shape': (1,)}, ValueError,
                     id="numpy-shape 1 tuple"),
        pytest.param(_valid_tensor(), {'shape': (None, None)}, ValueError,
                     id="numpy-shape None tuple"),
        pytest.param(_valid_tensor(), {'shape': [2, 2]}, ValueError,
                     id="numpy-shape list"),
        pytest.param(_valid_tensor(), {'shape': (1, 2, 3)}, ValueError,
                     id="numpy-shape 3 tuple"),
        pytest.param(_valid_tensor(), {'shape': (-1, 1)}, ValueError,
                     id="numpy-negative shape"),
        pytest.param(_valid_tensor(), {'shape': (-4, -4)}, ValueError,
                     id="numpy-both negative shape"),
        pytest.param(_valid_tensor(), {'shape': (1213, 1217)}, ValueError,
                     id="numpy-different shape"),
    ])
    def test_init_from_wrong_input(self, arg, kwargs, error):
        """
        Test that the __init__ method raises a suitable error when passed
        incorrectly formatted inputs.

        This test also serves as a *partial* check that Dense safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            DenseTensor(arg, **kwargs)

    @pytest.mark.parametrize(['arg', 'kwargs', 'error'], [
        pytest.param(_valid_numpy(), {'shape': ()}, ValueError,
                     id="numpy-shape 0 tuple"),
        pytest.param(_valid_numpy(), {'shape': (1,)}, ValueError,
                     id="numpy-shape 1 tuple"),
        pytest.param(_valid_numpy(), {'shape': (None, None)}, ValueError,
                     id="numpy-shape None tuple"),
        pytest.param(_valid_numpy(), {'shape': [2, 2]}, ValueError,
                     id="numpy-shape list"),
        pytest.param(_valid_numpy(), {'shape': (1, 2, 3)}, ValueError,
                     id="numpy-shape 3 tuple"),
        pytest.param(_valid_numpy(), {'shape': (-1, 1)}, ValueError,
                     id="numpy-negative shape"),
        pytest.param(_valid_numpy(), {'shape': (-4, -4)}, ValueError,
                     id="numpy-both negative shape"),
        pytest.param(_valid_numpy(), {'shape': (1213, 1217)}, ValueError,
                     id="numpy-different shape"),
        pytest.param(_valid_list(), {'shape': ()}, ValueError,
                     id="numpy-shape 0 tuple"),
        pytest.param(_valid_list(), {'shape': (1,)}, ValueError,
                     id="numpy-shape 1 tuple"),
        pytest.param(_valid_list(), {'shape': (None, None)}, ValueError,
                     id="numpy-shape None tuple"),
        pytest.param(_valid_list(), {'shape': [2, 2]}, ValueError,
                     id="numpy-shape list"),
        pytest.param(_valid_list(), {'shape': (1, 2, 3)}, ValueError,
                     id="numpy-shape 3 tuple"),
        pytest.param(_valid_list(), {'shape': (-1, 1)}, ValueError,
                     id="numpy-negative shape"),
        pytest.param(_valid_list(), {'shape': (-4, -4)}, ValueError,
                     id="numpy-both negative shape"),
        pytest.param(_valid_list(), {'shape': (1213, 1217)}, ValueError,
                     id="numpy-different shape"),
        pytest.param(_valid_tensor(), {'shape': ()}, ValueError,
                     id="numpy-shape 0 tuple"),
        pytest.param(_valid_tensor(), {'shape': (1,)}, ValueError,
                     id="numpy-shape 1 tuple"),
        pytest.param(_valid_tensor(), {'shape': (None, None)}, ValueError,
                     id="numpy-shape None tuple"),
        pytest.param(_valid_tensor(), {'shape': [2, 2]}, ValueError,
                     id="numpy-shape list"),
        pytest.param(_valid_tensor(), {'shape': (1, 2, 3)}, ValueError,
                     id="numpy-shape 3 tuple"),
        pytest.param(_valid_tensor(), {'shape': (-1, 1)}, ValueError,
                     id="numpy-negative shape"),
        pytest.param(_valid_tensor(), {'shape': (-4, -4)}, ValueError,
                     id="numpy-both negative shape"),
        pytest.param(_valid_tensor(), {'shape': (1213, 1217)}, ValueError,
                     id="numpy-different shape"),
    ])
    def test_init_from_wrong_input(self, arg, kwargs, error):
        """
        Test that the __init__ method raises a suitable error when passed
        incorrectly formatted inputs.

        This test also serves as a *partial* check that Dense safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            DenseTensor(arg, **kwargs)

    def test_copy_returns_a_correct_copy(self, data_tensor_dense):
        """
        Test that the copy() method produces an actual copy, and that the
        result represents the same matrix.
        """
        original = data_tensor_dense
        copy = data_tensor_dense.copy()
        assert original is not copy
        assert np.all(original._tf == copy._tf)
        assert original._tf is not copy._tf


    def test_to_array_is_correct_result(self, data_tensor_dense):
        test_array = data_tensor_dense.to_array()
        assert isinstance(test_array, np.ndarray)
        assert test_array.ndim == 2
        assert test_array.shape == data_tensor_dense.shape

        tensor = tf.constant(test_array)
        assert np.all(test_array == data_tensor_dense._tf)

