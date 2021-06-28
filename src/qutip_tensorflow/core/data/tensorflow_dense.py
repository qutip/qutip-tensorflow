import qutip as qt
import numpy as np
import tensorflow as tf
import qutip.core.data as data

class TensorDense(data.Data):
    def __init__(self, data, shape=None, copy=True, initialize_tensor = True):

        # Try to inherit shape from data
        if shape==None:
            shape = data.shape

        # TensorFlow uses its own shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()

        if isinstance(shape, list):
            shape = tuple(shape)

        # TODO: if tensors should always have len(shape) == 2, this should be
        # check in super, shouldnt it?
        if not len(shape) == 2:
            raise ValueError

        super().__init__(shape)

        # TODO: This is created and the overridend if using VariableDense, maybe not
        # create if not necessary. 
        if initialize_tensor:
            self._tf = tf.constant(data, shape=shape, dtype=tf.complex128)


    def copy(self):
        return TensorDense(tf.identity(self._tf))

    def to_array(self):
        return self._tf.numpy()

    def conj(self):
        return TensorDense(tf.math.conj(self._tf))

    def transpose(self):
        return TensorDense(tf.transpose(self._tf))

    def adjoint(self):
        return TensorDense(tf.linalg.adjoint(self._tf))

    # TODO: for auto differentiation it may be necessary to return a tensor
    def trace(self):
        return tf.linalg.trace(self._tf).numpy()

    # operator that is nice to have but not necessary since dispached functions
    # are used by Qobj.
    def __add__(left, right):
        return TensorDense(left._tf + right._tf)

class VariableDense(TensorDense):
    def __init__(self, data, shape=None, copy=True):

        super().__init__(data, shape, copy, initialize_tensor=False)

        self._tf = tf.Variable(data, shape=shape, dtype=tf.complex128)


    def copy(self):
        return VariableDense(tf.identity(self._tf))



