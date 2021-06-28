import qutip as qt
import numpy as np
import tensorflow as tf
import qutip.core.data as data


class TensorflowDense(data.Data):
    def __init__(self, data, shape=None, copy=True):
        # Try to inherit shape from data
        if shape==None:
            shape = data.shape

        # TensorFlow uses its own shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()

        if isinstance(shape, list):
            shape = tuple(shape)

        self._tf = tf.constant(data, shape=shape, dtype=tf.complex128)

        if not len(shape) == 2:
            raise ValueError

        super().__init__(shape)

    def copy(self):
        return TensorflowDense(tf.identity(self._tf))

    def to_array(self):
        return self._tf.numpy()

    def conj(self):
        return TensorflowDense(tf.math.conj(self._tf))

    def transpose(self):
        return TensorflowDense(tf.transpose(self._tf))

    def adjoint(self):
        return TensorflowDense(tf.linalg.adjoint(self._tf))

    # TODO: for auto differentiation it may be necessary to return a tensor
    def trace(self):
        return tf.linalg.trace(self._tf).numpy()

    # operator that is nice to have but not necessary since dispached functions
    # are used by Qobj.
    def __add__(left, right):
        return TensorflowDense(left._tf + right._tf)
