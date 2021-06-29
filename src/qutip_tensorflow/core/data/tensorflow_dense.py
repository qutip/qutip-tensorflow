import qutip as qt
import numpy as np
import tensorflow as tf
import qutip.core.data as data
import numbers

class DenseTensor(data.Data):
    def __init__(self, data, shape=None, copy=True):
        """We assume that """

        # Try to inherit shape from data
        if shape==None:
            try:
                shape = data.shape
            except AttributeError:
                raise ValueError("""Shape could not be inferred from data. Please,
                                 include the shape of data.""")

            if isinstance(shape, tf.TensorShape):
                shape = tuple(shape.as_list())

            if len(shape) == 0:
                shape = (1, 1)
                # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape[0], 1)


        if not (
            len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError("shape must be a 2-tuple of positive ints, but is " + repr(shape))

        super().__init__(shape)

        if copy:
            self._tf = tf.identity(data, shape=shape, dtype=tf.complex128)
        else:
            self._tf = tf.constant(data, shape=shape, dtype=tf.complex128)

    def copy(self):
        return TensorDense(self._tf, shape = self.shape, copy=True)

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

    def __add__(left, right):
        return TensorDense(left._tf + right._tf)

    def __mul__(left, right):
        return TensorDense(left._tf * right._tf)


