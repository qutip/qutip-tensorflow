import qutip as qt
import numpy as np
import tensorflow as tf
import qutip.core.data as data
import numbers

__all__ = ['DenseTensor']

class DenseTensor(data.Data):
    def __init__(self, data, shape=None, copy=True):
        # Copy is False by default so that when creating a tensor the graph is
        # preserved. I would say that this is the default behaviour.

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
            and isinstance(shape, tuple)
        ):
            raise ValueError("shape must be a 2-tuple of positive ints, but is " + repr(shape))

        super().__init__(shape)

        # if not copy and isinstance(data, tf.Tensor):
            # self._tf = tf

        try:
            self._tf = tf.constant(data, shape=shape)
        except TypeError as e:
            raise ValueError("Shape of data must match shape argument.") from e

        self._tf = tf.cast(self._tf, tf.complex128)

        if copy:
            self._tf = tf.identity(self._tf) # Copy

    def copy(self):
        return DenseTensor(self._tf, shape = self.shape, copy=True)

    def to_array(self):
        return self._tf.numpy()

    def conj(self):
        return TensorDense(tf.math.conj(self._tf))

    def transpose(self):
        return TensorDense(tf.transpose(self._tf))

    def adjoint(self):
        return TensorDense(tf.linalg.adjoint(self._tf))

    def trace(self):
        return tf.linalg.trace(self._tf).numpy()


