import tensorflow as tf
from tensorflow.errors import InvalidArgumentError
import qutip
import numbers

__all__ = ["DenseTensor"]


class DenseTensor(qutip.core.data.Data):
    """This class provide a wraps around TensorFlow's Tensor. It will store data
    as a Tensor of dtype `tf.complex128`. Data will be expanded into a 2D
    Tensor.

    Parameters
    ----------
    data: array-like
        Data to be stored. Accepts same array-like as tensorflow.constant.
    shape: (int, int)
        Shape of data. Default `None`, it tries to infer the shape from data
        accessing the attribute `data.shape`.
    copy: bool
        If `True` (default) then the object is copied. Otherwise, copy will only
        be made if tf.constant returns a copy of data(when input is not a
        `Tenor`) or if a copy is needed to satisfy dtype (tf.complex128) and
        shape (2D Tensor)."""

    def __init__(self, data, shape=None, copy=False):
        # If the input is a tensor this does not copy it. Otherwise it will
        # return a copy.
        data = tf.constant(data)

        # If dtype of Tensor is already a tf.complex128 then this will not
        # return a copy
        data = tf.cast(data, tf.complex128)

        # Try to inherit shape from data and expand shape
        if shape is None:
            try:
                shape = data.shape
            except AttributeError:
                raise ValueError(
                    """Shape could not be inferred from data.
                                 Please, include the shape of data."""
                )

            if isinstance(shape, tf.TensorShape):
                shape = tuple(shape.as_list())

            if len(shape) == 0:
                shape = (1, 1)
                # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape[0], 1)

        if not (
            isinstance(shape, tuple)
            and len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError(
                """Shape must be a 2-tuple of positive ints, but
                             is """
                + repr(shape)
            )

        super().__init__(shape)

        # Only reshape when needed as reshape always returns a copy of the input
        # Tensor.
        if shape != tuple(data.shape.as_list()):
            try:
                data = tf.reshape(data, shape)
            # We return ValueError to match what qutip returns.
            except InvalidArgumentError as e:
                raise ValueError(
                    """Shape of data must
                                 match shape argument."""
                ) from e
        if copy:
            self._tf = tf.identity(data)
        else:
            self._tf = data

    def copy(self):
        return DenseTensor(self._tf, shape=self.shape, copy=True)

    def to_array(self):
        return self._tf.numpy()

    def conj(self):
        return DenseTensor(tf.math.conj(self._tf))

    def transpose(self):
        return DenseTensor(tf.transpose(self._tf))

    def adjoint(self):
        return DenseTensor(tf.linalg.adjoint(self._tf))

    def trace(self):
        return tf.linalg.trace(self._tf).numpy()
