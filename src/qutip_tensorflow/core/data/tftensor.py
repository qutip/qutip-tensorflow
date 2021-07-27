import tensorflow as tf
from tensorflow.errors import InvalidArgumentError
import qutip
import numbers

__all__ = ["TfTensor"]


class TfTensor(qutip.core.data.Data):
    """This class provide a wraps around TensorFlow's Tensor. It will store data
    as a Tensor of dtype ``tensorflow.complex128``. Data will be expanded into a 2D
    Tensor.

    Parameters
    ----------
    data: array-like
        Data to be stored. Accepts same array-like as ``tensorflow.constant``.
    shape: (int, int)
        Shape of data. Default `None`, it tries to infer the shape from data
        accessing the attribute `data.shape`.
    copy: bool
        Default ``False``. If ``True`` then the object is copied. Otherwise, a
        copy will only be made if tf.constant returns a copy of data (when input
        is not a `Tenor`) or if a copy is needed to satisfy dtype
        (tensorflow.complex128) and shape (2D Tensor)."""

    def __init__(self, data, shape=None, copy=False):
        # If the input is a tensor this does not copy it. Otherwise it will
        # return a copy.
        data = tf.constant(data)

        # If dtype of Tensor is already a tf.complex128 then this will not
        # return a copy
        data = tf.cast(data, tf.complex128)

        # Inherit shape from data and expand shape
        if shape is None:
            shape = tuple(data.shape.as_list())

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
                """Shape must be a 2-tuple of positive ints, but is """ + repr(shape)
            )

        super().__init__(shape)

        # Only reshape when needed as reshape always returns a copy of the input
        # Tensor.
        if shape != tuple(data.shape.as_list()):
            try:
                data = tf.reshape(data, shape)
            # We return ValueError to match what qutip returns.
            except InvalidArgumentError as e:
                raise ValueError("""Shape of data must match shape argument.""") from e

        if copy:
            self._tf = tf.identity(data)
        else:
            self._tf = data

    def copy(self):
        return TfTensor(self._tf, shape=self.shape, copy=True)

    def to_array(self):
        return self._tf.numpy()

    def conj(self):
        return TfTensor(tf.math.conj(self._tf))

    def transpose(self):
        return TfTensor(tf.transpose(self._tf))

    def adjoint(self):
        return TfTensor(tf.linalg.adjoint(self._tf))

    def trace(self):
        return tf.linalg.trace(self._tf).numpy()

    @classmethod
    def _fast_constructor(cls, data, shape):
        """
        A fast low-level constructor for wrapping an existing Tensor array in a
        TfTensor object without copying it. The ``data`` argument must be a
        Tensor array with the correct shape.
        """
        out = cls.__new__(cls)
        super(cls, out).__init__(shape)
        out._tf = data
        return out
