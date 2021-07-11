import qutip
from .dense_tensor import DenseTensor
import warnings

__all__ = ["transpose_DenseTensor", "conj_DenseTensor", "adjoint_DenseTensor"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


def transpose_DenseTensor(matrix):
    return DenseTensor._fast_constructor(
        tf.transpose(matrix._tf), shape=(matrix.shape[1], matrix.shape[0])
    )


def conj_DenseTensor(matrix):
    return DenseTensor._fast_constructor(tf.math.conj(matrix._tf), shape=matrix.shape)


def adjoint_DenseTensor(matrix):
    return DenseTensor._fast_constructor(
        tf.linalg.adjoint(matrix._tf), shape=(matrix.shape[1], matrix.shape[0])
    )


qutip.data.transpose.add_specialisations(
    [
        (DenseTensor, DenseTensor, transpose_DenseTensor),
    ]
)

qutip.data.conj.add_specialisations(
    [
        (DenseTensor, DenseTensor, conj_DenseTensor),
    ]
)
qutip.data.adjoint.add_specialisations(
    [
        (DenseTensor, DenseTensor, adjoint_DenseTensor),
    ]
)
