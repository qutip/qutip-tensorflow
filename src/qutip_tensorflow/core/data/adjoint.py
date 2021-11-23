import qutip
from .tftensor import TfTensor128, TfTensor64
import warnings

__all__ = ["transpose_tftensor", "conj_tftensor", "adjoint_tftensor"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


def transpose_tftensor(matrix):
    return matrix._fast_constructor(
        tf.transpose(matrix._tf), shape=(matrix.shape[1], matrix.shape[0])
    )


def conj_tftensor(matrix):
    return matrix._fast_constructor(
        tf.math.conj(matrix._tf), shape=matrix.shape
    )


def adjoint_tftensor(matrix):
    return matrix._fast_constructor(
        tf.linalg.adjoint(matrix._tf), shape=(matrix.shape[1], matrix.shape[0])
    )


qutip.data.transpose.add_specialisations(
    [
        (TfTensor128, TfTensor128, transpose_tftensor),
        (TfTensor64, TfTensor64, transpose_tftensor),
    ]
)

qutip.data.conj.add_specialisations(
    [
        (TfTensor128, TfTensor128, conj_tftensor),
        (TfTensor64, TfTensor64, conj_tftensor),
    ]
)
qutip.data.adjoint.add_specialisations(
    [
        (TfTensor128, TfTensor128, adjoint_tftensor),
        (TfTensor64, TfTensor64, adjoint_tftensor),
    ]
)
