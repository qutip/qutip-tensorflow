import qutip
from .tftensor import TfTensor128, TfTensor64
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


def frobenius_tftensor(matrix):
    return tf.norm(matrix._tf)


def l2_tftensor(matrix):
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_tftensor(matrix)


def trace_tftensor(matrix):
    # For column and row vectors we simply take frobenius norm as they are
    # equivalent
    if matrix.shape[0] == 1 or matrix.shape[1] == 1:
        return frobenius_tftensor(matrix)

    out = matrix._tf
    out = tf.matmul(matrix._tf, matrix._tf, adjoint_b=True)
    out = tf.linalg.sqrtm(out)
    out = tf.linalg.trace(out)
    return tf.math.real(out)


def one_tftensor(matrix):
    return tf.norm(matrix._tf, ord=1, axis=[-2, -1])


def max_tftensor(matrix):
    return tf.math.reduce_max(tf.abs(matrix._tf))


qutip.data.norm.frobenius.add_specialisations(
    [
        (TfTensor128, frobenius_tftensor),
        (TfTensor64, frobenius_tftensor),
    ]
)

qutip.data.norm.l2.add_specialisations(
    [
        (TfTensor128, l2_tftensor),
        (TfTensor64, l2_tftensor),
    ]
)

qutip.data.norm.trace.add_specialisations(
    [
        (TfTensor128, trace_tftensor),
        (TfTensor64, trace_tftensor),
    ]
)

qutip.data.norm.max.add_specialisations(
    [
        (TfTensor128, max_tftensor),
        (TfTensor64, max_tftensor),
    ]
)

qutip.data.norm.one.add_specialisations(
    [
        (TfTensor128, one_tftensor),
        (TfTensor64, one_tftensor),
    ]
)
