import qutip
from .tftensor import TfTensor128, TfTensor64
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


__all__ = ["inner_tftensor", "inner_op_tftensor"]


def _check_shape_inner(left, right):
    if (
        (left.shape[0] != 1 and left.shape[1] != 1)  # Check left shape has 1
        or right.shape[1] != 1  # Check right shape has 1
        # Check left and right shapes are compatible
        or (left.shape[0] != right.shape[0] and left.shape[1] != right.shape[0])
    ):
        raise ValueError(
            "Incompatible matrix shapes for states: left "
            + str(left.shape)
            + " and right "
            + str(right.shape)
        )


def _check_shape_inner_op(left, op, right):
    # conditions for inner still apply
    _check_shape_inner(left, right)
    if (
        op.shape[0] != op.shape[1]  # Not an square matrix
        or op.shape[1] != right.shape[0]  # Not valid op@right
    ):
        raise ValueError(
            "Incompatible matrix shapes for op "
            + str(op.shape)
            + " and right"
            + str(right.shape)
        )


def inner_tftensor(left, right, scalar_is_ket=False):
    _check_shape_inner(left, right)
    left_is_scalar = left.shape[0] == left.shape[1] == 1
    left_is_ket = (
        not left_is_scalar and left.shape[1] == 1 or (left_is_scalar and scalar_is_ket)
    )
    if left_is_ket:
        return tf.reduce_sum(tf.math.conj(left._tf) * right._tf)
    else:
        return tf.reshape(left._tf @ right._tf, shape=())


def inner_op_tftensor(left, op, right, scalar_is_ket=False):
    _check_shape_inner_op(left, op, right)

    left_is_scalar = left.shape[0] == left.shape[1] == 1
    left_is_ket = (
        not left_is_scalar and left.shape[1] == 1 or (left_is_scalar and scalar_is_ket)
    )

    ket = op._tf @ right._tf

    if left_is_ket:
        return tf.reduce_sum(tf.math.conj(left._tf) * ket)
    else:
        return tf.reshape(left._tf @ ket, shape=())


qutip.data.inner.add_specialisations(
    [
        (TfTensor128, TfTensor128, inner_tftensor),
        (TfTensor64, TfTensor64, inner_tftensor),
    ]
)

qutip.data.inner_op.add_specialisations(
    [
        (TfTensor128, TfTensor128, TfTensor128, inner_op_tftensor),
        (TfTensor64, TfTensor64, TfTensor64, inner_op_tftensor),
    ]
)
