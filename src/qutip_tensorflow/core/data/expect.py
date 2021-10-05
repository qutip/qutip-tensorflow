import qutip
from math import sqrt
from .tftensor import TfTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


__all__ = ["expect_tftensor", "expect_super_tftensor"]


def _check_shape_expect(op, state):
    if (
        op.shape[0] != op.shape[1]  # Op square matrix.
        or not (
            state.shape[0] == state.shape[1]  # state is square matrix
            or state.shape[1] == 1  # state is ket
        )
        or op.shape[1] != state.shape[0]  # Not possible op@state
    ):
        raise ValueError(
            f"incompatible matrix shapes for op {op.shape} and state {state.shape}"
        )


def _check_shape_super(op, state):
    if state.shape[1] != 1:
        raise ValueError(
            "expected a column-stacked matrix but state input matrix "
            f"has shape {state.shape}"
        )
    if op.shape[1] != state.shape[0]:
        raise ValueError(f"incompatible shapes op:{op.shape} and state:{state.shape}")
    if op.shape[0] != op.shape[1]:
        raise ValueError(
            "Expect_super only works for square op matrices. The "
            f"provided op has shape: {op.shape}"
        )


def expect_ket_tftensor(op, state):
    return tf.reduce_sum(tf.math.conj(state._tf) * (op._tf @ state._tf))


def expect_dm_tftensor(op, state):
    return tf.reduce_sum(op._tf * tf.transpose(state._tf))


def expect_tftensor(op, state):
    _check_shape_expect(op, state)
    is_ket = state.shape[1] == 1

    if is_ket:
        return expect_ket_tftensor(op, state)
    else:
        return expect_dm_tftensor(op, state)


def expect_super_tftensor(op, state):
    _check_shape_super(op, state)

    out_shape = int(sqrt(op.shape[0]))

    op = tf.reshape(op._tf, shape=(out_shape, out_shape, out_shape, out_shape))
    state = tf.reshape(state._tf, shape=(out_shape, out_shape))

    return tf.einsum("iijk,jk->", op, state)


qutip.data.expect.add_specialisations([(TfTensor, TfTensor, expect_tftensor)])
qutip.data.expect_super.add_specialisations(
    [(TfTensor, TfTensor, expect_super_tftensor)]
)
