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
            f"incompatible matrix shapes for op {op.shape}" " and state {state.shape}"
        )


def _check_shape_super(op, state):
    if state.shape[1] != 1:
        raise ValueError(
            "expected a column-stacked matrix but state input matrix"
            + f"has shape {state.shape}"
        )
    if op.shape[1] != state.shape[0]:
        raise ValueError(f"incompatible shapes op:{op.shape} and state:{state.shape}")
    if op.shape[0] != op.shape[1]:
        raise ValueError(
            "Expect_super only works for square op matrices. The"
            f"provided op has shape: {op.shape}"
        )


def expect_ket_tftensor(op, state):
    return tf.reduce_sum(tf.math.conj(state._tf) * (op._tf @ state._tf))


def expect_dm_tftensor(op, state):
    """In order to perform this operation with O(N^2) multiplications/sums, we
    reshape op into a batch of row vectors and state into a batch of column
    vectors. In this way matrix multiplication only computes diagonal elements."""
    shape = op.shape[0]  # op and state are square matrices

    op = tf.reshape(op._tf, (shape, 1, shape))  # Row vectors.

    # For state we first need to transpose such that reshape is done properly.
    state = tf.transpose(state._tf)
    state = tf.reshape(state, (shape, shape, 1))

    # The trace is just the sum of the computed elements
    return tf.reduce_sum(op @ state)


def expect_tftensor(op, state):
    _check_shape_expect(op, state)
    is_ket = state.shape[1] == 1

    if is_ket:
        return expect_ket_tftensor(op, state)
    else:
        return expect_dm_tftensor(op, state)


def expect_super_tftensor(op, state):
    """This is a naive implementation of expect_super that takes O(N^4)
    operations. A more efficient implementation with O(N^3) operations is
    possible."""
    _check_shape_super(op, state)

    out = op._tf @ state._tf

    out_shape = int(sqrt(op.shape[0]))
    out = tf.reshape(out, shape=(out_shape, out_shape))

    return tf.linalg.trace(out)


qutip.data.expect.add_specialisations([(TfTensor, TfTensor, expect_tftensor)])
qutip.data.expect_super.add_specialisations(
    [(TfTensor, TfTensor, expect_super_tftensor)]
)
