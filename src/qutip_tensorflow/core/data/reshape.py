import qutip
from .tftensor import TfTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


__all__ = [
    "reshape_tftensor",
    "column_stack_tftensor",
    "column_unstack_tftensor",
    "split_columns_tftensor",
]


def _reshape_check_input(in_shape, out_shape):
    """Checks input and output shapes are compatible."""
    if out_shape[0] * out_shape[1] != in_shape[0] * in_shape[1]:
        raise ValueError(f"cannot reshape {in_shape} to {out_shape}.")

    if out_shape[0] < 1 or out_shape[1] < 1:
        raise ValueError("shape must be positive.")


def _column_unstack_shape_check(in_shape, rows):
    if in_shape[1] != 1:
        raise ValueError(f"input shape, {in_shape}, is not a single column")
    if rows < 1:
        raise ValueError(f"rows, {rows}, must be a positive integer")
    if in_shape[0] % rows:
        raise ValueError(
            f"""number of rows, {rows} does not divide into the
                         shape {in_shape}"""
        )


def reshape_tftensor(matrix, n_rows_out, n_cols_out):
    out_shape = (n_rows_out, n_cols_out)
    _reshape_check_input(matrix.shape, out_shape)
    return TfTensor._fast_constructor(
        tf.reshape(matrix._tf, out_shape), shape=out_shape
    )


def column_stack_tftensor(matrix):
    out_shape = (matrix.shape[0] * matrix.shape[1], 1)

    # We first need to transpose the input for the reshape to work
    out = tf.transpose(matrix._tf)
    return TfTensor._fast_constructor(
        tf.reshape(out, out_shape),
        shape=out_shape,
    )


def column_unstack_tftensor(matrix, rows):
    _column_unstack_shape_check(matrix.shape, rows)
    out_shape = (rows, matrix.shape[0] // rows)

    out = tf.reshape(matrix._tf, (matrix.shape[0] // rows, rows))
    out = tf.transpose(out)
    return TfTensor._fast_constructor(out, shape=out_shape)


def split_columns_tftensor(matrix, copy=True):
    return [TfTensor(matrix._tf[:, k], copy=copy) for k in range(matrix.shape[1])]


qutip.data.reshape.add_specialisations([(TfTensor, TfTensor, reshape_tftensor)])

qutip.data.column_unstack.add_specialisations(
    [(TfTensor, TfTensor, column_unstack_tftensor)]
)

qutip.data.column_stack.add_specialisations(
    [(TfTensor, TfTensor, column_stack_tftensor)]
)

qutip.data.split_columns.add_specialisations([(TfTensor, split_columns_tftensor)])
