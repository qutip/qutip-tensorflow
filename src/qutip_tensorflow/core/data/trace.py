import qutip
from .tftensor import TfTensor128, TfTensor64
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

__all__ = ["trace_tftensor"]


def trace_tftensor(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"""Trace can only be performed in square matrix. This
                         matrix has shape={matrix.shape}"""
        )
    return tf.linalg.trace(matrix._tf)


qutip.data.trace.add_specialisations(
    [
        (TfTensor128, trace_tftensor),
        (TfTensor64, trace_tftensor),
    ]
)
