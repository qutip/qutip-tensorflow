import qutip
from .dense_tensor import DenseTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

__all__ = ["trace_DenseTensor"]

def trace_DenseTensor(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"""Trace can only be performed in square matrix. This
                         matrix has shape={matrix.shape}""")
    return tf.linalg.trace(matrix._tf).numpy()


qutip.data.trace.add_specialisations(
    [
        (DenseTensor, trace_DenseTensor),
    ]
)
