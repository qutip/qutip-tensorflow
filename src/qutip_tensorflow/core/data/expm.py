import qutip
from .dense_tensor import DenseTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

__all__ = ["expm_DenseTensor"]

def expm_DenseTensor(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"""Trace can only be performed in square matrix. This
                         matrix has shape={matrix.shape}""")
    return DenseTensor._fast_constructor(tf.linalg.expm(matrix._tf),
                                        shape=matrix.shape)


qutip.data.expm.add_specialisations(
    [
        (DenseTensor, DenseTensor, expm_DenseTensor),
    ]
)
