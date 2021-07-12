import qutip
from .dense_tensor import DenseTensor
from .matmul import matmul_DenseTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

__all__ = ["expm_DenseTensor"]

def pow_DenseTensor(matrix, n):
    """Matrix power."""

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"""Trace can only be performed in square matrix. This
                         matrix has shape={matrix.shape}""")

    pow = matrix._tf

    I = tf.eye(10, dtype=tf.complex128)
    out = pow if n&1 else I

    n >>= 1
    while n:
        pow = matmul_DenseTensor(pow, pow)

        if n & 1:
            out = matmul_DenseTensor(out, pow)
        n >>= 1

    return DenseTensor._fast_constructor(out, shape=matrix.shape)


qutip.data.expm.add_specialisations(
    [
        (DenseTensor, DenseTensor, expm_DenseTensor),
    ]
)
