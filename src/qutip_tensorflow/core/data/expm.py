import qutip
from .tftensor import TfTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

__all__ = ["expm_tftensor"]


def expm_tftensor(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"""Expm can only be performed in square matrix. This
                         matrix has shape={matrix.shape}"""
        )
    return TfTensor._fast_constructor(tf.linalg.expm(matrix._tf), shape=matrix.shape)


qutip.data.expm.add_specialisations(
    [
        (TfTensor, TfTensor, expm_tftensor),
    ]
)
