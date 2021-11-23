import qutip
from .tftensor import TfTensor128, TfTensor64
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


__all__ = ["kron_tftensor"]


def kron_tftensor(left, right):
    return left._fast_constructor(
        tf.linalg.LinearOperatorKronecker(
            [
                tf.linalg.LinearOperatorFullMatrix(left._tf),
                tf.linalg.LinearOperatorFullMatrix(right._tf),
            ]
        ).to_dense(),
        shape=(left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]),
    )


qutip.data.kron.add_specialisations(
    [
        (TfTensor128, TfTensor128, TfTensor128, kron_tftensor),
        (TfTensor64, TfTensor64, TfTensor64, kron_tftensor),
    ]
)
