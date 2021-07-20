import qutip
from .tftensor import TfTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensorflow import matmul


__all__ = ["matmul_tftensor"]


def _check_shape(left, right, out):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape) + " and " + str(right.shape)
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )


def matmul_tftensor(left, right, scale=1, out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1. If `out` not gives is assumed to be 0.
    """
    _check_shape(left, right, out)
    shape = (left.shape[0], right.shape[1])

    if scale == 1:
        result = matmul(left._tf, right._tf)
    else:
        result = matmul(scale * left._tf, right._tf)

    if out is None:
        return TfTensor._fast_constructor(result, shape=shape)
    else:
        out._tf = result + out._tf


qutip.data.matmul.add_specialisations(
    [(TfTensor, TfTensor, TfTensor, matmul_tftensor)]
)
