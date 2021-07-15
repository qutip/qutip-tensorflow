import qutip
from .tftensor import TfTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensorflow.errors import InvalidArgumentError

__all__ = ["add_tftensor", "sub_tftensor"]

# Conversion function
def add_tftensor(left, right, scale=1):
    if left.shape != right.shape:
        raise ValueError(
            f"""Incompatible shapes for addition of two matrices:
                         left={left.shape} and right={right.shape}"""
        )

    # If scale=1 we obtain a x2 speed-up if we do not multiply by the scale.
    if scale == 1:
        return TfTensor._fast_constructor(left._tf + right._tf, shape=left.shape)
    else:
        return TfTensor._fast_constructor(
            left._tf + scale * right._tf, shape=left.shape
        )


def sub_tftensor(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"""Incompatible shapes for addition of two matrices:
                         left={left.shape} and right={right.shape}"""
        )
    return TfTensor(left._tf - right._tf, shape=left.shape, copy=False)


# `add_conversions` will register the data layer
qutip.data.add.add_specialisations(
    [
        (TfTensor, TfTensor, TfTensor, add_tftensor),
    ]
)

qutip.data.sub.add_specialisations(
    [
        (TfTensor, TfTensor, TfTensor, sub_tftensor),
    ]
)
