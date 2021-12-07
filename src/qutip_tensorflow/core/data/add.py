import qutip
from .tftensor import TfTensor128, TfTensor64
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensorflow.errors import InvalidArgumentError

__all__ = ["add_tftensor", "sub_tftensor", "iadd_tftensor"]


def _check_shape(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"""Incompatible shapes for addition of two matrices:
                         left={left.shape} and right={right.shape}"""
        )


def add_tftensor(left, right, scale=1):
    _check_shape(left, right)

    # If scale=1 we obtain a x2 speed-up if we do not multiply by the scale.
    if scale == 1:
        return left._fast_constructor(left._tf + right._tf, shape=left.shape)
    else:
        return left._fast_constructor(left._tf + scale * right._tf, shape=left.shape)


def iadd_tftensor(left, right, scale=1):
    """This function performs an in-place addition. However, TensorFlow returns
    a new object after a mathematical operation. This means that in-place here
    only serves to avoid the creation of a TfTensor instance. We do not have
    any control over the memory where the Tensor is stored."""
    _check_shape(left, right)

    # If scale=1 we obtain a x2 speed-up if we do not multiply by the scale.
    if scale == 1:
        left._tf = left._tf + right._tf
    else:
        left._tf = left._tf + scale * right._tf

    return left


def sub_tftensor(left, right):
    _check_shape(left, right)
    return left._fast_constructor(left._tf - right._tf, shape=left.shape)


# `add_conversions` will register the data layer
qutip.data.add.add_specialisations(
    [
        (TfTensor128, TfTensor128, TfTensor128, add_tftensor),
        (TfTensor64, TfTensor64, TfTensor64, add_tftensor),
    ]
)

qutip.data.sub.add_specialisations(
    [
        (TfTensor128, TfTensor128, TfTensor128, sub_tftensor),
        (TfTensor64, TfTensor64, TfTensor64, sub_tftensor),
    ]
)
