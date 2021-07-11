import qutip
from .dense_tensor import DenseTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensorflow.errors import InvalidArgumentError

__all__ = ['add_DenseTensor', 'sub_DenseTensor']

# Conversion function
def add_DenseTensor(left, right, scale=1):
    if left.shape != right.shape:
        raise ValueError(f"""Incompatible shapes for adition of two matrices:
                         left={left.shape} and right={right.shape}""")
    return DenseTensor(left._tf + scale*right._tf, shape=left.shape, copy=False)

def sub_DenseTensor(left, right):
    if left.shape != right.shape:
        raise ValueError(f"""Incompatible shapes for adition of two matrices:
                         left={left.shape} and right={right.shape}""")
    return DenseTensor(left._tf - right._tf, shape=left.shape, copy=False)

# `add_conversions` will register the data layer
qutip.data.add.add_specialisations([
     (DenseTensor, DenseTensor, DenseTensor, add_DenseTensor),
])

qutip.data.sub.add_specialisations([
     (DenseTensor, DenseTensor, DenseTensor, sub_DenseTensor),
])
