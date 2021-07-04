import qutip as qt
from .tensorflow_dense import DenseTensor

__all__ = ['add', 'add_DenseTensor', 'sub', 'sub_DenseTensor']

# Conversion function
def add_DenseTensor(left, right, scale=1):
    return DenseTensor(left._tf + scale*right._tf, copy=False)

def sub_DenseTensor(left, right):
    return DenseTensor(left._tf - right._tf, copy=False)

# `add_conversions` will register the data layer
qt.data.add.add_specialisations([
     (DenseTensor, DenseTensor, DenseTensor, add_DenseTensor),
])

qt.data.sub.add_specialisations([
     (DenseTensor, DenseTensor, DenseTensor, sub_DenseTensor),
])

add = qt.data.add
sub = qt.data.sub
