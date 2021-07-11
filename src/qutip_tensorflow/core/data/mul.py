import qutip
from .dense_tensor import DenseTensor

__all__ = ["mul_DenseTensor", "neg_DenseTensor"]


def mul_DenseTensor(matrix, value):
    """
    Performs the operation:
        ``out := value*matrix``
    where `value` is a complex scalar.
    """
    return DenseTensor._fast_constructor(matrix._tf * value, shape=matrix.shape)


def neg_DenseTensor(matrix):
    """
    Performs the operation:
        ``out := -matrix``
    """
    return DenseTensor._fast_constructor(-matrix._tf, shape=matrix.shape)


qutip.data.mul.add_specialisations([(DenseTensor, DenseTensor, mul_DenseTensor)])
qutip.data.neg.add_specialisations([(DenseTensor, DenseTensor, neg_DenseTensor)])
