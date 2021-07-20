import qutip
from .tftensor import TfTensor

__all__ = ["mul_tftensor", "neg_tftensor"]


def mul_tftensor(matrix, value):
    """
    Performs the operation:
        ``out := value*matrix``
    where `value` is a complex scalar.
    """
    return TfTensor._fast_constructor(matrix._tf * value, shape=matrix.shape)


def neg_tftensor(matrix):
    """
    Performs the operation:
        ``out := -matrix``
    """
    return TfTensor._fast_constructor(-matrix._tf, shape=matrix.shape)


qutip.data.mul.add_specialisations([(TfTensor, TfTensor, mul_tftensor)])
qutip.data.neg.add_specialisations([(TfTensor, TfTensor, neg_tftensor)])
