import qutip
from .tftensor import TfTensor128, TfTensor64

__all__ = ["mul_tftensor", "neg_tftensor", "imul_tftensor"]


def imul_tftensor(matrix, value):
    """
    Performs the operation:
        ``out := value*matrix``
    where `value` is a complex scalar.
    """
    matrix._tf = matrix._tf * value
    return matrix


def mul_tftensor(matrix, value):
    """
    Performs the operation:
        ``out := value*matrix``
    where `value` is a complex scalar.
    """
    return matrix._fast_constructor(matrix._tf * value, shape=matrix.shape)


def neg_tftensor(matrix):
    """
    Performs the operation:
        ``out := -matrix``
    """
    return matrix._fast_constructor(-matrix._tf, shape=matrix.shape)


qutip.data.mul.add_specialisations([(TfTensor128, TfTensor128, mul_tftensor),
                                   (TfTensor64, TfTensor64, mul_tftensor)])
qutip.data.imul.add_specialisations([(TfTensor128, TfTensor128, imul_tftensor),
                                    (TfTensor64, TfTensor64, imul_tftensor)])
qutip.data.neg.add_specialisations([(TfTensor128, TfTensor128, neg_tftensor),
                                   (TfTensor64, TfTensor64, neg_tftensor)])
