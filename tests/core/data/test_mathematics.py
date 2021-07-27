import itertools
import numpy as np
import pytest

from qutip_tensorflow.core.data import TfTensor
from qutip_tensorflow import data
from qutip.core.data import Data, Dense, CSR
import qutip.tests.core.data.test_mathematics as testing


from . import conftest

testing._ALL_CASES = {
    TfTensor: lambda shape: [lambda: conftest.random_tftensor(shape)],
}
testing._RANDOM = {TfTensor: lambda shape: [lambda: conftest.random_tftensor(shape)]}


class TestAdd(testing.TestAdd):
    specialisations = [
        pytest.param(data.add_tftensor, TfTensor, TfTensor, TfTensor),
        pytest.param(data.iadd_tftensor, TfTensor, TfTensor, TfTensor),
    ]


class TestSub(testing.TestSub):
    specialisations = [
        pytest.param(data.sub_tftensor, TfTensor, TfTensor, TfTensor),
    ]


class TestMul(testing.TestMul):
    specialisations = [
        pytest.param(data.mul_tftensor, TfTensor, TfTensor),
        pytest.param(data.imul_tftensor, TfTensor, TfTensor),
    ]


class TestMatmul(testing.TestMatmul):
    specialisations = [
        pytest.param(data.matmul_tftensor, TfTensor, TfTensor, TfTensor),
    ]


class TestNeg(testing.TestNeg):
    specialisations = [
        pytest.param(data.neg_tftensor, TfTensor, TfTensor),
    ]
