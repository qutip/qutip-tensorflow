import itertools
import numpy as np
import pytest

from qutip_tensorflow.core.data import TfTensor
from qutip_tensorflow import data
from qutip.core.data import Data, Dense, CSR
import qutip.tests.core.data.test_mathematics as testing
from qutip.tests.core.data.test_norm import (
    TestTraceNorm,
    TestFrobeniusNorm,
    TestL2Norm,
    TestMaxNorm,
    TestOneNorm,
)
import tensorflow as tf


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


class TestAdjoint(testing.TestAdjoint):
    specialisations = [
        pytest.param(data.adjoint_tftensor, TfTensor, TfTensor),
    ]


class TestConj(testing.TestConj):
    specialisations = [
        pytest.param(data.conj_tftensor, TfTensor, TfTensor),
    ]


class TestTranspose(testing.TestTranspose):
    specialisations = [
        pytest.param(data.transpose_tftensor, TfTensor, TfTensor),
    ]


class TestInner(testing.TestInner):
    specialisations = [
        pytest.param(data.inner_tftensor, TfTensor, TfTensor, tf.Tensor),
    ]


class TestInnerOp(testing.TestInnerOp):
    specialisations = [
        pytest.param(data.inner_op_tftensor, TfTensor, TfTensor, TfTensor, tf.Tensor),
    ]


class TestTrace(testing.TestTrace):
    specialisations = [
        pytest.param(data.trace_tftensor, TfTensor, tf.Tensor),
    ]


class TestKron(testing.TestKron):
    specialisations = [
        pytest.param(data.kron_tftensor, TfTensor, TfTensor, TfTensor),
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


class TestExpm(testing.TestExpm):
    specialisations = [
        pytest.param(data.expm_tftensor, TfTensor, TfTensor),
    ]


class TestPow(testing.TestPow):
    specialisations = [
        pytest.param(data.pow_tftensor, TfTensor, TfTensor),
    ]


class TestProject(testing.TestProject):
    specialisations = [
        pytest.param(data.project_tftensor, TfTensor, TfTensor),
    ]


class TestTraceNorm(TestTraceNorm):
    specialisations = [
        pytest.param(data.norm.trace_tftensor, TfTensor, tf.Tensor),
    ]


class TestOneNorm(TestOneNorm):
    specialisations = [
        pytest.param(data.norm.one_tftensor, TfTensor, tf.Tensor),
    ]


class TestL2Norm(TestL2Norm):
    specialisations = [
        pytest.param(data.norm.l2_tftensor, TfTensor, tf.Tensor),
    ]


class TestMaxNorm(TestMaxNorm):
    specialisations = [
        pytest.param(data.norm.max_tftensor, TfTensor, tf.Tensor),
    ]


class TestFrobeniusNorm(TestFrobeniusNorm):
    specialisations = [
        pytest.param(data.norm.frobenius_tftensor, TfTensor, tf.Tensor),
    ]
