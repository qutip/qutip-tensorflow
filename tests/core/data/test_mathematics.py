import numpy as np
import tensorflow as tf
import pytest

import qutip.tests.core.data.test_mathematics as testing
from qutip.tests.core.data.test_reshape import (TestReshape,
                                                TestColumnStack,
                                                TestColumnUnstack,
                                                TestSplitColumns)

from qutip.tests.core.data.test_expect import TestExpect, TestExpectSuper
from qutip.tests.core.data.test_norm import (
    TestTraceNorm,
    TestFrobeniusNorm,
    TestL2Norm,
    TestMaxNorm,
    TestOneNorm,
)

from qutip_tensorflow.core.data import TfTensor
from qutip_tensorflow import data
from . import conftest

testing._ALL_CASES = {
    TfTensor: lambda shape: [lambda: conftest.random_tftensor(shape)],
}
testing._RANDOM = {TfTensor: lambda shape: [lambda: conftest.random_tftensor(shape)]}

print('hi')


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
    tol = 1e-3
    specialisations = [
        pytest.param(data.inner_tftensor, TfTensor, TfTensor, tf.Tensor),
    ]


class TestInnerOp(testing.TestInnerOp):
    tol = 1e-3
    specialisations = [
        pytest.param(data.inner_op_tftensor, TfTensor, TfTensor, TfTensor, tf.Tensor),
    ]


class TestTrace(testing.TestTrace):
    tol = 1e-5
    specialisations = [
        pytest.param(data.trace_tftensor, TfTensor, tf.Tensor),
    ]


class TestKron(testing.TestKron):
    tol = 1e-5
    specialisations = [
        pytest.param(data.kron_tftensor, TfTensor, TfTensor, TfTensor),
    ]


class TestMul(testing.TestMul):
    specialisations = [
        pytest.param(data.mul_tftensor, TfTensor, TfTensor),
        pytest.param(data.imul_tftensor, TfTensor, TfTensor),
    ]


class TestMatmul(testing.TestMatmul):
    tol = 1e-4
    specialisations = [
        pytest.param(data.matmul_tftensor, TfTensor, TfTensor, TfTensor),
    ]


class TestNeg(testing.TestNeg):

    specialisations = [
        pytest.param(data.neg_tftensor, TfTensor, TfTensor),
    ]


class TestReshape(TestReshape):
    specialisations = [
        pytest.param(data.reshape_tftensor, TfTensor, TfTensor),
    ]


class TestSplitColumns(TestSplitColumns):
    specialisations = [
        pytest.param(data.split_columns_tftensor, TfTensor, list),
    ]


class TestColumnUnstack(TestColumnUnstack):
    specialisations = [
        pytest.param(data.column_unstack_tftensor, TfTensor, TfTensor),
    ]


class TestColumnStack(TestColumnStack):
    specialisations = [
        pytest.param(data.column_stack_tftensor, TfTensor, TfTensor),
    ]

class TestExpect(TestExpect):
    tol = 1e-3
    specialisations = [
        pytest.param(data.expect_tftensor, TfTensor, TfTensor, tf.Tensor),
    ]


class TestExpectSuper(TestExpectSuper):
    tol = 1e-4
    specialisations = [
        pytest.param(data.expect_super_tftensor, TfTensor, TfTensor, tf.Tensor),
    ]


class TestExpm(testing.TestExpm):
    tol = 1e-5
    specialisations = [
        pytest.param(data.expm_tftensor, TfTensor, TfTensor),
    ]


class TestPow(testing.TestPow):
    rtol = 1e-4
    specialisations = [
        pytest.param(data.pow_tftensor, TfTensor, TfTensor),
    ]


class TestProject(testing.TestProject):
    tol = 1e-5
    specialisations = [
        pytest.param(data.project_tftensor, TfTensor, TfTensor),
    ]


class TestTraceNorm(TestTraceNorm):
    # this one needs larger tol because the output is larger in magnitude and
    # tol refers to absolute tolerance. This suggest that we may want to use
    # a relative tol instead.
    tol = 1e-2
    specialisations = [
        pytest.param(data.norm.trace_tftensor, TfTensor, tf.Tensor),
    ]


class TestOneNorm(TestOneNorm):
    tol = 1e-4
    specialisations = [
        pytest.param(data.norm.one_tftensor, TfTensor, tf.Tensor),
    ]


class TestL2Norm(TestL2Norm):
    tol = 1e-5
    specialisations = [
        pytest.param(data.norm.l2_tftensor, TfTensor, tf.Tensor),
    ]


class TestMaxNorm(TestMaxNorm):
    tol = 1e-5
    specialisations = [
        pytest.param(data.norm.max_tftensor, TfTensor, tf.Tensor),
    ]


class TestFrobeniusNorm(TestFrobeniusNorm):
    tol = 1e-5
    specialisations = [
        pytest.param(data.norm.frobenius_tftensor, TfTensor, tf.Tensor),
    ]
