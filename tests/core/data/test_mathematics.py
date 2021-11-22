import numpy as np
import tensorflow as tf
import pytest

import qutip.tests.core.data.test_mathematics as testing
# from qutip.tests.core.data.test_reshape import (TestReshape,
                                                # TestColumnStack,
                                                # TestColumnUnstack,
                                                # TestSplitColumns)

from qutip.tests.core.data.test_expect import TestExpect, TestExpectSuper
# from qutip.tests.core.data.test_norm import (
    # TestTraceNorm,
    # TestFrobeniusNorm,
    # TestL2Norm,
    # TestMaxNorm,
    # TestOneNorm,
# )

from qutip_tensorflow.core.data import TfTensor64, TfTensor128
from qutip_tensorflow import data
from . import conftest

testing._ALL_CASES = {
    TfTensor64: lambda shape: [lambda: conftest.random_tftensor64(shape)],
    TfTensor128: lambda shape: [lambda: conftest.random_tftensor128(shape)],
}
testing._RANDOM = {
    TfTensor64: lambda shape: [lambda: conftest.random_tftensor64(shape)],
    TfTensor128: lambda shape: [lambda: conftest.random_tftensor128(shape)],
}


class TestAdd(testing.TestAdd):
    specialisations = [
        pytest.param(data.add_tftensor, TfTensor128, TfTensor128, TfTensor128),
        pytest.param(data.add_tftensor, TfTensor64, TfTensor64, TfTensor64),
        pytest.param(data.iadd_tftensor, TfTensor128, TfTensor128, TfTensor128),
        pytest.param(data.iadd_tftensor, TfTensor64, TfTensor64, TfTensor64),
    ]


class TestSub(testing.TestSub):
    specialisations = [
        pytest.param(data.sub_tftensor, TfTensor128, TfTensor128, TfTensor128),
        pytest.param(data.sub_tftensor, TfTensor64, TfTensor64, TfTensor64),
    ]


class TestAdjoint(testing.TestAdjoint):
    specialisations = [
        pytest.param(data.adjoint_tftensor, TfTensor128, TfTensor128, TfTensor128),
        pytest.param(data.adjoint_tftensor, TfTensor64, TfTensor64, TfTensor64),
    ]


class TestConj(testing.TestConj):
    specialisations = [
        pytest.param(data.conj_tftensor, TfTensor128, TfTensor128, TfTensor128),
        pytest.param(data.conj_tftensor, TfTensor64, TfTensor64, TfTensor64),
    ]


class TestTranspose(testing.TestTranspose):
    specialisations = [
        pytest.param(data.transpose_tftensor, TfTensor128, TfTensor128, TfTensor128),
        pytest.param(data.transpose_tftensor, TfTensor64, TfTensor64, TfTensor64),
    ]


# class TestInner(testing.TestInner):
    # specialisations = [
        # pytest.param(data.inner_tftensor, TfTensor, TfTensor, tf.Tensor),
    # ]


# class TestInnerOp(testing.TestInnerOp):
    # specialisations = [
        # pytest.param(data.inner_op_tftensor, TfTensor, TfTensor, TfTensor, tf.Tensor),
    # ]


# class TestTrace(testing.TestTrace):
    # specialisations = [
        # pytest.param(data.trace_tftensor, TfTensor, tf.Tensor),
    # ]


# class TestKron(testing.TestKron):
    # specialisations = [
        # pytest.param(data.kron_tftensor, TfTensor, TfTensor, TfTensor),
    # ]


# class TestMul(testing.TestMul):
    # specialisations = [
        # pytest.param(data.mul_tftensor, TfTensor, TfTensor),
        # pytest.param(data.imul_tftensor, TfTensor, TfTensor),
    # ]


# class TestMatmul(testing.TestMatmul):
    # specialisations = [
        # pytest.param(data.matmul_tftensor, TfTensor, TfTensor, TfTensor),
    # ]


# class TestNeg(testing.TestNeg):
    # specialisations = [
        # pytest.param(data.neg_tftensor, TfTensor, TfTensor),
    # ]


# class TestReshape(TestReshape):
    # specialisations = [
        # pytest.param(data.reshape_tftensor, TfTensor, TfTensor),
    # ]


# class TestSplitColumns(TestSplitColumns):
    # specialisations = [
        # pytest.param(data.split_columns_tftensor, TfTensor, list),
    # ]


# class TestColumnUnstack(TestColumnUnstack):
    # specialisations = [
        # pytest.param(data.column_unstack_tftensor, TfTensor, TfTensor),
    # ]


# class TestColumnStack(TestColumnStack):
    # specialisations = [
        # pytest.param(data.column_stack_tftensor, TfTensor, TfTensor),
    # ]

class TestExpect(TestExpect):
    specialisations = [
        pytest.param(data.expect_tftensor, TfTensor128, TfTensor128, tf.Tensor),
        pytest.param(data.expect_tftensor, TfTensor64, TfTensor64, tf.Tensor),
    ]

class TestExpectSuper(TestExpectSuper):
    specialisations = [
        pytest.param(data.expect_super_tftensor, TfTensor128, TfTensor128, tf.Tensor),
        pytest.param(data.expect_super_tftensor, TfTensor64, TfTensor64, tf.Tensor),
    ]

class TestExpm64(testing.TestExpm):
    rtol = 1e-4
    specialisations = [
        pytest.param(data.expm_tftensor, TfTensor64, TfTensor64),
    ]

class TestExpm128(testing.TestExpm):
    specialisations = [
        pytest.param(data.expm_tftensor, TfTensor128, TfTensor128),
    ]

# class TestPow(testing.TestPow):
    # specialisations = [
        # pytest.param(data.pow_tftensor, TfTensor, TfTensor),
    # ]


# class TestProject(testing.TestProject):
    # specialisations = [
        # pytest.param(data.project_tftensor, TfTensor, TfTensor),
    # ]


# class TestTraceNorm(TestTraceNorm):
    # specialisations = [
        # pytest.param(data.norm.trace_tftensor, TfTensor, tf.Tensor),
    # ]


# class TestOneNorm(TestOneNorm):
    # specialisations = [
        # pytest.param(data.norm.one_tftensor, TfTensor, tf.Tensor),
    # ]


# class TestL2Norm(TestL2Norm):
    # specialisations = [
        # pytest.param(data.norm.l2_tftensor, TfTensor, tf.Tensor),
    # ]


# class TestMaxNorm(TestMaxNorm):
    # specialisations = [
        # pytest.param(data.norm.max_tftensor, TfTensor, tf.Tensor),
    # ]


# class TestFrobeniusNorm(TestFrobeniusNorm):
    # specialisations = [
        # pytest.param(data.norm.frobenius_tftensor, TfTensor, tf.Tensor),
    # ]
