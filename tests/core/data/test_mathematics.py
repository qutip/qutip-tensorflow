import numpy as np
import tensorflow as tf
import pytest

# import qutip.tests.core.data.test_mathematics as testing
# from qutip.tests.core.data.test_reshape import (TestReshape,
                                                # TestColumnStack,
                                                # TestColumnUnstack,
                                                # TestSplitColumns)

# from qutip.tests.core.data.test_expect import TestExpect, TestExpectSuper
# from qutip.tests.core.data.test_norm import (
    # TestTraceNorm,
    # TestFrobeniusNorm,
    # TestL2Norm,
    # TestMaxNorm,
    # TestOneNorm,
# )

import qutip.tests.core.data as testing
from qutip_tensorflow.core.data import TfTensor64, TfTensor128
from qutip_tensorflow import data
from . import conftest

testing.test_mathematics._ALL_CASES = {
    TfTensor64: lambda shape: [lambda: conftest.random_tftensor64(shape)],
    TfTensor128: lambda shape: [lambda: conftest.random_tftensor128(shape)],
}
testing.test_mathematics._RANDOM = {
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


class TestInner128(testing.TestInner):
    specialisations = [
        pytest.param(data.inner_tftensor, TfTensor128, TfTensor128, tf.Tensor),
    ]


class TestInner64(testing.TestInner):
    rtol = 1e-6
    specialisations = [
        pytest.param(data.inner_tftensor, TfTensor64, TfTensor64, tf.Tensor),
    ]


class TestInnerOp128(testing.TestInnerOp):
    specialisations = [
        pytest.param(data.inner_op_tftensor, TfTensor128, TfTensor128,
                     TfTensor128, tf.Tensor),
    ]

class TestInnerOp64(testing.TestInnerOp):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.inner_op_tftensor, TfTensor64, TfTensor64,
                     TfTensor64, tf.Tensor),
    ]

class TestTrace128(testing.TestTrace):
    specialisations = [
        pytest.param(data.trace_tftensor, TfTensor128, tf.Tensor),
    ]

class TestTrace64(testing.TestTrace):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.trace_tftensor, TfTensor64, tf.Tensor),
    ]

class TestKron128(testing.TestKron):
    specialisations = [
        pytest.param(data.kron_tftensor, TfTensor128, TfTensor128, TfTensor128),
    ]

class TestKron64(testing.TestKron):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.kron_tftensor, TfTensor64, TfTensor64, TfTensor64),
    ]

class TestMul(testing.TestMul):
    specialisations = [
        pytest.param(data.mul_tftensor, TfTensor64, TfTensor64),
        pytest.param(data.imul_tftensor, TfTensor64, TfTensor64),
        pytest.param(data.mul_tftensor, TfTensor128, TfTensor128),
        pytest.param(data.imul_tftensor, TfTensor128, TfTensor128),
    ]


class TestMatmul128(testing.TestMatmul):
    specialisations = [
        pytest.param(data.matmul_tftensor, TfTensor128, TfTensor128,
                     TfTensor128),
    ]

class TestMatmul64(testing.TestMatmul):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.matmul_tftensor, TfTensor64, TfTensor64,
                     TfTensor64),
    ]

class TestNeg(testing.TestNeg):
    specialisations = [
        pytest.param(data.neg_tftensor, TfTensor128, TfTensor128),
        pytest.param(data.neg_tftensor, TfTensor64, TfTensor64),
    ]


class TestReshape(testing.TestReshape):
    specialisations = [
        pytest.param(data.reshape_tftensor, TfTensor128, TfTensor128),
        pytest.param(data.reshape_tftensor, TfTensor64, TfTensor64),
    ]


class TestSplitColumns(testing.TestSplitColumns):
    specialisations = [
        pytest.param(data.split_columns_tftensor, TfTensor128, list),
        pytest.param(data.split_columns_tftensor, TfTensor64, list),
    ]


class TestColumnUnstack(testing.TestColumnUnstack):
    specialisations = [
        pytest.param(data.column_unstack_tftensor, TfTensor128, TfTensor128),
        pytest.param(data.column_unstack_tftensor, TfTensor64, TfTensor64),
    ]


class TestColumnStack(testing.TestColumnStack):
    specialisations = [
        pytest.param(data.column_stack_tftensor, TfTensor128, TfTensor128),
        pytest.param(data.column_stack_tftensor, TfTensor64, TfTensor64),
    ]

class TestExpect64(testing.TestExpect):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.expect_tftensor, TfTensor64, TfTensor64, tf.Tensor),
    ]

class TestExpect128(testing.TestExpect):
    specialisations = [
        pytest.param(data.expect_tftensor, TfTensor128, TfTensor128, tf.Tensor),
    ]

class TestExpectSuper128(testing.TestExpectSuper):
    specialisations = [
        pytest.param(data.expect_super_tftensor, TfTensor128, TfTensor128, tf.Tensor),
    ]

class TestExpectSuper64(testing.TestExpectSuper):
    rtol = 1e-5
    specialisations = [
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

class TestPow128(testing.TestPow):
    specialisations = [
        pytest.param(data.pow_tftensor, TfTensor128, TfTensor128),
    ]

class TestPow64(testing.TestPow):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.pow_tftensor, TfTensor64, TfTensor64),
    ]

class TestProject128(testing.TestProject):
    specialisations = [
        pytest.param(data.project_tftensor, TfTensor128, TfTensor128),
    ]

class TestProject64(testing.TestProject):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.project_tftensor, TfTensor64, TfTensor64),
    ]

class TestTraceNorm128(testing.TestTraceNorm):
    specialisations = [
        pytest.param(data.norm.trace_tftensor, TfTensor128, tf.Tensor),
    ]

class TestTraceNorm64(testing.TestTraceNorm):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.norm.trace_tftensor, TfTensor64, tf.Tensor),
    ]

class TestOneNorm128(testing.TestOneNorm):
    specialisations = [
        pytest.param(data.norm.one_tftensor, TfTensor128, tf.Tensor),
    ]

class TestOneNorm64(testing.TestOneNorm):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.norm.one_tftensor, TfTensor64, tf.Tensor),
    ]

class TestL2Norm128(testing.TestL2Norm):
    specialisations = [
        pytest.param(data.norm.l2_tftensor, TfTensor128, tf.Tensor),
    ]

class TestL2Norm64(testing.TestL2Norm):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.norm.l2_tftensor, TfTensor64, tf.Tensor),
    ]

class TestMaxNorm(testing.TestMaxNorm):
    specialisations = [
        pytest.param(data.norm.max_tftensor, TfTensor128, tf.Tensor),
        pytest.param(data.norm.max_tftensor, TfTensor64, tf.Tensor),
    ]


class TestFrobeniusNorm128(testing.TestFrobeniusNorm):
    specialisations = [
        pytest.param(data.norm.frobenius_tftensor, TfTensor128, tf.Tensor),
    ]

class TestFrobeniusNorm64(testing.TestFrobeniusNorm):
    rtol = 1e-5
    specialisations = [
        pytest.param(data.norm.frobenius_tftensor, TfTensor64, tf.Tensor),
    ]
