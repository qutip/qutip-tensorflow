import numpy as np
import pytest
import qutip
import qutip_tensorflow as qtf
from qutip_tensorflow.core.data import TfTensor128
from qutip import data
from .conftest import random_tensor_dense


_dense_tensor = TfTensor128(random_tensor_dense((2, 2)))


@pytest.mark.parametrize(
    ["base", "dtype"],
    [
        pytest.param(random_tensor_dense((2, 2)), TfTensor128, id="Tensor"),
        pytest.param(_dense_tensor, TfTensor128, id="TfTensor128"),
    ],
)
def test_create(base, dtype):
    created = qutip.data.create(base)
    assert isinstance(created, dtype)


@pytest.mark.parametrize(
    ["from_", "base"],
    [
        pytest.param("tftensor", _dense_tensor, id="from tensorflow str"),
        pytest.param("TfTensor128", _dense_tensor, id="from TfTensor str"),
        pytest.param(TfTensor128, _dense_tensor, id="from TfTensor type"),
    ],
)
@pytest.mark.parametrize(
    ["to_", "dtype"],
    [
        pytest.param("dense", data.Dense, id="to Dense str"),
        pytest.param("Dense", data.Dense, id="to Dense STR"),
        pytest.param(data.Dense, data.Dense, id="to Dense type"),
        pytest.param("csr", data.CSR, id="to CSR str"),
        pytest.param("CSR", data.CSR, id="to CSR STR"),
        pytest.param(data.CSR, data.CSR, id="to CSR type"),
        pytest.param("tftensor", TfTensor128, id="to tensorflow str"),
        pytest.param("TfTensor128", TfTensor128, id="to tensorflow str_type"),
        pytest.param(TfTensor128, TfTensor128, id="to tensorflow type"),
    ],
)
def test_converters_qtf_to_qt(from_, base, to_, dtype):
    converter = data.to[to_, from_]
    assert isinstance(converter(base), dtype)
    converter = data.to[to_]
    assert isinstance(converter(base), dtype)
    assert isinstance(data.to(to_, base), dtype)


@pytest.mark.parametrize(
    ["from_", "base"],
    [
        pytest.param("dense", data.dense.zeros(2, 2), id="from Dense str"),
        pytest.param("Dense", data.dense.zeros(2, 2), id="from Dense STR"),
        pytest.param(data.Dense, data.dense.zeros(2, 2), id="from Dense type"),
        pytest.param("csr", data.csr.zeros(2, 2), id="from CSR str"),
        pytest.param("CSR", data.csr.zeros(2, 2), id="from CSR STR"),
        pytest.param(data.CSR, data.csr.zeros(2, 2), id="from CSR type"),
    ],
)
@pytest.mark.parametrize(
    ["to_", "dtype"],
    [
        pytest.param("tftensor", TfTensor128, id="to tensorflow str"),
        pytest.param("TfTensor128", TfTensor128, id="to tensorflow str_type"),
        pytest.param(TfTensor128, TfTensor128, id="to tensorflow type"),
    ],
)
def test_converters_qt_to_qtf(from_, base, to_, dtype):
    converter = data.to[to_, from_]
    assert isinstance(converter(base), dtype)
    converter = data.to[to_]
    assert isinstance(converter(base), dtype)
    assert isinstance(data.to(to_, base), dtype)


