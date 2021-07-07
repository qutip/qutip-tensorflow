import numpy as np
import pytest
import qutip_tensorflow as qtf
import qutip as qt
from .conftest import random_tensor_dense


_dense_tensor = qtf.data.DenseTensor(random_tensor_dense((2,2)))

@pytest.mark.parametrize(['from_', 'base'], [
    pytest.param('tftensor', _dense_tensor, id='from tensorflow str'),
    pytest.param('DenseTensor', _dense_tensor, id='from DenseTensor str'),
    pytest.param(qtf.data.DenseTensor, _dense_tensor, id='from DenseTensor type'),
])
@pytest.mark.parametrize(['to_', 'dtype'], [
    pytest.param('dense', qt.data.Dense, id='to Dense str'),
    pytest.param('Dense', qt.data.Dense, id='to Dense STR'),
    pytest.param(qt.data.Dense, qt.data.Dense, id='to Dense type'),
    pytest.param('csr', qt.data.CSR, id='to CSR str'),
    pytest.param('CSR', qt.data.CSR, id='to CSR STR'),
    pytest.param(qt.data.CSR, qt.data.CSR, id='to CSR type'),
    pytest.param('tftensor', qtf.data.DenseTensor, id='to tensorflow str'),
    pytest.param('DenseTensor', qtf.data.DenseTensor, id='to tensorflow str_type'),
    pytest.param(qtf.data.DenseTensor, qtf.data.DenseTensor, id='to tensorflow type'),
])
def test_converters_qtf_to_qt(from_, base, to_, dtype):
    converter = qtf.data.to[to_, from_]
    assert isinstance(converter(base), dtype)
    converter = qtf.data.to[to_]
    assert isinstance(converter(base), dtype)
    assert isinstance(qtf.data.to(to_, base), dtype)


@pytest.mark.parametrize(['from_', 'base'], [
    pytest.param('dense', qt.data.dense.zeros(2, 2), id='from Dense str'),
    pytest.param('Dense', qt.data.dense.zeros(2, 2), id='from Dense STR'),
    pytest.param(qt.data.Dense, qt.data.dense.zeros(2, 2), id='from Dense type'),
    pytest.param('csr', qt.data.csr.zeros(2, 2), id='from CSR str'),
    pytest.param('CSR', qt.data.csr.zeros(2, 2), id='from CSR STR'),
    pytest.param(qt.data.CSR, qt.data.csr.zeros(2, 2), id='from CSR type'),
])
@pytest.mark.parametrize(['to_', 'dtype'], [
    pytest.param('tftensor', qtf.data.DenseTensor, id='to tensorflow str'),
    pytest.param('DenseTensor', qtf.data.DenseTensor, id='to tensorflow str_type'),
    pytest.param(qtf.data.DenseTensor, qtf.data.DenseTensor, id='to tensorflow type'),
])
def test_converters_qt_to_qtf(from_, base, to_, dtype):
    print(base)
    converter = qtf.data.to[to_, from_]
    assert isinstance(converter(base), dtype)
    converter = qtf.data.to[to_]
    assert isinstance(converter(base), dtype)
    assert isinstance(qtf.data.to(to_, base), dtype)
