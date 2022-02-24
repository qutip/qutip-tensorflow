import warnings
import qutip
import pytest

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # from tensorflow import Variable, GradientTape, co
    import tensorflow as tf

@pytest.mark.parametrize("operation, expected", [(qutip.data.matmul, 5),
                                                 (qutip.data.add, 3)])
def test_scale_1(operation, expected):
    """Tests that matmul and add is correctly multiplied by ``Variable`` 1.
    This functions have fast paths for the case of scale=1 as it is the most
    common one."""
    left = qutip.num(3, dtype='tftensor')
    right = qutip.num(3, dtype='tftensor')
    variable = tf.Variable(1, dtype=tf.complex128)
    with tf.GradientTape() as tape:
        out = operation(left.data, right.data, variable)

    result = tape.gradient(out._tf, variable)
    assert result == expected
