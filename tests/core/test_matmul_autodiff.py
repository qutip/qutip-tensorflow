import warnings
import qutip

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # from tensorflow import Variable, GradientTape, co
    import tensorflow as tf

def test_matmul_variable_1():
    """Tests that matmul is correctly multiplied by ``Variable`` 1."""
    num = qutip.num(3, dtype='tftensor')
    state = qutip.basis(3, 2, dtype='tftensor')
    variable = tf.Variable(1, dtype=tf.complex128)
    with tf.GradientTape() as tape:
        out = qutip.data.matmul(num.data, state.data, variable)

    result = tape.gradient(out._tf, variable)
