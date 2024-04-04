import qutip
import numpy as np
import tensorflow as tf
import qutip_tensorflow as qtf
import tensorflow.experimental.numpy as tnp

def test_left_right_variable_multiplication():
    tnp.experimental_enable_numpy_behavior(dtype_conversion_mode="safe")

    variable = tf.Variable(0.01, dtype=tf.float64)
    sz= qutip.sigmaz().to('tftensor')

    result = sz*variable
    assert np.all(result.data._tf==sz.data._tf*0.01)
    result = variable*sz
    assert np.all(result.data._tf==sz.data._tf*0.01)
