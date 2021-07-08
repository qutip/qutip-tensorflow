import qutip
from .dense_tensor import DenseTensor
import tensorflow as tf

# Conversion function
def _tf_from_dense(dense):
    return DenseTensor(dense.to_array())

def _tf_to_dense(tftensor):
    return qutip.data.Dense(tftensor.to_array(), copy=False)

def is_tftensor(data):
    return isinstance(data, tf.Tensor)

# Register the data layer
qutip.data.to.add_conversions([
     (DenseTensor, qutip.data.Dense, _tf_from_dense),
     (qutip.data.Dense, DenseTensor, _tf_to_dense),
])

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(['tftensor', 'DenseTensor'], DenseTensor)

qutip.data.create.add_creators([
    (is_tftensor, DenseTensor, 85),
])
