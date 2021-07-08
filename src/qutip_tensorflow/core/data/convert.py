import qutip
from .dense_tensor import DenseTensor
import tensorflow as tf

__all__ = ['to']

# Conversion function
def _tf_from_dense(dense):
    return DenseTensor(dense.to_array())

def _tf_to_dense(tftensor):
    return qt.data.Dense(tftensor.to_array(), copy=False)

def is_tftensor(data):
    return isinstance(data, tf.Tensor)

# Register the data layer
qutip.data.to.add_conversions([
     (DenseTensor, qt.data.Dense, _tf_from_dense),
     (qt.data.Dense, DenseTensor, _tf_to_dense),
])

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(['tftensor', 'DenseTensor'], DenseTensor)

qutip.data.create.add_creators([
    (is_tftensor, DenseTensor, 85),
])
