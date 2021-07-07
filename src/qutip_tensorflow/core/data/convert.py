import qutip as qt
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

# `add_conversions` will register the data layer
qt.data.to.add_conversions([
     (DenseTensor, qt.data.Dense, _tf_from_dense),
     (qt.data.Dense, DenseTensor, _tf_to_dense),
])

# You can add user friendly name for conversion with `to` or Qobj creation functions:
qt.data.to.register_aliases(['tftensor'], DenseTensor)
qt.data.to.register_aliases(['DenseTensor'], DenseTensor)

qt.data.create.add_creators([
    (is_tftensor, DenseTensor, 85),
])

to = qt.data.to
create = qt.data.create

