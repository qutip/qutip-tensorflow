import qutip
from .tftensor import TfTensor
import tensorflow as tf

# Conversion function
def _tf_from_dense(dense):
    return TfTensor(dense.to_array())


def _tf_to_dense(tftensor):
    return qutip.data.Dense(tftensor.to_array(), copy=False)


def is_tftensor(data):
    return isinstance(data, tf.Tensor)


# Register the data layer
qutip.data.to.add_conversions(
    [
        (TfTensor, qutip.data.Dense, _tf_from_dense),
        (qutip.data.Dense, TfTensor, _tf_to_dense),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["tftensor", "TfTensor"], TfTensor)

qutip.data.create.add_creators(
    [
        (is_tftensor, TfTensor, 85),
    ]
)
